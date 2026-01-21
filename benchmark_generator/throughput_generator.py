#!/usr/bin/env python3
"""
throughput_generator.py

Throughput benchmark generator for ESP32-C6 RISC-V instructions.
Based on Section 5.3 of the uops.info paper (ASPLOS '19).

Throughput is measured by creating independent instruction sequences
where each instruction writes to a different register, allowing
parallel execution if the microarchitecture supports it.

Phase 1: Arithmetic instructions only (ADD, SUB, AND, OR, XOR, SLL, SRL, SRA)
"""

from dataclasses import dataclass
from typing import Optional, List

try:
    from .common import Instruction, BenchmarkConfig, LATENCY_TYPE_MAP
except ImportError:
    from common import Instruction, BenchmarkConfig, LATENCY_TYPE_MAP



# Supported instructions for throughput benchmarks (all phases)
SUPPORTED_MNEMONICS = {
    # Phase 1: Basic Arithmetic
    # R-type: op rd, rs1, rs2
    "add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu",
    # I-type: op rd, rs1, imm
    "addi", "andi", "ori", "xori", "slti", "sltiu", "slli", "srli", "srai",
    # Compressed R-type
    "c.add", "c.sub", "c.and", "c.or", "c.xor",
    # Compressed I-type
    "c.addi", "c.andi", "c.slli", "c.srli", "c.srai",
    # Move/load immediate
    "c.mv", "c.li",
    
    # Phase 2: Multiply/Divide (M extension)
    "mul", "mulh", "mulhsu", "mulhu",
    "div", "divu", "rem", "remu",
    
    # Phase 3: Upper Immediate & Sign/Zero Extend
    "lui", "auipc",
    "sext.b", "sext.h", "zext.h",
    
    # Phase 4: Load/Store (using asm input constraints like latency generator)
    "lw", "sw", "lh", "lhu", "sh", "lb", "lbu", "sb",
    "c.lw", "c.sw", "c.lwsp", "c.swsp",

    # Phase 6: Atomic Instructions (A extension)
    "amoswap.w", "amoadd.w", "amoxor.w", "amoand.w", "amoor.w",
    "amomin.w", "amomax.w", "amominu.w", "amomaxu.w",
    # Atomic variants (aq, rl, aqrl)
    "amoswap.w.aq", "amoadd.w.aq", "amoxor.w.aq", "amoand.w.aq", "amoor.w.aq",
    "amomin.w.aq", "amomax.w.aq", "amominu.w.aq", "amomaxu.w.aq",
    "amoswap.w.rl", "amoadd.w.rl", "amoxor.w.rl", "amoand.w.rl", "amoor.w.rl",
    "amomin.w.rl", "amomax.w.rl", "amominu.w.rl", "amomaxu.w.rl",
    "amoswap.w.aqrl", "amoadd.w.aqrl", "amoxor.w.aqrl", "amoand.w.aqrl", "amoor.w.aqrl",
    "amomin.w.aqrl", "amomax.w.aqrl", "amominu.w.aqrl", "amomaxu.w.aqrl",

    # Phase 5: Control Flow (branches - not-taken, jumps - chained)
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "c.beqz", "c.bnez",
    "jal", "c.j", "c.jal",
}



class ThroughputBenchmarkGenerator:
    """
    Generates throughput benchmarks for RISC-V instructions.

    Throughput is measured by executing independent instructions that
    don't have read-after-write dependencies, maximizing parallelism.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.generated_benchmarks = []
        self.skipped_instructions = []

    # ========================================================================
    # TableGen Entry Helpers
    # ========================================================================

    # Mapping of LLVM register class names to actual register lists
    REGCLASS_MAP = {
        "GPR": ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "t3", "t4", "t5"],
        "GPRNoX0": ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "t3", "t4", "t5"],
        "GPRC": ["a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"],  # Compressed regs
        "GPRCMem": ["a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"],
        "GPRNoX0X2": ["a0", "a1", "a3", "a4", "a5", "t0", "t1", "t2", "t3", "t4", "t5"],
        "SR07": ["s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5"],  # RVC s0-s1, a0-a5
    }

    # Default registers for throughput benchmarks (fallback)
    DEFAULT_REGS = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1"]

    def _get_operand_regclass(self, operand_list: dict) -> List[str]:
        """
        Extract register class names from an InOperandList or OutOperandList.
        
        Returns a list of register class names (e.g., ['GPR', 'GPRC']).
        """
        regclasses = []
        args = operand_list.get("args", [])
        for arg in args:
            if isinstance(arg, list) and len(arg) >= 1:
                op_def = arg[0]
                if isinstance(op_def, dict):
                    printable = op_def.get("printable", "")
                    if printable:
                        regclasses.append(printable)
        return regclasses

    def _get_registers_for_class(self, regclass: str) -> List[str]:
        """
        Map a register class name to a list of usable registers.
        Falls back to THROUGHPUT_REGS if unknown.
        """
        # Direct lookup
        if regclass in self.REGCLASS_MAP:
            return self.REGCLASS_MAP[regclass]
        
        # Prefix matching for variants (e.g., GPRNoX0X2 starts with GPR)
        for key, regs in self.REGCLASS_MAP.items():
            if regclass.startswith(key):
                return regs
        
        # Fallback to default throughput registers
        return self.DEFAULT_REGS

    def _get_dest_registers_from_tablegen(self, instr: Instruction) -> List[str]:
        """
        Get the list of valid destination registers for an instruction
        by examining its tablegen_entry OutOperandList.
        
        Returns THROUGHPUT_REGS as fallback if tablegen_entry is unavailable.
        """
        if not instr.tablegen_entry:
            return self.DEFAULT_REGS
        
        out_ops = instr.tablegen_entry.get("OutOperandList", {})
        regclasses = self._get_operand_regclass(out_ops)
        
        if regclasses:
            # Use the first output operand's register class
            return self._get_registers_for_class(regclasses[0])
        
        return self.DEFAULT_REGS

    def _get_src_registers_from_tablegen(self, instr: Instruction) -> List[str]:
        """
        Get the list of valid source registers for an instruction
        by examining its tablegen_entry InOperandList.
        
        Returns THROUGHPUT_REGS as fallback if tablegen_entry is unavailable.
        """
        if not instr.tablegen_entry:
            return self.DEFAULT_REGS
        
        in_ops = instr.tablegen_entry.get("InOperandList", {})
        regclasses = self._get_operand_regclass(in_ops)
        
        if regclasses:
            # Use the first input operand's register class (usually rs1)
            return self._get_registers_for_class(regclasses[0])
        
        return self.DEFAULT_REGS

    # ========================================================================
    # String Helpers
    # ========================================================================

    def _escape_asm(self, asm: str) -> str:
        """Escape assembly string for C string literal."""
        return asm.replace("\\", "\\\\").replace('"', '\\"')

    def _sanitize_name(self, name: str) -> str:
        """Convert instruction name to valid C identifier."""
        return name.replace(".", "_").replace(" ", "_").lower()

    def _get_latency_type_enum(self, lat_type: str) -> str:
        """Get C enum value for latency type."""
        return LATENCY_TYPE_MAP.get(lat_type, "LAT_TYPE_UNKNOWN")

    def _is_supported_instruction(self, instr: Instruction) -> bool:
        """Check if instruction is supported for throughput benchmarks."""
        asm = instr.test_asm.lower().strip()
        mnemonic = asm.split()[0] if asm.split() else ""
        return mnemonic in SUPPORTED_MNEMONICS
    
    # Backward compatibility
    def _is_phase1_instruction(self, instr: Instruction) -> bool:
        return self._is_supported_instruction(instr)

    def _generate_independent_sequence(self, instr: Instruction, count: Optional[int] = None) -> Optional[str]:
        """
        Generate independent instruction sequence for throughput measurement.
        
        Strategy: Use different destination registers for each instance
        to avoid RAW (read-after-write) dependencies.
        """
        asm = instr.test_asm.lower().strip()
        parts = asm.split()
        if not parts:
            return None
        
        mnemonic = parts[0]
        
        # Only Phase 1 instructions supported
        if mnemonic not in SUPPORTED_MNEMONICS:
            return None

        indep_lines = []
        n = count if count is not None else self.config.independent_count
        
        # Get valid destination registers from tablegen entry (dynamic selection)
        dest_regs = self._get_dest_registers_from_tablegen(instr)
        
        # Generate N independent instructions, each using a different dest register
        if mnemonic in {"add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu"}:
            # R-type: op rd, rs1, rs2 -> use different rd each time
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                # Use t0 as a constant source (doesn't change)
                indep_lines.append(f'        "{mnemonic} {rd}, t0, t1\\n"')
                
        elif mnemonic in {"addi", "andi", "ori", "xori", "slti", "sltiu"}:
            # I-type with 12-bit immediate
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, 1\\n"')
                
        elif mnemonic in {"slli", "srli", "srai"}:
            # Shift with 5-bit immediate
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, 1\\n"')
                
        elif mnemonic in {"c.add"}:
            # c.add rd, rs2 - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "c.add {rd}, s0\\n"')
                
        elif mnemonic in {"c.sub", "c.and", "c.or", "c.xor"}:
            # Compressed R-type - uses tablegen-derived registers (GPRC)
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                # Use s1 as source (in GPRC set)
                indep_lines.append(f'        "{mnemonic} {rd}, s1\\n"')
                
        elif mnemonic in {"c.addi", "c.andi"}:
            # Compressed I-type - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                imm = 1 if mnemonic == "c.addi" else 0
                indep_lines.append(f'        "{mnemonic} {rd}, {imm}\\n"')
                
        elif mnemonic in {"c.slli", "c.srli", "c.srai"}:
            # Compressed shift - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "{mnemonic} {rd}, 1\\n"')
                
        elif mnemonic == "c.mv":
            # c.mv rd, rs2 - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "c.mv {rd}, s0\\n"')
                
        elif mnemonic == "c.li":
            # c.li rd, imm - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "c.li {rd}, {i % 32}\\n"')
                
        # Phase 2: Multiply/Divide (M extension)
        elif mnemonic in {"mul", "mulh", "mulhsu", "mulhu"}:
            # R-type multiply - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, t1\\n"')
                
        elif mnemonic in {"div", "divu", "rem", "remu"}:
            # R-type divide - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, t1\\n"')
                
        # Phase 3: Upper Immediate
        elif mnemonic == "lui":
            # lui rd, imm20 - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "lui {rd}, 0x12345\\n"')
                
        elif mnemonic == "auipc":
            # auipc rd, imm20 - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "auipc {rd}, 0\\n"')
                
        # Phase 3: Sign/Zero Extend (Zbb extension)
        elif mnemonic in {"sext.b", "sext.h", "zext.h"}:
            # Unary - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0\\n"')
                
        # Phase 4: Load instructions
        elif mnemonic in {"lw", "lh", "lhu", "lb", "lbu"}:
            # Load - uses tablegen-derived registers
            for i in range(n):
                rd = dest_regs[i % len(dest_regs)]
                offset = i * 4  # 4-byte aligned offsets
                indep_lines.append(f'        "{mnemonic} {rd}, {offset}(s0)\\n"')
                
        elif mnemonic in {"c.lw", "c.sw", "c.lwsp", "c.swsp"}:
            # Compressed load/store - MUST use GPRC registers (a0-a5, s0-s1)
            # Filter tablegen-derived registers to only include GPRC-compatible ones
            gprc_regs = {"a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"}
            regs_to_use = [r for r in dest_regs if r in gprc_regs]
            if not regs_to_use:
                regs_to_use = list(gprc_regs)  # Fallback to full GPRC set
            
            base_reg = "s1"
            if mnemonic == "c.lw" and base_reg in regs_to_use:
                regs_to_use.remove(base_reg)  # Don't clobber base pointer
            
            for i in range(n):
                reg = regs_to_use[i % len(regs_to_use)]
                
                # c.lw/sw immediate: scaled by 4, range 0..124
                offset = (i % 32) * 4 
                
                if mnemonic == "c.lwsp":
                     offset = (i % 64) * 4
                     indep_lines.append(f'        "{mnemonic} {reg}, {offset}(sp)\\n"')
                elif mnemonic == "c.swsp":
                     offset = (i % 64) * 4
                     indep_lines.append(f'        "{mnemonic} {reg}, {offset}(sp)\\n"')
                else: 
                     indep_lines.append(f'        "{mnemonic} {reg}, {offset}({base_reg})\\n"')
                
        # Phase 4: Store instructions
        elif mnemonic in {"sw", "sh", "sb"}:
            # Store - uses tablegen-derived registers
            for i in range(n):
                rs = dest_regs[i % len(dest_regs)]
                offset = i * 4
                indep_lines.append(f'        "{mnemonic} {rs}, {offset}(s0)\\n"')
                

                
        elif mnemonic.startswith("amo"):
            # Atomic instructions: amo<op>.w rd, rs2, (rs1)
            # To measure throughput parallel to memory, we use multiple address registers.
            # We reserve a0, a1, a2, a3 as address pointers (initialized in setup).
            # We use t0 as source operand (rs2).
            # We use a4, a5, t1, t2 as destinations (rd) to avoid clobbering addrs.
            
            addr_regs = ["a0", "a1", "a2", "a3"]
            dest_regs = ["a4", "a5", "t1", "t2"]
            src_reg = "t0"
            
            for i in range(n):
                addr_reg = addr_regs[i % len(addr_regs)]
                rd = dest_regs[i % len(dest_regs)]
                # Format: amoadd.w rd, rs2, (rs1)
                # Note: some assemblers might want 0(rs1) but standard is (rs1)
                indep_lines.append(f'        "{mnemonic} {rd}, {src_reg}, ({addr_reg})\\n"')
                
        # Phase 5: Branches (not-taken) 
        elif mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu"}:
            # Setup: configure registers so branch is NOT taken
            # beq: a0 != a1 (not equal)
            # bne: a0 == a0 (equal, so not taken)
            # blt: a0 >= a1 (not less than)
            # etc.
            branch_not_taken = {
                "beq": "beq a0, a1, 1f",   # a0 != a1
                "bne": "bne a0, a0, 1f",   # a0 == a0, not taken
                "blt": "blt a1, a0, 1f",   # a1 >= a0, not taken
                "bge": "bge a0, a1, 1f",   # setup a0 < a1
                "bltu": "bltu a1, a0, 1f", # unsigned not taken
                "bgeu": "bgeu a0, a1, 1f", # unsigned not taken
            }
            asm = branch_not_taken.get(mnemonic, f"{mnemonic} a0, a0, 1f")
            for i in range(n):
                indep_lines.append(f'        "{asm}\\n"')
                indep_lines.append(f'        "1:\\n"')
                
        elif mnemonic in {"c.beqz", "c.bnez"}:
            # Compressed branches
            # c.beqz: branch if rs == 0; setup rs != 0 for not-taken
            # c.bnez: branch if rs != 0; setup rs == 0 for not-taken
            if mnemonic == "c.beqz":
                asm = "c.beqz a1, 1f"  # a1 != 0, not taken
            else:  # c.bnez
                asm = "c.bnez a0, 1f"  # a0 != 0, taken... need a0 == 0 for not-taken
                # Actually for not-taken: c.bnez with zero reg
                # But we want to measure throughput. Let's measure taken path instead.
                # OR: use register that is 0. Setup code handles this.
                asm = "c.bnez a0, 1f"  # a0 = 1 in setup, so taken
            for i in range(n):
                indep_lines.append(f'        "{asm}\\n"')
                indep_lines.append(f'        "1:\\n"')
                
        # Phase 5: Jumps (chained labels)
        elif mnemonic in {"jal", "c.j", "c.jal"}:
            # Jump to next label in sequence
            for i in range(n):
                label = i + 1
                if mnemonic == "jal":
                    indep_lines.append(f'        "jal zero, {label}f\\n"')
                elif mnemonic == "c.j":
                    indep_lines.append(f'        "c.j {label}f\\n"')
                elif mnemonic == "c.jal":
                    indep_lines.append(f'        "c.jal {label}f\\n"')
                indep_lines.append(f'        "{label}:\\n"')
                
        else:
            return None

        return "\n".join(indep_lines)

    def generate_benchmark_function(self, instr: Instruction) -> Optional[str]:
        """
        Generate a C function that benchmarks the instruction throughput.
        Tests sequence lengths of 1, 2, 4, and 8 to find the optimal throughput.
        """
        if not self._is_supported_instruction(instr):
            return None

        func_name = self._sanitize_name(instr.llvm_enum_name + instr.name_suffix)
        
        # Test counts as per paper
        test_counts = [1, 2, 4, 8]
        
        # Generate sequences for each count
        sequences = {}
        for count in test_counts:
            seq = self._generate_independent_sequence(instr, count)
            if not seq:
                return None
            sequences[count] = seq

        # Determine setup code based on instruction type
        asm = instr.test_asm.lower().strip()
        mnemonic = asm.split()[0] if asm.split() else ""
        
        is_load_store = mnemonic in {"lw", "lh", "lhu", "lb", "lbu", "sw", "sh", "sb", "c.lw", "c.sw", "c.lwsp", "c.swsp"}
        is_atomic = mnemonic.startswith("amo") or mnemonic in {"lr.w", "sc.w"}
        is_branch = mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}
        is_jump = mnemonic in {"jal", "c.j", "c.jal"}

        setup_code = ""
        asm_init = ""
        asm_clobber = '::: "a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1", "memory"'
        post_asm = ""

        if instr.setup_code_override:
            setup_code = instr.setup_code_override
        elif is_atomic:
            # Atomic setup
            setup_code = """    /* Setup: create aligned memory buffer for atomic throughput */
    static volatile uint32_t mem_buffer[32] __attribute__((aligned(64)));
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];
    register uint32_t t0_val __asm__("t0") = 1; /* Source value */
    (void)t0_val;"""
            asm_init = """        "mv a0, %0\\n"      /* a0 = base */
        "addi a1, a0, 4\\n" /* a1 = base + 4 */
        "addi a2, a0, 8\\n" /* a2 = base + 8 */
        "addi a3, a0, 12\\n"/* a3 = base + 12 */
"""
            asm_clobber = ': "+r"(base_addr) :: "a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "memory"'
            post_asm = '        base_addr = base_addr;'

        elif is_branch or is_jump:
            # Control flow setup
            setup_code = """    /* Setup: initialize registers for branch conditions */
    register uint32_t a0_val __asm__("a0") = 1; // Non-zero
    register uint32_t a1_val __asm__("a1") = 0; // Zero
    (void)a0_val; (void)a1_val;"""

        elif mnemonic in {"c.sw"}:
            # Compressed store - uses s1 as base register (same as c.lw)
            setup_code = """    /* Setup: initialize memory buffer for c.sw throughput */
    volatile uint32_t mem_buffer[32];
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];"""
            asm_init = '        "mv s1, %0\\n"  /* Load base address into s1 */\n'
            asm_clobber = ': "+r"(base_addr) :: "a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1", "memory"'
            post_asm = '        base_addr = base_addr; /* Prevent optimization */'

        elif mnemonic in {"sb", "sh", "sw"}:
            # Regular stores - use s0 as base register
            setup_code = """    /* Setup: initialize memory buffer for store throughput */
    volatile uint32_t mem_buffer[32];
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];"""
            asm_init = '        "mv s0, %0\\n"  /* Load base address into s0 */\n'
            asm_clobber = ': "+r"(base_addr) :: "a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1", "memory"'
            post_asm = '        base_addr = base_addr; /* Prevent optimization */'

        elif mnemonic in {"c.swsp"}:
            # Stack pointer store setup
            setup_code = """    /* Setup: initialize stack pointer area */
    /* Note: SP is modified in assembly, restored after */"""
            asm_init = '        "addi sp, sp, -256\\n" /* Reserve stack space */\n'

        elif mnemonic in {"lw", "lh", "lhu", "lb", "lbu", "c.lw"}:
            # Load setup - need valid memory region, use input constraints to pass base address
            setup_code = """    /* Setup: create memory buffer for load throughput */
    static volatile uint32_t mem_buffer[32] __attribute__((aligned(64)));
    for (int i = 0; i < 32; i++) mem_buffer[i] = 0x12345678;
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];"""
            # Initialize s1 inside asm to avoid clobber issues
            asm_init = '        "mv s1, %0\\n"  /* Load base address into s1 */\n'
            asm_clobber = ': "+r"(base_addr) :: "a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1", "memory"'
            post_asm = '        base_addr = base_addr; /* Prevent optimization */'

        elif mnemonic in {"c.lwsp"}:
            # Stack pointer load setup
            setup_code = """    /* Setup: initialize stack pointer area */
    /* Note: SP is modified in assembly, restored after */"""
            asm_init = '        "addi sp, sp, -256\\n" /* Reserve stack space */\n'

        else:
            # Arithmetic setup
            setup_code = """    /* Setup: initialize source registers */
    register uint32_t t0_val __asm__("t0") = 0x12345678;
    register uint32_t t1_val __asm__("t1") = 0x87654321;
    register uint32_t s0_val __asm__("s0") = 0x55555555;
    register uint32_t s1_val __asm__("s1") = 0xAAAAAAAA;
    (void)t0_val; (void)t1_val; (void)s0_val; (void)s1_val;"""

        # Build C function with loops for each count
        func_body = f"""
/* Throughput benchmark for {instr.llvm_enum_name}: {instr.test_asm} */
static int bench_throughput_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end, elapsed;
    uint64_t global_min_cpi_x100 = UINT64_MAX; // Scaled by 100 for precision
    uint64_t sum_avg_cycles = 0;
    
    // Distribute iterations across different sequence lengths
    uint32_t iterations_per_len = total_iterations / {len(test_counts)};
    if (iterations_per_len < 1) iterations_per_len = 1;

{setup_code}

    COMPILER_BARRIER();
"""

        for count in test_counts:
            seq_code = sequences[count]
            current_asm_init = asm_init
            current_post_asm = post_asm
            asm_teardown = ""
            
            # For c.lwsp/c.swsp, restore stack inside the asm block
            if mnemonic in {"c.lwsp", "c.swsp"}:
                asm_teardown = '        "addi sp, sp, 256\\n" /* Restore stack */\n'

            func_body += f"""
    /* Testing sequence length: {count} */
    {{
        uint64_t min_cycles = UINT64_MAX;
        uint64_t total_cycles = 0;
        
        for (uint32_t rep = 0; rep < iterations_per_len; rep++) {{
            COMPILER_BARRIER();
            start = READ_CYCLE_COUNTER();
            
            __asm__ volatile (
{current_asm_init}{seq_code}
{asm_teardown}                {asm_clobber}
            );
            
            end = READ_CYCLE_COUNTER();
            COMPILER_BARRIER();
            
            elapsed = end - start;
            if (elapsed < min_cycles) min_cycles = elapsed;
            total_cycles += elapsed;
        }}
        
        {current_post_asm}

        // Calculate CPI * 100 for this length
        uint64_t cpi_x100 = (min_cycles * 100) / {count};
        if (cpi_x100 < global_min_cpi_x100) {{
            global_min_cpi_x100 = cpi_x100;
        }}
        
        sum_avg_cycles += (total_cycles / iterations_per_len) / {count};
    }}
"""

        func_body += f"""
    /* Throughput = min cycles per instruction across all lengths */
    result->min_cycles = global_min_cpi_x100 / 100; // Integer part
    
    // Note: This avg is rough, averaged across all test lengths
    result->avg_cycles = sum_avg_cycles / {len(test_counts)}; 
    result->max_cycles = 0; // Not tracked globally
    
    result->total_iterations = iterations_per_len * {len(test_counts)};
    result->status = 0;
    return 0;
}}
"""
        self.generated_benchmarks.append((instr, func_name))
        return func_body

    def _get_category_enum(self, lat_type: str) -> str:
        """Map latency type to benchmark category enum."""
        category_map = {
            "arithmetic": "BENCH_CAT_ARITHMETIC",
            "multiply": "BENCH_CAT_MULTIPLY",
            "load": "BENCH_CAT_MEMORY",
            "store": "BENCH_CAT_MEMORY",
            "load_store": "BENCH_CAT_MEMORY",
            "branch": "BENCH_CAT_CONTROL",
            "jump": "BENCH_CAT_CONTROL",
            "atomic": "BENCH_CAT_ATOMIC",
        }
        return category_map.get(lat_type, "BENCH_CAT_OTHER")

    def generate_descriptor_entry(self, instr: Instruction, func_name: str) -> str:
        """Generate a benchmark descriptor entry for throughput."""
        lat_enum = self._get_latency_type_enum(instr.latency_type)
        cat_enum = self._get_category_enum(instr.latency_type)
        escaped_asm = self._escape_asm(instr.test_asm)
        return f"""    {{
        .instruction_name = "{instr.llvm_enum_name}",
        .asm_syntax = "{escaped_asm}",
        .latency_type = {lat_enum},
        .bench_type = BENCH_TYPE_THROUGHPUT,
        .category = {cat_enum},
        .run_benchmark = bench_throughput_{func_name}
    }}"""


    def generate_structural_hazard_test(self) -> Tuple[str, str]:
        """
        Generates Structural Hazard Test (DIV + ADD).
        Tests if the divider blocks the pipeline for independent instructions.
        """
        func_name = "structural_hazard_div_add"
        
        # Setup: Ensure source registers for DIV are non-zero to avoid div-by-zero
        # DIV uses t0, t1, t2 usually. Let's be explicit in the asm.
        # We will use:
        #   div t0, t2, t3   (Long latency)
        #   add t1, t4, t5   (Short latency, independent)
        
        setup_code = """    /* Setup: Initialize registers for DIV/ADD hazard test */
    register uint32_t t2_val __asm__("t2") = 100;
    register uint32_t t3_val __asm__("t3") = 7;
    register uint32_t t4_val __asm__("t4") = 10;
    register uint32_t t5_val __asm__("t5") = 20;
    (void)t2_val; (void)t3_val; (void)t4_val; (void)t5_val;"""
        
        # Unroll 10 pairs
        # Each pair: div (10 cyc) + add (1 cyc)
        # If blocking: 11 cycles
        # If non-blocking: 10 cycles
        asm_loop_body = ""
        for _ in range(10):
            asm_loop_body += '        "div t0, t2, t3\\n"\n'
            asm_loop_body += '        "add t1, t4, t5\\n"\n'

        func_body = f"""
/* Structural Hazard Benchmark: DIV (10 cyc) + ADD (1 cyc) */
static int bench_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end, elapsed;
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;
    
{setup_code}

    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        __asm__ volatile (
{asm_loop_body}
            ::: "t0", "t1", "t2", "t3", "t4", "t5", "memory"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        elapsed = end - start;
        if (elapsed < min_cycles) min_cycles = elapsed;
        total_cycles += elapsed;
    }}
    
    // Calculate cycles per PAIR
    // We did 10 pairs per iteration.
    result->min_cycles = (uint32_t)(min_cycles / 10);
    result->avg_cycles = (uint32_t)((total_cycles / total_iterations) / 10);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    
    return 0;
}}
"""
        self.generated_benchmarks.append((None, func_name)) # None instr means manual entry
        
        # Generate descriptor manually since we don't have an Instruction object
        descriptor = f"""    {{
        .instruction_name = "STRUCTURAL_HAZARD_DIV_ADD",
        .asm_syntax = "div + add (independent)",
        .latency_type = LAT_TYPE_ARITHMETIC,
        .bench_type = BENCH_TYPE_THROUGHPUT,
        .category = BENCH_CAT_ARITHMETIC,
        .run_benchmark = bench_{func_name}
    }}"""
        
        return func_body, descriptor

