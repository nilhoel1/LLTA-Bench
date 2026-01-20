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

import re
from dataclasses import dataclass
from typing import Optional, List

try:
    from .common import Instruction, BenchmarkConfig, LATENCY_TYPE_MAP
except ImportError:
    from common import Instruction, BenchmarkConfig, LATENCY_TYPE_MAP


# Registers available for throughput benchmarks (avoid special registers)
# Using a0-a5 and t0-t5 for independent operations
THROUGHPUT_REGS = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1"]

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

# Backward compatibility alias
ARITHMETIC_MNEMONICS = SUPPORTED_MNEMONICS


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

    def _generate_independent_sequence(self, instr: Instruction) -> Optional[str]:
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
        if mnemonic not in ARITHMETIC_MNEMONICS:
            return None

        indep_lines = []
        n = self.config.independent_count
        
        # Generate N independent instructions, each using a different dest register
        if mnemonic in {"add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu"}:
            # R-type: op rd, rs1, rs2 -> use different rd each time
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                # Use t0 as a constant source (doesn't change)
                indep_lines.append(f'        "{mnemonic} {rd}, t0, t1\\n"')
                
        elif mnemonic in {"addi", "andi", "ori", "xori", "slti", "sltiu"}:
            # I-type with 12-bit immediate
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, 1\\n"')
                
        elif mnemonic in {"slli", "srli", "srai"}:
            # Shift with 5-bit immediate
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, 1\\n"')
                
        elif mnemonic in {"c.add"}:
            # c.add rd, rs2 - rd can be any register, rs2 can be any except x0
            # Use a0-a5 as destinations with s0 as constant source (RVC compatible)
            compressed_regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1"]
            for i in range(n):
                rd = compressed_regs[i % len(compressed_regs)]
                indep_lines.append(f'        "c.add {rd}, s0\\n"')
                
        elif mnemonic in {"c.sub", "c.and", "c.or", "c.xor"}:
            # These compressed R-type instructions require BOTH operands from RVC set
            # (s0-s1, a0-a5). We can't make them truly independent without dependencies.
            # For throughput, we'll use different dest regs but same source (creates dependency)
            # This measures "throughput with dependency" which is still useful
            compressed_regs = ["a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"]
            for i in range(n):
                rd = compressed_regs[i % len(compressed_regs)]
                # Use s1 as source (it's in RVC set)
                indep_lines.append(f'        "{mnemonic} {rd}, s1\\n"')
                
        elif mnemonic in {"c.addi", "c.andi"}:
            # Compressed I-type
            compressed_regs = ["a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"]
            for i in range(n):
                rd = compressed_regs[i % len(compressed_regs)]
                imm = 1 if mnemonic == "c.addi" else 0
                indep_lines.append(f'        "{mnemonic} {rd}, {imm}\\n"')
                
        elif mnemonic in {"c.slli", "c.srli", "c.srai"}:
            # Compressed shift
            compressed_regs = ["a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"]
            for i in range(n):
                rd = compressed_regs[i % len(compressed_regs)]
                indep_lines.append(f'        "{mnemonic} {rd}, 1\\n"')
                
        elif mnemonic == "c.mv":
            # c.mv rd, rs2 - rd can be any register except x0, rs2 can be any except x0
            # Use s0 as source (RVC compatible) for consistency
            compressed_regs = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1"]
            for i in range(n):
                rd = compressed_regs[i % len(compressed_regs)]
                indep_lines.append(f'        "c.mv {rd}, s0\\n"')
                
        elif mnemonic == "c.li":
            # c.li rd, imm - load immediate
            compressed_regs = ["a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"]
            for i in range(n):
                rd = compressed_regs[i % len(compressed_regs)]
                indep_lines.append(f'        "c.li {rd}, {i % 32}\\n"')
                
        # Phase 2: Multiply/Divide (M extension)
        elif mnemonic in {"mul", "mulh", "mulhsu", "mulhu"}:
            # R-type multiply: op rd, rs1, rs2
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, t1\\n"')
                
        elif mnemonic in {"div", "divu", "rem", "remu"}:
            # R-type divide: op rd, rs1, rs2
            # Note: Division is typically not pipelined, so throughput = latency
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0, t1\\n"')
                
        # Phase 3: Upper Immediate
        elif mnemonic == "lui":
            # lui rd, imm20 - each to different register
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                indep_lines.append(f'        "lui {rd}, 0x12345\\n"')
                
        elif mnemonic == "auipc":
            # auipc rd, imm20 - each to different register
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                indep_lines.append(f'        "auipc {rd}, 0\\n"')
                
        # Phase 3: Sign/Zero Extend (Zbb extension)
        elif mnemonic in {"sext.b", "sext.h", "zext.h"}:
            # Unary: op rd, rs1
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                indep_lines.append(f'        "{mnemonic} {rd}, t0\\n"')
                
        # Phase 4: Load instructions
        elif mnemonic in {"lw", "lh", "lhu", "lb", "lbu"}:
            # Load from different offsets in memory array
            # Each load goes to a different register from independent addresses
            for i in range(n):
                rd = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
                offset = i * 4  # 4-byte aligned offsets
                indep_lines.append(f'        "{mnemonic} {rd}, {offset}(s0)\\n"')
                
        elif mnemonic in {"c.lw", "c.sw", "c.lwsp", "c.swsp"}:
            # Compressed load/store
            # c.lw/sw use s1 as base (except *sp which uses sp implicitly)
            # For c.lw, we MUST NOT overwrite s1 (base).
            # For c.sw, s1 is handled as data source, which is fine, but we'll exclude it for consistency if desired.
            
            base_reg = "s1"
            
            # RVC-compatible registers that can be used as destinations/sources
            regs_to_use = ["a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1"]
            
            if mnemonic == "c.lw":
                # Exclude base_reg from destinations so we don't clobber the pointer
                if base_reg in regs_to_use:
                    regs_to_use.remove(base_reg)
            
            for i in range(n):
                # Use available registers for destination (c.lw, c.lwsp) or source (c.sw, c.swsp)
                reg = regs_to_use[i % len(regs_to_use)]
                
                # c.lw/sw immediate: 5 bits + shift, complicated. 
                # uimm: [5:3|2|6]. scaled by 4. range 0..124.
                # Use a small range of offsets to ensure validity.
                offset = (i % 32) * 4 
                
                if mnemonic == "c.lwsp":
                     # c.lwsp rd, uimm(sp)
                     # uimm range for c.lwsp is 0-252 in steps of 4.
                     offset = (i % 64) * 4
                     indep_lines.append(f'        "{mnemonic} {reg}, {offset}(sp)\\n"')
                elif mnemonic == "c.swsp":
                     # c.swsp rs2, uimm(sp)
                     # uimm range for c.swsp is 0-252 in steps of 4.
                     offset = (i % 64) * 4
                     indep_lines.append(f'        "{mnemonic} {reg}, {offset}(sp)\\n"')
                else: 
                     # c.lw / c.sw
                     indep_lines.append(f'        "{mnemonic} {reg}, {offset}({base_reg})\\n"')
                
        # Phase 4: Store instructions
        elif mnemonic in {"sw", "sh", "sb"}:
            # Store to different offsets in memory array
            for i in range(n):
                rs = THROUGHPUT_REGS[i % len(THROUGHPUT_REGS)]
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
        """Generate complete throughput benchmark function for an instruction."""
        
        # Only process supported instructions
        if not self._is_phase1_instruction(instr):
            return None
            
        sequence = self._generate_independent_sequence(instr)
        
        if sequence is None:
            self.skipped_instructions.append(instr)
            return None

        func_name = self._sanitize_name(instr.llvm_enum_name)
        n = self.config.independent_count
        
        # Check instruction type
        asm = instr.test_asm.lower().strip()
        mnemonic = asm.split()[0] if asm.split() else ""
        is_load_store = mnemonic in {"lw", "lh", "lhu", "lb", "lbu", "sw", "sh", "sb", "c.lw", "c.sw", "c.lwsp", "c.swsp"}
        is_atomic = mnemonic.startswith("amo")
        is_branch = mnemonic in {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}
        is_jump = mnemonic in {"jal", "c.j", "c.jal"}
        
        if is_atomic:
            # Setup for Atomics
            # We need multiple valid addresses to prevent serialization on a single cache line/lock.
            # We allocate a buffer and point a0, a1, a2, a3 to different words.
            setup_code = '''    /* Setup: create aligned memory buffer for atomic throughput */
    static volatile uint32_t mem_buffer[32] __attribute__((aligned(64)));
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];
    register uint32_t t0_val __asm__("t0") = 1; /* Source value */
    (void)t0_val;'''
            
            # Init address registers a0-a3
            asm_init = '''        "mv a0, %0\\n"      /* a0 = base */
        "addi a1, a0, 4\\n" /* a1 = base + 4 */
        "addi a2, a0, 8\\n" /* a2 = base + 8 */
        "addi a3, a0, 12\\n"/* a3 = base + 12 */
'''
            asm_clobber = ': "+r"(base_addr) :: "a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "t2", "memory"'
            post_asm = '        base_addr = base_addr;'

        elif is_branch or is_jump:
            # Setup for control flow: initialize comparison registers
            # a0 = 1 (non-zero), a1 = 0 (zero) for branch conditions
            setup_code = '''    /* Setup: initialize registers for branch conditions */
    register uint32_t a0_val __asm__("a0") = 1;  /* non-zero */
    register uint32_t a1_val __asm__("a1") = 0;  /* zero */
    (void)a0_val; (void)a1_val;'''
            asm_init = ''
            asm_clobber = '::: "a0", "a1", "ra", "memory"'
            post_asm = ''

        elif is_load_store:
            # Memory setup for load/store throughput
            # Uses asm input constraints like the latency generator
            setup_code = '''    /* Setup: create aligned memory buffer for load/store throughput */
    static volatile uint32_t mem_buffer[32] __attribute__((aligned(64)));
    for (int i = 0; i < 32; i++) mem_buffer[i] = 0x12345678 + i;
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];'''
            
            # Use mv to load base address into a register within the asm block
            asm_init = '        "mv s0, %0\\n"  /* Load base address into s0 */\n        "mv s1, %0\\n"  /* Also into s1 for compressed */\n'
            asm_clobber = ': "+r"(base_addr) :: "a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1", "t0", "t1", "memory"'
            post_asm = '        base_addr = base_addr; /* Prevent optimization */'

            # Special handling for Stack Pointer operations
            if mnemonic in {"c.lwsp", "c.swsp"}:
                # c.lwsp/swsp access relative to SP. We must allocate scratch space 
                # on the stack to avoid corrupting the current stack frame (return addr, etc).
                # Unrolled loop uses offsets up to ~256 bytes.
                asm_init += '        "addi sp, sp, -256\\n" /* Reserve stack space */\n'
                # Restore SP at the end of the sequence. 
                # Note: We append to sequence, assuming sequence lines end with \n"
                sequence += '        "addi sp, sp, 256\\n"  /* Restore stack space */\n' 

        else:
            # Standard setup for arithmetic operations
            setup_code = '''    /* Setup: initialize source registers with stable values */
    register uint32_t t0_val __asm__("t0") = 0x12345678;
    register uint32_t t1_val __asm__("t1") = 0x87654321;
    register uint32_t s0_val __asm__("s0") = 0x55555555;
    register uint32_t s1_val __asm__("s1") = 0xAAAAAAAA;
    (void)t0_val; (void)t1_val; (void)s0_val; (void)s1_val;'''
            asm_init = ''
            asm_clobber = '::: "a0", "a1", "a2", "a3", "a4", "a5", "s0", "s1", "memory"'
            post_asm = ''

        func = f'''
/* Throughput benchmark for {instr.llvm_enum_name}: {instr.test_asm} */
static int bench_throughput_{func_name}(uint32_t iterations, benchmark_result_t *result) {{
    uint32_t start, end, elapsed;
    uint64_t total = 0;
    uint64_t min_cycles = UINT64_MAX;
    uint64_t max_cycles = 0;

{setup_code}

    COMPILER_BARRIER();

    for (uint32_t rep = 0; rep < iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();

        __asm__ volatile (
{asm_init}{sequence}
            {asm_clobber}
        );
{post_asm}

        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();

        elapsed = end - start;
        total += elapsed;
        if (elapsed < min_cycles) min_cycles = elapsed;
        if (elapsed > max_cycles) max_cycles = elapsed;
    }}

    /* Throughput = cycles per instruction */
    result->min_cycles = min_cycles / {n};
    result->max_cycles = max_cycles / {n};
    result->avg_cycles = total / ((uint64_t)iterations * {n});
    result->total_iterations = iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append((instr, func_name))
        return func

    def generate_descriptor_entry(self, instr: Instruction, func_name: str) -> str:
        """Generate a benchmark descriptor entry for throughput."""
        lat_enum = self._get_latency_type_enum(instr.latency_type)
        escaped_asm = self._escape_asm(instr.test_asm)
        return f'''    {{
        .instruction_name = "{instr.llvm_enum_name}",
        .asm_syntax = "{escaped_asm}",
        .latency_type = {lat_enum},
        .bench_type = BENCH_TYPE_THROUGHPUT,
        .run_benchmark = bench_throughput_{func_name}
    }}'''
