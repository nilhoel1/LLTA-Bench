#!/usr/bin/env python3
"""
latency_generator.py

Latency benchmark generator for ESP32-C6 RISC-V instructions.
Extracted from generate_latency_benchmarks.py for modular structure.

Latency is measured by creating dependency chains where the output of one
instruction becomes the input of the next, forcing sequential execution.
"""

from dataclasses import dataclass
from typing import Optional, List

try:
    from .common import Instruction, BenchmarkConfig, LATENCY_TYPE_MAP
except ImportError:
    from common import Instruction, BenchmarkConfig, LATENCY_TYPE_MAP


class LatencyBenchmarkGenerator:
    """
    Generates latency benchmarks for RISC-V instructions.

    Latency is measured by creating a dependency chain where each instruction
    depends on the result of the previous one. This forces the CPU to execute
    them sequentially, and the cycle count divided by chain length gives the
    per-instruction latency.

    ==========================================================================
    SKIPPED INSTRUCTION CATEGORIES
    ==========================================================================

    1. BYTE/HALF LOADS (lb, lbu, lh, lhu)
       - Load only 1-2 bytes, which is not a valid 32-bit address
       - Self-referential pointer trick doesn't work

    2. STACK POINTER INSTRUCTIONS (c.addi16sp, c.addi4spn, c.lwsp, c.swsp)
       - Modify or depend on stack pointer (sp)
       - Corrupting sp causes stack protection faults

    3. ZERO REGISTER HINTS (c.add zero, c.li zero, etc.)
       - Write to x0 (zero) register which is hardwired to 0

    4. SYSTEM INSTRUCTIONS (ecall, ebreak, mret, sret, wfi, fence, csr*)
       - Require privileged mode or crash the system

    5. GENERIC INSTRUCTION ENCODINGS (.insn)
       - Pseudo-instruction for custom encodings
    """

    # ========================================================================
    # TableGen Entry Helpers (Register Class Mapping)
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

    # Default registers for latency benchmarks
    DEFAULT_REGS = ["a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1"]

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.generated_benchmarks = []
        self.skipped_instructions = []

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
        Falls back to DEFAULT_REGS if unknown.
        """
        # Direct lookup
        if regclass in self.REGCLASS_MAP:
            return self.REGCLASS_MAP[regclass]
        
        # Prefix matching for variants (e.g., GPRNoX0X2 starts with GPR)
        for key, regs in self.REGCLASS_MAP.items():
            if regclass.startswith(key):
                return regs
        
        # Fallback to default registers
        return self.DEFAULT_REGS

    def _get_dest_registers_from_tablegen(self, instr: Instruction) -> List[str]:
        """
        Get the list of valid destination registers for an instruction
        by examining its tablegen_entry OutOperandList.
        
        Returns DEFAULT_REGS as fallback if tablegen_entry is unavailable.
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
        
        Returns DEFAULT_REGS as fallback if tablegen_entry is unavailable.
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

    def _generate_dependency_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate a dependency chain for an instruction.
        Returns None if the instruction cannot be benchmarked for latency.
        """
        asm = instr.test_asm.lower().strip()
        lat_type = instr.latency_type

        # Skip patterns - instructions that cannot be safely benchmarked
        skip_patterns = [
            "ecall", "ebreak", "mret", "sret", "dret", "wfi",
            "fence", "sfence", "unimp", "c.unimp", ".insn", "c.nop", "nop",
        ]

        dangerous_patterns = [
            "addi16sp", "addi4spn", ", sp,", " sp,", ",sp)", "(sp)",
            " zero,", ",zero,", "slli64", "srai64", "srli64",
        ]

        for pattern in skip_patterns:
            if pattern in asm:
                return None

        for pattern in dangerous_patterns:
            if pattern in asm:
                return None

        # Handle different instruction types
        if lat_type == "arithmetic":
            return self._gen_arithmetic_chain(instr)
        elif lat_type == "multiply":
            return self._gen_arithmetic_chain(instr)
        elif lat_type == "load":
            return self._gen_load_chain(instr)
        elif lat_type == "store":
            return self._gen_store_chain(instr)
        elif lat_type == "load_store":
            return self._gen_load_store_chain(instr)
        elif lat_type == "branch":
            return self._gen_branch_chain(instr)
        elif lat_type == "jump":
            return self._gen_jump_chain(instr)
        elif lat_type == "atomic":
            return self._gen_atomic_chain(instr)
        elif lat_type == "system":
            return None
        else:
            return self._gen_arithmetic_chain(instr)

    def _gen_arithmetic_chain(self, instr: Instruction) -> Optional[str]:
        """Generate dependency chain for arithmetic instructions."""
        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{instr.test_asm}\\n"')
        return "\n".join(chain_lines)

    def _gen_load_chain(self, instr: Instruction) -> Optional[str]:
        """Generate dependency chain for load instructions."""
        asm = instr.test_asm.lower()

        # Only allow word-sized loads for dependency chaining
        if "lb" in asm or "lh" in asm:
            return None

        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{instr.test_asm}\\n"')
        return "\n".join(chain_lines)

    def _gen_store_chain(self, instr: Instruction) -> Optional[str]:
        """Generate measurement for store instructions using store-to-load forwarding."""
        asm = instr.test_asm.lower()

        if "sp" in asm:
            return None

        parts = asm.split()
        if not parts:
            return None

        mnemonic = parts[0]

        store_load_pairs = {
            "sw": ("sw a0, 0(a1)", "lw a0, 0(a1)"),
            "sh": ("sh a0, 0(a1)", "lhu a0, 0(a1)"),
            "sb": ("sb a0, 0(a1)", "lbu a0, 0(a1)"),
            "c.sw": ("c.sw a0, 0(a1)", "c.lw a0, 0(a1)"),
        }

        if mnemonic not in store_load_pairs:
            return None

        store_instr, load_instr = store_load_pairs[mnemonic]

        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{store_instr}\\n"')
            chain_lines.append(f'        "{load_instr}\\n"')

        return "\n".join(chain_lines)

    def _gen_load_store_chain(self, instr: Instruction) -> Optional[str]:
        """Handle compressed load/store instructions."""
        asm = instr.test_asm.lower()

        if "lw" in asm and "sw" not in asm:
            return self._gen_load_chain(instr)
        elif "sw" in asm:
            return self._gen_store_chain(instr)
        elif "sh" in asm or "sb" in asm:
            return self._gen_store_chain(instr)
        elif "lh" in asm or "lb" in asm:
            return None

        return None

    def _gen_branch_chain(self, instr: Instruction) -> Optional[str]:
        """Generate measurement for branch instructions (not-taken)."""
        asm = instr.test_asm.lower()
        parts = asm.split()
        if not parts:
            return None

        mnemonic = parts[0]

        branch_map = {
            "beq": "beq a0, a1, 1f",
            "bne": "bne a0, a0, 1f",
            "blt": "blt a1, a0, 1f",
            "bge": "bge a0, a1, 1f",
            "bltu": "bltu a1, a0, 1f",
            "bgeu": "bgeu a0, a1, 1f",
            "c.beqz": "c.beqz a1, 1f",
            "c.bnez": "c.bnez a0, 1f",
        }

        if mnemonic not in branch_map:
            return None

        new_asm = branch_map[mnemonic]

        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{new_asm}\\n"')
            chain_lines.append('        "1:\\n"')

        return "\n".join(chain_lines)

    def _gen_jump_chain(self, instr: Instruction) -> Optional[str]:
        """Generate measurement for jump instructions."""
        asm = instr.test_asm.lower()
        parts = asm.split()
        if not parts:
            return None

        mnemonic = parts[0]
        chain_lines = []

        if mnemonic == "jal":
            for i in range(self.config.chain_length):
                chain_lines.append('        "jal zero, 1f\\n"')
                chain_lines.append('        "1:\\n"')
        elif mnemonic == "c.j":
            for i in range(self.config.chain_length):
                chain_lines.append('        "c.j 1f\\n"')
                chain_lines.append('        "1:\\n"')
        elif mnemonic == "c.jal":
            for i in range(self.config.chain_length):
                chain_lines.append('        "c.jal 1f\\n"')
                chain_lines.append('        "1:\\n"')
        else:
            return None

        return "\n".join(chain_lines)

    def _gen_atomic_chain(self, instr: Instruction) -> Optional[str]:
        """Generate dependency chain for atomic instructions."""
        asm = instr.test_asm.lower()

        if "lr." in asm or "sc." in asm:
            return None

        parts = asm.split()
        if not parts:
            return None

        mnemonic = parts[0]
        if not mnemonic.startswith("amo"):
            return None

        new_asm = f"{mnemonic} a1, a2, (a0)"

        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{new_asm}\\n"')

        return "\n".join(chain_lines)

    def _is_store_instruction(self, instr: Instruction) -> bool:
        """Check if instruction is a store."""
        asm = instr.test_asm.lower()
        store_mnemonics = ["sw ", "sh ", "sb ", "c.sw ", "c.sh ", "c.sb "]
        return any(asm.startswith(m) for m in store_mnemonics)

    def _generate_setup_code(self, instr: Instruction, is_store: bool = False) -> str:
        """Generate setup code to initialize registers/memory before benchmark."""
        if instr.setup_code_override:
            return instr.setup_code_override

        lat_type = instr.latency_type

        if lat_type == "atomic":
            return '''    /* Setup: create aligned memory for atomic operations */
    static volatile uint32_t mem_buffer[16] __attribute__((aligned(64)));
    mem_buffer[0] = (uint32_t)(uintptr_t)&mem_buffer[0];
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];
    register uint32_t a2_val __asm__("a2") = 1;
    (void)a2_val;'''
        elif lat_type == "store" or (lat_type == "load_store" and is_store):
            return '''    /* Setup: create aligned memory for store-to-load forwarding */
    static volatile uint32_t mem_buffer[16] __attribute__((aligned(64)));
    mem_buffer[0] = 0x12345678;
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];'''
        elif lat_type == "branch":
            return '''    /* Setup: initialize registers for never-taken branches */
    register uint32_t a0_val __asm__("a0") = 0;
    register uint32_t a1_val __asm__("a1") = 1;
    (void)a0_val; (void)a1_val;'''
        elif lat_type in ["load", "load_store"]:
            return '''    /* Setup: create a memory location that points to itself */
    static volatile uint32_t mem_buffer[16] __attribute__((aligned(64)));
    mem_buffer[0] = (uint32_t)(uintptr_t)&mem_buffer[0];
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];'''
        else:
            return '''    /* Setup: initialize registers with known values */
    register uint32_t a0_val __asm__("a0") = 0x12345678;
    register uint32_t a1_val __asm__("a1") = 0x87654321;
    (void)a0_val; (void)a1_val;'''

    def generate_benchmark_function(self, instr: Instruction) -> Optional[str]:
        """Generate complete benchmark function for an instruction."""
        chain = self._generate_dependency_chain(instr)

        if chain is None:
            self.skipped_instructions.append(instr)
            return None

        func_name = self._sanitize_name(instr.llvm_enum_name + instr.name_suffix)
        is_store = self._is_store_instruction(instr)
        setup_code = self._generate_setup_code(instr, is_store=is_store)
        needs_addr_setup = instr.latency_type in ["load", "store", "load_store", "atomic"]

        if needs_addr_setup:
            if is_store:
                asm_constraints = ': "+r"(base_addr) :: "a0", "a2", "a3", "a4", "a5", "t0", "t1", "memory"'
                asm_init = '        "mv a1, %0\\n"  /* Load base address into a1 */\n        "li a0, 0x5555\\n"  /* Initialize data register */\n'
                post_asm = '        base_addr = base_addr; /* Prevent optimization */'
            else:
                asm_constraints = ': "+r"(base_addr) :: "a1", "a2", "a3", "a4", "a5", "t0", "t1", "memory"'
                asm_init = '        "mv a0, %0\\n"  /* Load base address into a0 */\n'
                post_asm = '        base_addr = base_addr; /* Prevent optimization */'
        else:
            asm_constraints = '::: "a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "memory"'
            asm_init = ''
            post_asm = ''

        func = f'''
/* Latency benchmark for {instr.llvm_enum_name}: {instr.test_asm} */
static int bench_latency_{func_name}(uint32_t iterations, benchmark_result_t *result) {{
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
{asm_init}{chain}
            {asm_constraints}
        );
{post_asm}

        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();

        elapsed = end - start;
        total += elapsed;
        if (elapsed < min_cycles) min_cycles = elapsed;
        if (elapsed > max_cycles) max_cycles = elapsed;
    }}

    result->min_cycles = min_cycles / {self.config.chain_length};
    result->max_cycles = max_cycles / {self.config.chain_length};
    result->avg_cycles = total / ((uint64_t)iterations * {self.config.chain_length});
    result->total_iterations = iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append((instr, func_name))
        return func

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
        """Generate a benchmark descriptor entry."""
        lat_enum = self._get_latency_type_enum(instr.latency_type)
        cat_enum = self._get_category_enum(instr.latency_type)
        escaped_asm = self._escape_asm(instr.test_asm)
        return f'''    {{
        .instruction_name = "{instr.llvm_enum_name}",
        .asm_syntax = "{escaped_asm}",
        .latency_type = {lat_enum},
        .bench_type = BENCH_TYPE_LATENCY,
        .category = {cat_enum},
        .run_benchmark = bench_latency_{func_name}
    }}'''
