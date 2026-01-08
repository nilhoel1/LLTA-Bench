#!/usr/bin/env python3
"""
generate_latency_benchmarks.py

Generates C header files containing latency benchmarks for ESP32-C6 RISC-V instructions.
Based on the methodology from "uops.info: Characterizing Latency, Throughput, and
Port Usage of Instructions on Intel Microarchitectures" (ASPLOS '19).

For latency measurement, we create dependency chains where the output of one
instruction becomes the input of the next, forcing sequential execution.

Usage:
    python benchmark_generator/generate_latency_benchmarks.py \
        --input isa_extraction/output/esp32c6_instructions.json \
        --output esp32c6_benchmark/main/generated_benchmarks.h
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datetime import datetime

# Default paths relative to project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "isa_extraction" / "output" / "esp32c6_instructions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "esp32c6_benchmark" / "main" / "generated_benchmarks.h"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark generation."""
    warmup_iterations: int = 100
    measurement_iterations: int = 1000
    repeat_count: int = 5
    chain_length: int = 100  # Number of instructions in dependency chain


@dataclass
class Instruction:
    """Represents an instruction to benchmark."""
    llvm_enum_name: str
    opcode_hex: str
    test_asm: str
    latency_type: str


# Mapping from JSON latency_type to C enum
LATENCY_TYPE_MAP = {
    "arithmetic": "LAT_TYPE_ARITHMETIC",
    "load": "LAT_TYPE_LOAD",
    "store": "LAT_TYPE_STORE",
    "load_store": "LAT_TYPE_LOAD_STORE",
    "branch": "LAT_TYPE_BRANCH",
    "jump": "LAT_TYPE_JUMP",
    "multiply": "LAT_TYPE_MULTIPLY",
    "atomic": "LAT_TYPE_ATOMIC",
    "system": "LAT_TYPE_SYSTEM",
    "unknown": "LAT_TYPE_UNKNOWN",
}


class LatencyBenchmarkGenerator:
    """
    Generates latency benchmarks for RISC-V instructions.

    Latency is measured by creating a dependency chain where each instruction
    depends on the result of the previous one. This forces the CPU to execute
    them sequentially, and the cycle count divided by chain length gives the
    per-instruction latency.

    ==========================================================================
    SKIPPED INSTRUCTION CATEGORIES AND MEASUREMENT STRATEGIES
    ==========================================================================

    1. STORE INSTRUCTIONS (sb, sh, sw, c.sw, c.swsp)
       -------------------------------------------------------------------------
       WHY SKIPPED:
       - Stores don't produce a register result for chaining
       - Store latency is typically hidden by write buffers
       - True "latency" is store-to-load forwarding time

       HOW TO MEASURE:
       - Measure store-to-load forwarding: sw followed by lw from same address
       - Measure store throughput (independent stores) instead of latency
       - Use memory barriers to force store completion
       - Chain: store + load pairs where load depends on store
       - Example: "sw a0, 0(a1); lw a0, 0(a1)" measures forwarding latency

    2. BYTE/HALF LOADS (lb, lbu, lh, lhu)
       -------------------------------------------------------------------------
       WHY SKIPPED:
       - Load only 1-2 bytes, which is not a valid 32-bit address
       - Self-referential pointer trick doesn't work (need full 4 bytes)
       - Sign/zero extension makes chaining unreliable

       HOW TO MEASURE:
       - Use separate address and data registers
       - Keep address in a0, load into a1: "lb a1, 0(a0)"
       - Chain with arithmetic: "lb a1, 0(a0); add a0, a0, a1"
         (requires careful address setup so a0+a1 is valid)
       - Measure throughput instead of latency (independent loads)
       - Use indexed addressing with base address restoration

    3. STACK POINTER INSTRUCTIONS (c.addi16sp, c.addi4spn, c.lwsp, c.swsp)
       -------------------------------------------------------------------------
       WHY SKIPPED:
       - Modify or depend on stack pointer (sp)
       - Corrupting sp causes stack protection faults
       - System relies on valid sp for interrupts and function calls

       HOW TO MEASURE:
       - Save and restore sp around measurement
       - Disable interrupts during measurement
       - Use alternative stack area for testing
       - Measure equivalent non-sp instructions as proxy
       - Example: Save sp to s0, run benchmark, restore sp from s0

    4. ZERO REGISTER HINTS (c.add zero, c.li zero, c.mv zero, c.slli zero)
       -------------------------------------------------------------------------
       WHY SKIPPED:
       - Write to x0 (zero) register which is hardwired to 0
       - These are "hint" instructions with no visible effect
       - No output to chain with subsequent instructions

       HOW TO MEASURE:
       - Measure throughput (how many can execute per cycle)
       - These may be optimized away by the CPU (zero-latency)
       - Check if they consume execution resources
       - Use performance counters to verify execution

    5. SYSTEM INSTRUCTIONS (ecall, ebreak, mret, sret, wfi, fence, csr*)
       -------------------------------------------------------------------------
       WHY SKIPPED:
       - ecall/ebreak: Trigger exceptions/traps
       - mret/sret: Return from machine/supervisor mode (privileged)
       - wfi: Halts CPU until interrupt
       - fence: Memory barriers with variable latency
       - csr*: Access control/status registers (may be privileged)

       HOW TO MEASURE:
       - Requires privileged mode or trap handlers
       - Set up exception handlers to measure ecall/ebreak
       - For fence: Measure with dirty cache lines to see real cost
       - CSR access: Use only user-accessible CSRs (cycle, time, instret)
       - Run in machine mode or set up proper privilege levels

    6. GENERIC INSTRUCTION ENCODINGS (.insn)
       -------------------------------------------------------------------------
       WHY SKIPPED:
       - .insn is a pseudo-instruction for custom encodings
       - The test patterns encode NOPs or undefined instructions
       - No meaningful operation to benchmark

       HOW TO MEASURE:
       - Replace with actual custom instruction encodings
       - Only useful for vendor-specific extensions
       - Encode real instructions using .insn syntax if needed
    ==========================================================================
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

    def _generate_dependency_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate a dependency chain for an instruction.

        The key insight from the uops.info paper is that to measure latency,
        we need to create a chain where each instruction's output feeds into
        the next instruction's input.

        For RISC-V:
        - Register-to-register ops: Use same register as src and dst
        - Loads: Need memory setup (more complex)
        - Stores: Latency is typically to memory (harder to measure directly)
        - Branches/Jumps: Need special handling to prevent control flow issues

        Returns None if the instruction cannot be benchmarked for latency.
        """
        asm = instr.test_asm.lower().strip()
        lat_type = instr.latency_type

        # =======================================================================
        # SKIP PATTERNS - Instructions that cannot be safely benchmarked
        # See class docstring for detailed explanations and measurement strategies
        # =======================================================================

        # System/trap instructions - require privileged mode or crash the system
        skip_patterns = [
            "ecall",     # System call - triggers trap handler
            "ebreak",    # Breakpoint - triggers debug exception
            "mret",      # Machine mode return - privileged
            "sret",      # Supervisor mode return - privileged
            "dret",      # Debug return - privileged
            "wfi",       # Wait for interrupt - halts CPU
            "fence",     # Memory barrier - variable latency, no register output
            "sfence",    # TLB flush - privileged, variable latency
            "unimp",     # Undefined instruction - triggers exception
            "c.unimp",   # Compressed undefined instruction
            ".insn",     # Generic encoding - usually NOP or undefined
            "c.nop",     # No operation - no meaningful latency
            "nop",       # No operation - no meaningful latency
        ]

        # Instructions that modify critical registers or use problematic patterns
        # These can crash the system or produce undefined behavior
        dangerous_patterns = [
            "addi16sp",  # c.addi16sp - modifies stack pointer by 16*imm
            "addi4spn",  # c.addi4spn - computes sp + imm into register
            ", sp,",     # Any instruction with sp as destination
            " sp,",      # Instructions starting with sp as destination
            ",sp)",      # Memory operations using sp as base (stores)
            "(sp)",      # Memory operations using sp as base (loads)
            " zero,",    # Writing to zero register - these are hints
            ",zero,",    # Zero register as destination - no visible effect
            "slli64",    # c.slli64 - RV64-only hint, illegal on RV32
            "srai64",    # c.srai64 - RV64-only hint, illegal on RV32
            "srli64",    # c.srli64 - RV64-only hint, illegal on RV32
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
            return self._gen_arithmetic_chain(instr)  # Same approach
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
            return None  # Skip system instructions for now
        else:
            return self._gen_arithmetic_chain(instr)  # Try arithmetic approach

    def _gen_arithmetic_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate dependency chain for arithmetic instructions.

        Strategy: Create a chain where each instruction's destination register
        is used as the source for the next instruction.

        Example for "add a0, a0, a0":
        We generate N copies where a0 depends on previous result.
        """
        asm = instr.test_asm

        # For instructions like "add a0, a0, a0" - they already have dependency
        # For instructions like "addi a0, a0, 0" - same register reused
        # For instructions like "c.add a0, a0" - compressed, same pattern

        # Build the chain by repeating the instruction
        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{asm}\\n"')

        return "\n".join(chain_lines)

    def _gen_load_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate dependency chain for load instructions.

        Strategy: Load from memory using a self-referential pointer.
        Memory location contains its own address, so "lw a0, 0(a0)" loads
        the address back into a0, maintaining the chain.

        BYTE/HALF LOADS SKIPPED:
        - lb/lbu/lh/lhu only load 1-2 bytes
        - These partial values are not valid 32-bit addresses
        - Chain would crash on second iteration

        To measure byte/half loads, use separate address and data registers:
        - Keep address stable in a0
        - Load into different register: "lb a1, 0(a0)"
        - Chain via arithmetic: "lb a1, 0(a0); add a2, a2, a1"
        """
        asm = instr.test_asm.lower()

        # Only allow word-sized loads for dependency chaining
        # Byte and half loads can't maintain a valid address chain
        if "lb" in asm or "lh" in asm:
            # SKIPPED - byte/half loads can't chain with self-referential pointer
            return None

        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{instr.test_asm}\\n"')

        return "\n".join(chain_lines)

    def _gen_store_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate measurement for store instructions.

        Stores don't produce a register result, so we can't create a true
        dependency chain through registers.

        SKIPPED - See class docstring for measurement strategies.
        Key approach: Measure store-to-load forwarding latency instead.
        """
        return None

    def _gen_load_store_chain(self, instr: Instruction) -> Optional[str]:
        """Handle compressed load/store instructions."""
        asm = instr.test_asm.lower()

        # Only word loads can chain (c.lw) - byte/half loads can't
        if "lw" in asm:
            return self._gen_load_chain(instr)
        elif "sw" in asm or "sh" in asm or "sb" in asm:
            # SKIPPED - stores need special handling
            return self._gen_store_chain(instr)
        elif "lh" in asm or "lb" in asm:
            # SKIPPED - byte/half loads can't maintain valid address chain
            return None

        return None

    def _gen_branch_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate measurement for branch instructions.

        Strategy: Create "not-taken" branches to measure minimal branch latency.
        - Set up registers so the branch condition is never true
        - Use local labels with forward reference to next instruction
        - Measure throughput of never-taken branches

        On in-order cores like ESP32-C6, this gives per-instruction latency.
        This measures the compare-and-decide latency, NOT branch misprediction penalty.

        Register setup:
        - a0 = 0, a1 = 1 for equality/inequality comparisons
        - This ensures specific branch conditions are never taken
        """
        asm = instr.test_asm.lower()

        # Extract the mnemonic
        parts = asm.split()
        if not parts:
            return None

        mnemonic = parts[0]

        # Map branch types to never-taken versions
        # We use a0=0, a1=1 setup
        # beq: 0 == 1 is false -> never taken ✓
        # bne: 0 != 1 is true -> always taken, so swap operands: bne a1, a0 with a1=a0? No, use bge instead
        # blt: 0 < 1 is true -> always taken, swap: blt a1, a0 (1 < 0 is false)
        # bge: 0 >= 1 is false -> never taken ✓
        # bltu: 0 < 1 (unsigned) is true -> swap: bltu a1, a0 (1 < 0 is false)
        # bgeu: 0 >= 1 (unsigned) is false -> never taken ✓

        # For compressed branches:
        # c.beqz: a0 == 0 is true with a0=0 -> use a1 instead (a1=1, 1==0 is false)
        # c.bnez: a0 != 0 is false with a0=0 -> never taken ✓

        # Determine the not-taken branch instruction
        # Use "1f" as forward label reference to jump to next instruction if taken
        if mnemonic == "beq":
            new_asm = "beq a0, a1, 1f"  # 0 == 1 is false, never taken
        elif mnemonic == "bne":
            # bne with a0=0, a1=1: 0 != 1 is TRUE, so it would be taken
            # Instead use: beq (same latency, different condition) or swap args
            # Actually, let's just measure with taken branch to same location
            # Use "1f" label - even if taken, it goes to next instruction
            new_asm = "bne a0, a0, 1f"  # 0 != 0 is false, never taken
        elif mnemonic == "blt":
            new_asm = "blt a1, a0, 1f"  # 1 < 0 is false, never taken
        elif mnemonic == "bge":
            new_asm = "bge a0, a1, 1f"  # 0 >= 1 is false, never taken
        elif mnemonic == "bltu":
            new_asm = "bltu a1, a0, 1f"  # 1 < 0 (unsigned) is false, never taken
        elif mnemonic == "bgeu":
            new_asm = "bgeu a0, a1, 1f"  # 0 >= 1 (unsigned) is false, never taken
        elif mnemonic == "c.beqz":
            new_asm = "c.beqz a1, 1f"  # a1=1, 1 == 0 is false, never taken
        elif mnemonic == "c.bnez":
            new_asm = "c.bnez a0, 1f"  # a0=0, 0 != 0 is false, never taken
        else:
            return None

        # Generate chain with local labels
        # Each branch jumps to label "1:" which is placed right after it
        # The "1f" means "forward reference to label 1"
        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{new_asm}\\n"')
            chain_lines.append(f'        "1:\\n"')  # Label for the branch target

        return "\n".join(chain_lines)

    def _gen_jump_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate measurement for jump instructions.

        Strategy: Create jumps that target the next instruction to measure
        minimal jump latency without pipeline flush penalties.

        For direct jumps (JAL, C.J, C.JAL):
        - Use forward label reference "1f" to jump to next instruction
        - Place label "1:" immediately after the jump

        For register-indirect jumps (JALR, C.JR, C.JALR):
        - SKIPPED: Require computed addresses that are difficult to get right
        - The last iteration would jump outside the asm block
        - Would need special handling for chain termination

        Note: We use rd=zero where possible to measure pure jump latency
        without the link register write overhead.
        """
        asm = instr.test_asm.lower()

        # Extract the mnemonic
        parts = asm.split()
        if not parts:
            return None

        mnemonic = parts[0]

        chain_lines = []

        if mnemonic == "jal":
            # JAL rd, offset -> jal zero, 1f (no link save, jump forward)
            for i in range(self.config.chain_length):
                chain_lines.append('        "jal zero, 1f\\n"')
                chain_lines.append('        "1:\\n"')

        elif mnemonic == "jalr":
            # SKIPPED: Register-indirect jump with computed address
            # Difficult to terminate chain safely
            return None

        elif mnemonic == "c.j":
            # C.J offset -> c.j 1f (jump forward)
            for i in range(self.config.chain_length):
                chain_lines.append('        "c.j 1f\\n"')
                chain_lines.append('        "1:\\n"')

        elif mnemonic == "c.jal":
            # C.JAL offset -> c.jal 1f (jump forward, saves ra)
            for i in range(self.config.chain_length):
                chain_lines.append('        "c.jal 1f\\n"')
                chain_lines.append('        "1:\\n"')

        elif mnemonic == "c.jr":
            # SKIPPED: Register-indirect jump
            # Would need computed address, last iteration jumps outside asm block
            return None

        elif mnemonic == "c.jalr":
            # SKIPPED: Register-indirect jump with link
            # Would need computed address, last iteration jumps outside asm block
            return None

        else:
            return None

        return "\n".join(chain_lines)

    def _gen_atomic_chain(self, instr: Instruction) -> Optional[str]:
        """
        Generate dependency chain for atomic instructions.

        Strategy: Use separate registers for address and data to maintain
        a stable address throughout the chain:
        - a0: stable address register (points to aligned, self-referential memory)
        - a1: destination register (rd) - receives old value from memory
        - a2: source operand (rs2) - value to use in atomic operation

        The chain is maintained through memory dependencies - each atomic
        operation reads and writes the same memory location, creating a
        serialization point.

        LR/SC pairs are skipped as they require special handling.
        """
        asm = instr.test_asm.lower()

        # Skip LR/SC - they must be measured as pairs with reservation logic
        if "lr." in asm or "sc." in asm:
            return None

        # Extract the mnemonic (e.g., "amoadd.w", "amoswap.w", "amoand.w")
        parts = asm.split()
        if not parts:
            return None

        mnemonic = parts[0]

        # Validate it's an AMO instruction
        if not mnemonic.startswith("amo"):
            return None

        # Generate the rewritten instruction with proper register allocation:
        # amo*.w rd, rs2, (rs1) -> amo*.w a1, a2, (a0)
        # - a0 stays stable (address)
        # - a1 receives the old memory value
        # - a2 provides the source operand for the atomic operation
        new_asm = f"{mnemonic} a1, a2, (a0)"

        chain_lines = []
        for i in range(self.config.chain_length):
            chain_lines.append(f'        "{new_asm}\\n"')

        return "\n".join(chain_lines)

    def _generate_benchmark_function(self, instr: Instruction) -> Optional[str]:
        """Generate complete benchmark function for an instruction."""
        chain = self._generate_dependency_chain(instr)

        if chain is None:
            self.skipped_instructions.append(instr)
            return None

        func_name = self._sanitize_name(instr.llvm_enum_name)

        # Generate setup code based on instruction type
        setup_code = self._generate_setup_code(instr)

        # Determine if this is a load instruction that needs address setup
        needs_addr_setup = instr.latency_type in ["load", "load_store", "atomic"]

        # Build the function with appropriate asm constraints
        if needs_addr_setup:
            # For loads: Use input constraint to initialize a0 with base_addr
            asm_constraints = ': "+r"(base_addr) :: "a1", "a2", "a3", "a4", "a5", "t0", "t1", "memory"'
            # Prepend instruction to move base_addr into a0
            asm_init = '        "mv a0, %0\\n"  /* Load base address into a0 */\n'
            post_asm = '        base_addr = base_addr; /* Prevent optimization */'
        else:
            asm_constraints = '::: "a0", "a1", "a2", "a3", "a4", "a5", "t0", "t1", "memory"'
            asm_init = ''
            post_asm = ''

        func = f'''
/* Benchmark for {instr.llvm_enum_name}: {instr.test_asm} */
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
    result->avg_cycles = total / (iterations * {self.config.chain_length});
    result->total_iterations = iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append((instr, func_name))
        return func

    def _generate_setup_code(self, instr: Instruction) -> str:
        """Generate setup code to initialize registers/memory before benchmark."""
        lat_type = instr.latency_type

        if lat_type == "atomic":
            # For atomics, we need valid aligned memory and proper register setup
            # a0: stable address, a1: destination, a2: source operand
            return '''    /* Setup: create aligned memory for atomic operations */
    static volatile uint32_t mem_buffer[16] __attribute__((aligned(64)));
    mem_buffer[0] = (uint32_t)(uintptr_t)&mem_buffer[0];

    /* Store pointer address - will be loaded into a0 via asm input */
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];

    /* Initialize a2 with a small value for atomic operations */
    register uint32_t a2_val __asm__("a2") = 1;
    (void)a2_val;'''
        elif lat_type == "branch":
            # For branches, set up registers so branches are never taken
            # a0 = 0, a1 = 1 allows control over branch direction
            return '''    /* Setup: initialize registers for never-taken branches */
    /* a0 = 0, a1 = 1: beq never taken, bge never taken, etc. */
    register uint32_t a0_val __asm__("a0") = 0;
    register uint32_t a1_val __asm__("a1") = 1;
    (void)a0_val; (void)a1_val;'''
        elif lat_type in ["load", "load_store"]:
            # For loads, we need valid memory addresses
            # Store pointer in a regular C variable - it will be passed to asm via input constraint
            return '''    /* Setup: create a memory location that points to itself */
    static volatile uint32_t mem_buffer[16] __attribute__((aligned(64)));
    mem_buffer[0] = (uint32_t)(uintptr_t)&mem_buffer[0];

    /* Store pointer address - will be loaded into a0 via asm input */
    uint32_t base_addr = (uint32_t)(uintptr_t)&mem_buffer[0];'''
        else:
            # For arithmetic, just initialize registers
            return '''    /* Setup: initialize registers with known values */
    register uint32_t a0_val __asm__("a0") = 0x12345678;
    register uint32_t a1_val __asm__("a1") = 0x87654321;
    (void)a0_val; (void)a1_val;'''

    def generate_header(self, instructions: list, output_file: str) -> None:
        """Generate the complete benchmark header file."""
        benchmark_functions = []

        for instr_data in instructions:
            instr = Instruction(
                llvm_enum_name=instr_data["llvm_enum_name"],
                opcode_hex=instr_data.get("opcode_hex", ""),
                test_asm=instr_data["test_asm"],
                latency_type=instr_data.get("latency_type", "unknown")
            )

            func = self._generate_benchmark_function(instr)
            if func:
                benchmark_functions.append(func)

        # Generate the header file content
        header_content = self._generate_header_content(benchmark_functions)

        with open(output_file, 'w') as f:
            f.write(header_content)

        # Print summary
        print(f"Generated {len(self.generated_benchmarks)} benchmarks")
        print(f"Skipped {len(self.skipped_instructions)} instructions:")
        for instr in self.skipped_instructions:
            print(f"  - {instr.llvm_enum_name}: {instr.test_asm} ({instr.latency_type})")

    def _generate_header_content(self, functions: list) -> str:
        """Generate the complete header file content."""
        timestamp = datetime.now().isoformat()

        # Build the descriptor array entries
        descriptor_entries = []
        for instr, func_name in self.generated_benchmarks:
            lat_enum = self._get_latency_type_enum(instr.latency_type)
            escaped_asm = self._escape_asm(instr.test_asm)
            entry = f'''    {{
        .instruction_name = "{instr.llvm_enum_name}",
        .asm_syntax = "{escaped_asm}",
        .latency_type = {lat_enum},
        .bench_type = BENCH_TYPE_LATENCY,
        .run_benchmark = bench_latency_{func_name}
    }}'''
            descriptor_entries.append(entry)

        descriptors = ",\n".join(descriptor_entries)
        functions_code = "\n".join(functions)

        header = f'''/**
 * @file generated_benchmarks.h
 * @brief Auto-generated latency benchmarks for ESP32-C6 RISC-V instructions
 *
 * Generated: {timestamp}
 * Generator: generate_latency_benchmarks.py
 *
 * DO NOT EDIT THIS FILE MANUALLY - it is auto-generated.
 *
 * Methodology based on: "uops.info: Characterizing Latency, Throughput, and
 * Port Usage of Instructions on Intel Microarchitectures" (ASPLOS '19)
 */

#ifndef GENERATED_BENCHMARKS_H
#define GENERATED_BENCHMARKS_H

#include "benchmark_interface.h"
#include <limits.h>

/*
 * =============================================================================
 * Benchmark Set Information
 * =============================================================================
 */

const char *BENCHMARK_SET_NAME = "ESP32-C6 Latency Benchmarks v1.0";

/*
 * =============================================================================
 * Benchmark Configuration
 * =============================================================================
 */

const benchmark_config_t BENCHMARK_CONFIG = {{
    .warmup_iterations = {self.config.warmup_iterations},
    .measurement_iterations = {self.config.measurement_iterations},
    .repeat_count = {self.config.repeat_count},
    .chain_length = {self.config.chain_length}
}};

/*
 * =============================================================================
 * Benchmark Function Implementations
 * =============================================================================
 */

{functions_code}

/*
 * =============================================================================
 * Benchmark Descriptor Array
 * =============================================================================
 */

const benchmark_descriptor_t BENCHMARKS[] = {{
{descriptors}
}};

const size_t BENCHMARK_COUNT = sizeof(BENCHMARKS) / sizeof(BENCHMARKS[0]);

#endif /* GENERATED_BENCHMARKS_H */
'''
        return header


def main():
    parser = argparse.ArgumentParser(
        description="Generate latency benchmarks for ESP32-C6 RISC-V instructions"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(DEFAULT_INPUT),
        help=f"Input JSON file with instruction definitions (default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)})"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT),
        help=f"Output header file path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup iterations (default: 100)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of measurement iterations (default: 1000)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of times to repeat measurement (default: 5)"
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=100,
        help="Length of dependency chain (default: 100)"
    )

    args = parser.parse_args()

    # Load instructions from JSON
    try:
        with open(args.input, 'r') as f:
            instructions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}")
        sys.exit(1)

    print(f"Loaded {len(instructions)} instructions from {args.input}")

    # Create configuration
    config = BenchmarkConfig(
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        repeat_count=args.repeats,
        chain_length=args.chain_length
    )

    # Generate benchmarks
    generator = LatencyBenchmarkGenerator(config)
    generator.generate_header(instructions, args.output)

    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
