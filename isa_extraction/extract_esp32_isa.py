#!/usr/bin/env python3
"""
ESP32-C6 Instruction Set Extraction Tool

Extracts valid ESP32-C6 instructions from LLVM's TableGen JSON dump by:
1. Parsing the riscv_defs.json (generated via llvm-tblgen -dump-json)
2. Filtering pseudo instructions
3. Generating test assembly using type-based operand mapping
4. Validating instructions with llvm-mc for the ESP32-C6 target
5. Outputting a JSON file suitable for WCET analysis tools

Usage:
    python extract_esp32_isa.py riscv_defs.json -o esp32c6_instructions.json
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# Configuration
# ============================================================================

# Path to llvm-mc (relative to project root)
LLVM_MC_PATH = Path(__file__).parent.parent / "build" / "bin" / "llvm-mc"

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

# Target configuration for ESP32-C6 (RV32IMAC)
TARGET_TRIPLE = "riscv32"
TARGET_ATTRS = "+m,+a,+c"

# Operand type mappings: maps LLVM operand types to concrete assembly values
OPERAND_MAP = {
    # General Purpose Registers
    "GPR": "a0",
    "GPRNoX0": "a0",
    "GPRNoX0X2": "a0",
    "GPRC": "a0",
    "GPRCMem": "a0",
    "GPRMem": "a0",
    "GPRMemZeroOffset": "(a0)",
    "GPRJALR": "a0",
    "GPRJALRNonX7": "a0",
    "GPRPair": "a0",
    "GPRPairRV32": "a0",
    "GPRPairRV64": "a0",
    "GPRTC": "a0",
    "GPRTCNonX7": "a0",
    "GPRX0": "zero",
    "GPRX1": "ra",
    "GPRX1X5": "ra",
    "GPRX5": "t0",
    "GPRX7": "t2",
    "SP": "sp",
    "SPMem": "sp",
    "SR07": "s0",
    "AnyReg": "a0",
    "AnyRegC": "a0",

    # Floating Point Registers (ESP32-C6 doesn't have FPU, but include for completeness)
    "FPR16": "fa0",
    "FPR16INX": "a0",  # Uses GPR when no FPU
    "FPR32": "fa0",
    "FPR32C": "fa0",
    "FPR32INX": "a0",
    "FPR64": "fa0",
    "FPR64C": "fa0",
    "FPR64INX": "a0",
    "FPR64IN32X": "a0",
    "GPRF16": "a0",
    "GPRF16C": "a0",
    "GPRF32": "a0",
    "GPRF32C": "a0",
    "GPRF32NoX0": "a0",

    # Vector Registers (ESP32-C6 doesn't have V extension)
    "VR": "v0",
    "VRM2": "v0",
    "VRM2NoV0": "v2",
    "VRM4": "v0",
    "VRM4NoV0": "v4",
    "VRM8": "v0",
    "VRM8NoV0": "v8",
    "VMV0": "v0",
    "VMaskOp": "v0",
    "VMaskCarryInOp": "v0",

    # Signed Immediates
    "simm5": "0",
    "simm5_plus1": "1",
    "simm5_plus1_nonzero": "1",
    "simm6": "0",
    "simm6nonzero": "1",
    "simm9_lsb0": "0",
    "simm10_lsb0000nonzero": "16",
    "simm12": "0",
    "simm12_lsb0": "0",
    "simm12_lsb00000": "0",
    "simm12_no6": "0",
    "simm12_plus1": "1",
    "simm13_lsb0": "0",
    "simm21_lsb0_jal": "0",
    "simm26": "0",

    # Unsigned Immediates
    "uimm1": "0",
    "uimm2": "0",
    "uimm2_3": "0",
    "uimm2_4": "0",
    "uimm2_lsb0": "0",
    "uimm2_opcode": "0",
    "uimm3": "0",
    "uimm4": "0",
    "uimm5": "1",  # Use 1 to avoid shift-by-zero issues
    "uimm5_lsb0": "0",
    "uimm5gt3": "4",
    "uimm5nonzero": "1",
    "uimm6": "1",
    "uimm6_lsb0": "0",
    "uimm6gt32": "33",
    "uimm7": "0",
    "uimm7_lsb00": "0",
    "uimm7_lsb000": "0",
    "uimm7_opcode": "0",
    "uimm8": "0",
    "uimm8_lsb00": "0",
    "uimm8_lsb000": "0",
    "uimm8ge32": "32",
    "uimm9_lsb000": "0",
    "uimm10": "0",
    "uimm10_lsb00nonzero": "4",
    "uimm11": "0",
    "uimm16": "0",
    "uimm20": "0",
    "uimm20_auipc": "0",
    "uimm20_lui": "0",
    "uimm32": "0",
    "uimm48": "0",
    "uimm64": "0",
    "uimmlog2xlen": "1",
    "uimmlog2xlennonzero": "1",

    # CSR registers
    "csr_sysreg": "0",
    "sysreg": "0",

    # Round mode
    "frmarg": "rne",
    "frm": "rne",

    # AVL/VL for vector
    "AVL": "a0",

    # Vendor-specific
    "CVrr": "a0",

    # Misc
    "rnum": "0",
    "rlist": "{ra}",
    "stackadj": "16",
    "spimm": "0",
    "tsimm5": "0",
    "tuimm5": "1",
}

# Latency type inference based on superclasses or instruction name patterns
LATENCY_PATTERNS = {
    # Arithmetic
    r"^(ADD|SUB|AND|OR|XOR|SLL|SRL|SRA|SLT|SLTU)": "arithmetic",
    r"^C_(ADD|SUB|AND|OR|XOR|SLLI|SRLI|SRAI)": "arithmetic",
    r"^(ADDI|SLTI|SLTIU|XORI|ORI|ANDI|SLLI|SRLI|SRAI)": "arithmetic",
    r"^C_(ADDI|LI|LUI|MV)": "arithmetic",

    # Multiply/Divide
    r"^(MUL|MULH|MULHSU|MULHU|DIV|DIVU|REM|REMU)": "multiply",

    # Load/Store
    r"^(LB|LH|LW|LD|LBU|LHU|LWU)": "load",
    r"^(SB|SH|SW|SD)": "store",
    r"^C_(LW|SW|LD|SD|LWSP|SWSP|LDSP|SDSP)": "load_store",

    # Branch
    r"^(BEQ|BNE|BLT|BGE|BLTU|BGEU)": "branch",
    r"^C_(BEQZ|BNEZ)": "branch",

    # Jump
    r"^(JAL|JALR)": "jump",
    r"^C_(J|JR|JALR)": "jump",

    # Atomic
    r"^(LR|SC|AMO)": "atomic",
    r"^AMO": "atomic",

    # System
    r"^(ECALL|EBREAK|FENCE|CSR|MRET|SRET|WFI)": "system",

    # Bit manipulation
    r"^(CLZ|CTZ|CPOP|ANDN|ORN|XNOR|MIN|MAX|ROL|ROR|BSET|BCLR|BINV|BEXT)": "bitmanip",
}


@dataclass
class Instruction:
    """Represents a parsed RISC-V instruction."""
    llvm_enum_name: str
    asm_string: str
    in_operands: list[tuple[str, str]] = field(default_factory=list)  # (type, name)
    out_operands: list[tuple[str, str]] = field(default_factory=list)
    is_pseudo: bool = False
    superclasses: list[str] = field(default_factory=list)
    tablegen_entry: dict = field(default_factory=dict)  # Full tablegen record


@dataclass
class ValidInstruction:
    """A validated instruction ready for output."""
    llvm_enum_name: str
    opcode_hex: str
    test_asm: str
    latency_type: str
    tablegen_entry: dict = field(default_factory=dict)  # Full tablegen record


# ============================================================================
# JSON Parsing
# ============================================================================

def parse_operand_list(op_list: dict[str, Any]) -> list[tuple[str, str]]:
    """Parse an InOperandList or OutOperandList into (type, name) tuples."""
    result = []
    args = op_list.get("args", [])

    for arg in args:
        if isinstance(arg, list) and len(arg) >= 2:
            op_def = arg[0]
            op_name = arg[1]

            if isinstance(op_def, dict):
                op_type = op_def.get("printable", "unknown")
            else:
                op_type = str(op_def)

            result.append((op_type, op_name))

    return result


def load_instructions(json_path: Path) -> list[Instruction]:
    """Load and parse instructions from TableGen JSON dump."""
    print(f"Loading {json_path}...")

    with open(json_path, "r") as f:
        data = json.load(f)

    instructions = []

    for name, record in data.items():
        if not isinstance(record, dict):
            continue

        superclasses = record.get("!superclasses", [])
        if "Instruction" not in superclasses:
            continue

        asm_string = record.get("AsmString", "")
        is_pseudo = bool(record.get("isPseudo", 0))

        in_operands = parse_operand_list(record.get("InOperandList", {}))
        out_operands = parse_operand_list(record.get("OutOperandList", {}))

        instr = Instruction(
            llvm_enum_name=name,
            asm_string=asm_string,
            in_operands=in_operands,
            out_operands=out_operands,
            is_pseudo=is_pseudo,
            superclasses=superclasses,
            tablegen_entry=record,  # Store full tablegen record
        )
        instructions.append(instr)

    print(f"Loaded {len(instructions)} instruction records")
    return instructions


# ============================================================================
# Assembly Generation
# ============================================================================

def get_operand_value(op_type: str) -> Optional[str]:
    """Map an operand type to a concrete assembly value."""
    # Direct lookup
    if op_type in OPERAND_MAP:
        return OPERAND_MAP[op_type]

    # Try prefix matching for variants
    for key, value in OPERAND_MAP.items():
        if op_type.startswith(key):
            return value

    # Handle some common patterns
    if "GPR" in op_type:
        return "a0"
    if "FPR" in op_type:
        return "fa0"
    if "VR" in op_type or op_type.startswith("V"):
        return "v0"
    if "simm" in op_type.lower():
        return "0"
    if "uimm" in op_type.lower():
        return "1"
    if "imm" in op_type.lower():
        return "0"

    return None


def generate_assembly(instr: Instruction) -> list[str]:
    """
    Generate test assembly strings for an instruction.

    Returns multiple variants for memory operations to handle
    different assembly syntaxes (op rd, rs1, imm vs op rd, imm(rs1)).
    """
    asm_template = instr.asm_string
    if not asm_template:
        return []

    # Collect all operands (outputs first, then inputs, matching AsmString order)
    all_operands = instr.out_operands + instr.in_operands

    # Build substitution map
    operand_values = {}
    for op_type, op_name in all_operands:
        value = get_operand_value(op_type)
        if value is None:
            # Unknown operand type - can't generate assembly
            return []
        operand_values[op_name] = value

    # Substitute operands in template
    result = asm_template

    # Handle ${name} format first
    for name, value in operand_values.items():
        result = result.replace(f"${{{name}}}", value)

    # Handle $name format
    for name, value in operand_values.items():
        result = result.replace(f"${name}", value)

    # Clean up any remaining tab/whitespace issues
    result = re.sub(r'\s+', ' ', result).strip()

    # Check if all substitutions were made
    if '$' in result:
        # Still has unsubstituted variables
        return []

    return [result]


# ============================================================================
# llvm-mc Validation
# ============================================================================

def validate_assembly(asm: str, llvm_mc_path: Path) -> bool:
    """
    Validate assembly using llvm-mc.
    Returns True if the assembly is valid for the target.
    """
    try:
        result = subprocess.run(
            [
                str(llvm_mc_path),
                f"-triple={TARGET_TRIPLE}",
                f"-mattr={TARGET_ATTRS}",
                "--filetype=null",
            ],
            input=asm.encode(),
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False


def validate_instruction(args: tuple[str, str, str]) -> Optional[tuple[str, str]]:
    """
    Validate a single instruction. Used for multiprocessing.

    Args:
        args: (llvm_enum_name, test_asm, llvm_mc_path)

    Returns:
        (llvm_enum_name, test_asm) if valid, None otherwise
    """
    llvm_enum_name, test_asm, llvm_mc_path = args

    if validate_assembly(test_asm, Path(llvm_mc_path)):
        return (llvm_enum_name, test_asm)

    return None


# ============================================================================
# Latency Classification
# ============================================================================

def infer_latency_type(instr_name: str, superclasses: list[str]) -> str:
    """Infer the latency type from instruction name or superclasses."""
    for pattern, latency_type in LATENCY_PATTERNS.items():
        if re.match(pattern, instr_name):
            return latency_type

    # Check superclasses for hints
    superclass_str = " ".join(superclasses)
    if "Load" in superclass_str or "LD" in superclass_str:
        return "load"
    if "Store" in superclass_str or "ST" in superclass_str:
        return "store"
    if "Branch" in superclass_str:
        return "branch"
    if "Jump" in superclass_str:
        return "jump"
    if "ALU" in superclass_str:
        return "arithmetic"
    if "MUL" in superclass_str:
        return "multiply"
    if "DIV" in superclass_str:
        return "divide"

    return "unknown"


# ============================================================================
# Main Processing
# ============================================================================

def filter_instructions(instructions: list[Instruction]) -> list[Instruction]:
    """Filter out pseudo and vendor-specific instructions."""
    # Vendor prefixes to skip
    vendor_prefixes = ["QC_", "TH_", "CV_", "SF_", "XCV", "XTH", "XSF"]

    filtered = []
    for instr in instructions:
        # Skip pseudo instructions
        if instr.is_pseudo:
            continue

        # Skip vendor-specific
        if any(instr.llvm_enum_name.startswith(prefix) for prefix in vendor_prefixes):
            continue

        # Skip instructions without assembly string
        if not instr.asm_string:
            continue

        filtered.append(instr)

    return filtered


def process_instructions(
    instructions: list[Instruction],
    llvm_mc_path: Path,
    num_workers: int,
) -> list[ValidInstruction]:
    """
    Process instructions: generate assembly and validate with llvm-mc.
    Uses multiprocessing for faster validation.
    """
    # Generate assembly candidates
    candidates = []
    for instr in instructions:
        asm_variants = generate_assembly(instr)
        for asm in asm_variants:
            candidates.append((instr.llvm_enum_name, asm, str(llvm_mc_path), instr.superclasses, instr.tablegen_entry))

    print(f"Generated {len(candidates)} assembly candidates")

    # Validate in parallel
    valid_instructions = []
    validation_args = [(c[0], c[1], c[2]) for c in candidates]
    superclass_map = {c[0]: c[3] for c in candidates}
    tablegen_map = {c[0]: c[4] for c in candidates}

    print(f"Validating with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(validate_instruction, arg): arg for arg in validation_args}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0:
                print(f"  Validated {completed}/{len(validation_args)}...")

            result = future.result()
            if result is not None:
                llvm_enum_name, test_asm = result

                valid_instr = ValidInstruction(
                    llvm_enum_name=llvm_enum_name,
                    opcode_hex="",  # Could be computed from encoding bits
                    test_asm=test_asm,
                    latency_type=infer_latency_type(llvm_enum_name, superclass_map.get(llvm_enum_name, [])),
                    tablegen_entry=tablegen_map.get(llvm_enum_name, {}),
                )
                valid_instructions.append(valid_instr)

    print(f"Found {len(valid_instructions)} valid instructions")
    return valid_instructions


def write_output(instructions: list[ValidInstruction], output_path: Path):
    """Write validated instructions to JSON file."""
    output = []
    for instr in sorted(instructions, key=lambda x: x.llvm_enum_name):
        output.append({
            "llvm_enum_name": instr.llvm_enum_name,
            "opcode_hex": instr.opcode_hex,
            "test_asm": instr.test_asm,
            "latency_type": instr.latency_type,
            "tablegen_entry": instr.tablegen_entry,
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(output)} instructions to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract valid ESP32-C6 instructions from LLVM TableGen JSON dump",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_DIR / "riscv_defs.json",
        help="Path to riscv_defs.json (default: output/riscv_defs.json)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "esp32c6_instructions.json",
        help="Output JSON file path (default: output/esp32c6_instructions.json)",
    )
    parser.add_argument(
        "--llvm-mc",
        type=Path,
        default=LLVM_MC_PATH,
        help=f"Path to llvm-mc binary (default: {LLVM_MC_PATH})",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Verify llvm-mc exists
    if not args.llvm_mc.exists():
        print(f"Error: llvm-mc not found at {args.llvm_mc}", file=sys.stderr)
        print("Build it with: ./config.sh build", file=sys.stderr)
        sys.exit(1)

    # Verify input exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load and process
    instructions = load_instructions(args.input)

    filtered = filter_instructions(instructions)
    print(f"After filtering: {len(filtered)} candidate instructions")

    valid = process_instructions(filtered, args.llvm_mc, args.jobs)

    write_output(valid, args.output)

    # Summary by latency type
    print("\nSummary by latency type:")
    type_counts: dict[str, int] = {}
    for instr in valid:
        type_counts[instr.latency_type] = type_counts.get(instr.latency_type, 0) + 1
    for lt, count in sorted(type_counts.items()):
        print(f"  {lt}: {count}")


if __name__ == "__main__":
    main()
