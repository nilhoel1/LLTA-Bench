# ESP32-C6 ISA Extraction Tool

This tool generates an exhaustive list of valid ESP32-C6 instructions for use in WCET analysis tools. It works by filtering instructions from an LLVM TableGen dump and validating them against the LLVM assembler (`llvm-mc`) configured for the ESP32-C6 target (RISC-V 32-bit IMAC extensions).

## Prerequisites

1.  **LLVM Build**: You need a local build of LLVM with `llvm-tblgen` and `llvm-mc` built.
    *   Use the provided `config.sh` script to configure and build.
    *   Ensure `riscv_defs.json` is generated (see below).
2.  **Python 3.6+**: For running the extraction script.

## Step 1: Generate `riscv_defs.json`

First, you need to dump the RISC-V instruction definitions from LLVM's TableGen files into a JSON format.

You can use the helper command in `config.sh`:

```bash
./config.sh genDefs
```

This will run `llvm-tblgen` with the correct include paths and generate `riscv_defs.json` in the current directory.

## Step 2: Run the Extraction Script

Run the `extract_esp32_isa.py` script. It requires the path to `riscv_defs.json`.

```bash
python3 extract_esp32_isa.py riscv_defs.json -o esp32c6_instructions.json
```

**Options:**
*   `-o, --output`: Output JSON file path (default: `esp32c6_instructions.json`).
*   `--llvm-mc`: Path to `llvm-mc` binary (default: `./build/bin/llvm-mc`).
*   `-j, --jobs`: Number of parallel workers for validation (default: 8).
*   `-v, --verbose`: Enable verbose output.

## Output Format

The output is a JSON array of valid instruction objects:

```json
[
  {
    "llvm_enum_name": "ADD",      // LLVM Record Name (maps to RISCV::ADD)
    "opcode_hex": "",             // (Reserved for future use)
    "test_asm": "add a0, a0, a0", // Valid assembly string for testing
    "latency_type": "arithmetic"  // Inferred instruction type
  },
  ...
]
```

## How It Works

1.  **Parsing**: Loads `riscv_defs.json` and iterates over all records inheriting from `Instruction`.
2.  **Filtering**: Skips pseudo instructions (`isPseudo: 1`) and vendor-specific prefixes (e.g., `QC_`, `TH_`).
3.  **Operand Mapping**: Maps abstract LLVM operand types (e.g., `GPR`, `simm12`, `uimm5`) to concrete, safe values (e.g., `a0`, `0`, `1`).
4.  **Validation**: Generates candidate assembly strings and feeds them to `llvm-mc -triple=riscv32 -mattr=+m,+a,+c`.
    *   If `llvm-mc` accepts the assembly (exit code 0), the instruction is deemed valid.
    *   If `llvm-mc` errors, the instruction is discarded.
5.  **Classification**: Infers a `latency_type` (arithmetic, load, store, etc.) based on the instruction name and superclasses.

## Supported Extensions (ESP32-C6)

The script targets `riscv32` with `+m` (Multiply), `+a` (Atomic), and `+c` (Compressed) extensions.
