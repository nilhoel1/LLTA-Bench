# LLTA-Bench: Low-Level Timing Analysis Benchmark

A comprehensive benchmark suite for measuring instruction latencies on the ESP32-C6 RISC-V microcontroller, inspired by the methodology from ["uops.info: Characterizing Latency, Throughput, and Port Usage of Instructions on Intel Microarchitectures"](https://uops.info/) (ASPLOS '19).

## Project Structure

```
LLTA-Bench/
├── README.md                          # This file
├── config.sh                          # LLVM build configuration script
│
├── isa_extraction/                    # ISA extraction tool
│   ├── README.md                      # Extraction tool documentation
│   ├── extract_esp32_isa.py           # Extracts valid instructions from LLVM
│   └── output/                        # Generated instruction sets
│       ├── riscv_defs.json            # Raw LLVM TableGen dump
│       └── esp32c6_instructions.json  # Validated ESP32-C6 instructions
│
├── benchmark_generator/               # Benchmark generation tools
│   ├── README.md                      # Generator documentation
│   ├── generate_latency_benchmarks.py # Generates benchmark C headers
│   └── parse_benchmark_results.py     # Parses benchmark output
│
├── esp32c6_benchmark/                 # ESP-IDF firmware project
│   ├── README.md                      # Firmware documentation
│   ├── CMakeLists.txt                 # ESP-IDF project file
│   ├── sdkconfig.defaults             # ESP-IDF configuration
│   └── main/
│       ├── main.c                     # Benchmark runner (static)
│       ├── benchmark_interface.h      # Interface definitions
│       └── generated_benchmarks.h     # Auto-generated benchmarks
│
├── scripts/                           # Utility shell scripts
│   └── download_llvm.sh               # Downloads LLVM source
│
├── build/                             # LLVM build output (gitignored)
└── externalDeps/                      # LLVM source (gitignored)
```

## Quick Start

### Prerequisites

- **LLVM Tools**: Built via `config.sh` (requires CMake, Ninja or Make)
- **ESP-IDF v5.x**: For building ESP32-C6 firmware
- **Python 3.8+**: For ISA extraction and benchmark generation
- **ESP32-C6 Development Board**: For running benchmarks

### Full Workflow

```bash
# 1. Configure and build LLVM tools
./config.sh config
./config.sh build

# 2. Generate RISC-V instruction definitions
./config.sh genDefs

# 3. Extract valid ESP32-C6 instructions
python3 isa_extraction/extract_esp32_isa.py

# 4. Generate benchmark header
python3 benchmark_generator/generate_benchmarks.py

# 5. Build and flash ESP32-C6 firmware
cd esp32c6_benchmark
# Important: Source idf from esp32!
idf.py set-target esp32c6
idf.py build flash

# 6. Run benchmarks and capture output to file
idf.py monitor | tee benchmark_output.txt
# Press Ctrl+] to exit monitor after benchmarks complete
```

### Parse Results

After running the benchmarks, parse the captured output:

```bash
python3 benchmark_generator/parse_benchmark_results.py \
    --input benchmark_output.txt \
    --output report.json \
    --csv results.csv
```

## Configuration Script (`config.sh`)

| Command | Description |
|---------|-------------|
| `./config.sh download` | Download LLVM source |
| `./config.sh config` | Configure CMake build (auto-downloads if needed) |
| `./config.sh build` | Build llvm-mc and llvm-tblgen |
| `./config.sh genDefs` | Generate `isa_extraction/output/riscv_defs.json` |
| `./config.sh clean` | Remove build artifacts |
| `./config.sh mrproper` | Deep clean (build + externalDeps) |

## Methodology

Latency is measured using **dependency chains** where each instruction's output
becomes the next instruction's input, forcing sequential execution:

```
add a0, a0, a0   ; Cycle 1
add a0, a0, a0   ; Cycle 2 (depends on previous)
add a0, a0, a0   ; Cycle 3
... (100 times)
```

**Total cycles ÷ chain length = per-instruction latency**

**Throughput** is measured using independent instruction sequences (different
destination registers) to avoid dependencies per [uops.info](https://uops.info/) methodology.
We test sequence lengths of 1, 2, 4, and 8 instructions and report the minimum cycles per instruction
to find the optimal throughput (accounting for pipeline fill/issue width).

```
add a0, t0, t1   ; Cycle 1 (independent)
add a1, t0, t1   ; Cycle 1 (parallel execution)
...
```

**Total cycles ÷ instruction count = reciprocal throughput (cycles/instr)**

**Division Variants**: For division operations, we test with multiple operand values (Low/High latency pairs) to characterize data-dependent latency.

## Results Summary (ESP32-C6 @ 160MHz)

**154 benchmarks (104 latency + 50 throughput)**

| Category | Instructions | Latency | Throughput |
|----------|-------------|---------|------------|
| **Arithmetic** | ADD, ADDI, SUB, AND, ANDI, OR, ORI, XOR, XORI, SLT, SLTI, SLTIU, SLTU | 1 | 1 |
| **Shifts** | SLL, SLLI, SRA, SRAI, SRL, SRLI | 1 | 1 |
| **Compressed ALU** | C.ADD, C.ADDI, C.AND, C.ANDI, C.OR, C.XOR, C.SUB, C.MV, C.LI, C.SLLI, C.SRAI, C.SRLI | 1 | 1 |
| **Upper Immediate** | LUI, AUIPC | 1 | 1 |
| **Multiply (low)** | MUL | 1 | 1 |
| **Multiply (high)** | MULH, MULHSU, MULHU | 2 | 2 |
| **Division** | DIV, DIVU, REM, REMU | 10 | 10* |
| **Sign/Zero Extend** | SEXT.B, SEXT.H, ZEXT.H.RV32, ZEXT.H.RV64 | 2 | 2 |
| **Word Load** | LW, C.LW | 3 | 0-1 |
| **Atomic (AMO)** | AMOADD.W, AMOSWAP.W, etc. | 6 | 6 |
| **Branch (not-taken)** | BEQ, BNE | 1 | 1 |
| **Branch (taken/complex)** | BGE, BGEU, BLT, BLTU, C.BEQZ, C.BNEZ | 1-3 | 1-2 |
| **Jump (direct)** | C.J, C.JAL, JAL | 2-3 | 2 |

*Division is not pipelined, so throughput ≈ latency.

### Detailed Results

[View full detailed results in RESULTS.md](RESULTS.md)

## License

MIT License
