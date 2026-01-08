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
python3 benchmark_generator/generate_latency_benchmarks.py

# 5. Build and flash ESP32-C6 firmware
cd esp32c6_benchmark
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

## Results Summary (ESP32-C6 @ 160MHz)

| Category | Instructions | Latency (cycles) |
|----------|-------------|------------------|
| Basic ALU | ADD, SUB, AND, OR, XOR, SLT, shifts | 1 |
| Compressed ALU | C.ADD, C.SUB, C.AND, etc. | 1 |
| Multiply | MUL | 1 |
| Multiply High | MULH, MULHSU, MULHU | 2 |
| Division | DIV, DIVU, REM, REMU | 10 |
| Sign/Zero Extend | SEXT.B, SEXT.H, ZEXT.H | 2 |
| Word Load | LW, C.LW | 3 |

## License

MIT License
