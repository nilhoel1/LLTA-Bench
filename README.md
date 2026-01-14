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

## Results Summary (ESP32-C6 @ 160MHz)

**97 benchmarks executed successfully**

| Category | Instructions | Latency (cycles) |
|----------|-------------|------------------|
| **Arithmetic** | ADD, ADDI, SUB, AND, ANDI, OR, ORI, XOR, XORI, SLT, SLTI, SLTIU, SLTU | 1 |
| **Shifts** | SLL, SLLI, SRA, SRAI, SRL, SRLI | 1 |
| **Compressed ALU** | C.ADD, C.ADDI, C.AND, C.ANDI, C.OR, C.XOR, C.SUB, C.MV, C.LI, C.SLLI, C.SRAI, C.SRLI | 1 |
| **Upper Immediate** | LUI, AUIPC | 1 |
| **Multiply (low)** | MUL | 1 |
| **Multiply (high)** | MULH, MULHSU, MULHU | 2 |
| **Division** | DIV, DIVU, REM, REMU | 10 |
| **Sign/Zero Extend** | SEXT.B, SEXT.H, ZEXT.H | 2 |
| **Word Load** | LW, C.LW | 3 |
| **Atomic (AMO)** | AMOADD.W, AMOSWAP.W, AMOAND.W, AMOOR.W, AMOXOR.W, AMOMAX.W, AMOMIN.W, etc. | 6 |
| **Branch (not-taken)** | BEQ, BNE | 1 |
| **Branch (not-taken)** | BGE, BGEU, BLT, BLTU, C.BEQZ, C.BNEZ | 3-4 |
| **Jump (direct)** | C.J, C.JAL | 2 |
| **Jump (direct)** | JAL | 3 |

### Detailed Results

<details>
<summary>Click to expand full results table</summary>

| Instruction | Assembly | Type | Avg Cycles |
|-------------|----------|------|------------|
| ADD | `add a0, a0, a0` | arithmetic | 1 |
| ADDI | `addi a0, a0, 0` | arithmetic | 1 |
| AMOADD_W | `amoadd.w a0, a0, (a0)` | atomic | 6 |
| AMOAND_W | `amoand.w a0, a0, (a0)` | atomic | 6 |
| AMOMAX_W | `amomax.w a0, a0, (a0)` | atomic | 6 |
| AMOMAXU_W | `amomaxu.w a0, a0, (a0)` | atomic | 6 |
| AMOMIN_W | `amomin.w a0, a0, (a0)` | atomic | 6 |
| AMOMINU_W | `amominu.w a0, a0, (a0)` | atomic | 6 |
| AMOOR_W | `amoor.w a0, a0, (a0)` | atomic | 6 |
| AMOSWAP_W | `amoswap.w a0, a0, (a0)` | atomic | 6 |
| AMOXOR_W | `amoxor.w a0, a0, (a0)` | atomic | 6 |
| AND | `and a0, a0, a0` | arithmetic | 1 |
| ANDI | `andi a0, a0, 0` | arithmetic | 1 |
| AUIPC | `auipc a0, 0` | unknown | 1 |
| BEQ | `beq a0, a0, 0` | branch | 1 |
| BGE | `bge a0, a0, 0` | branch | 4 |
| BGEU | `bgeu a0, a0, 0` | branch | 4 |
| BLT | `blt a0, a0, 0` | branch | 4 |
| BLTU | `bltu a0, a0, 0` | branch | 4 |
| BNE | `bne a0, a0, 0` | branch | 1 |
| C_ADD | `c.add a0, a0` | arithmetic | 1 |
| C_ADDI | `c.addi a0, 1` | arithmetic | 1 |
| C_AND | `c.and a0, a0` | arithmetic | 1 |
| C_ANDI | `c.andi a0, 0` | arithmetic | 1 |
| C_BEQZ | `c.beqz a0, 0` | branch | 3 |
| C_BNEZ | `c.bnez a0, 0` | branch | 3 |
| C_J | `c.j 0` | jump | 2 |
| C_JAL | `c.jal 0` | jump | 2 |
| C_LI | `c.li a0, 0` | arithmetic | 1 |
| C_LW | `c.lw a0, 0(a0)` | load_store | 3 |
| C_MV | `c.mv a0, a0` | arithmetic | 1 |
| C_OR | `c.or a0, a0` | arithmetic | 1 |
| C_SLLI | `c.slli a0, 1` | arithmetic | 1 |
| C_SRAI | `c.srai a0, 1` | arithmetic | 1 |
| C_SRLI | `c.srli a0, 1` | arithmetic | 1 |
| C_SUB | `c.sub a0, a0` | arithmetic | 1 |
| C_XOR | `c.xor a0, a0` | arithmetic | 1 |
| DIV | `div a0, a0, a0` | multiply | 10 |
| DIVU | `divu a0, a0, a0` | multiply | 10 |
| JAL | `jal a0, 0` | jump | 3 |
| LUI | `lui a0, 0` | unknown | 1 |
| LW | `lw a0, 0(a0)` | load | 3 |
| MUL | `mul a0, a0, a0` | multiply | 1 |
| MULH | `mulh a0, a0, a0` | multiply | 2 |
| MULHSU | `mulhsu a0, a0, a0` | multiply | 2 |
| MULHU | `mulhu a0, a0, a0` | multiply | 2 |
| OR | `or a0, a0, a0` | arithmetic | 1 |
| ORI | `ori a0, a0, 0` | arithmetic | 1 |
| REM | `rem a0, a0, a0` | multiply | 10 |
| REMU | `remu a0, a0, a0` | multiply | 10 |
| SEXT_B | `sext.b a0, a0` | unknown | 2 |
| SEXT_H | `sext.h a0, a0` | unknown | 2 |
| SLL | `sll a0, a0, a0` | arithmetic | 1 |
| SLLI | `slli a0, a0, 1` | arithmetic | 1 |
| SLT | `slt a0, a0, a0` | arithmetic | 1 |
| SLTI | `slti a0, a0, 0` | arithmetic | 1 |
| SLTIU | `sltiu a0, a0, 0` | arithmetic | 1 |
| SLTU | `sltu a0, a0, a0` | arithmetic | 1 |
| SRA | `sra a0, a0, a0` | arithmetic | 1 |
| SRAI | `srai a0, a0, 1` | arithmetic | 1 |
| SRL | `srl a0, a0, a0` | arithmetic | 1 |
| SRLI | `srli a0, a0, 1` | arithmetic | 1 |
| SUB | `sub a0, a0, a0` | arithmetic | 1 |
| XOR | `xor a0, a0, a0` | arithmetic | 1 |
| XORI | `xori a0, a0, 0` | arithmetic | 1 |
| ZEXT_H | `zext.h a0, a0` | unknown | 2 |

*Note: Atomic instructions with `.aq`, `.rl`, and `.aqrl` suffixes all measure 6 cycles.*

</details>

## License

MIT License
