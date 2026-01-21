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
│   ├── generate_benchmarks.py         # Main entry point for benchmark generation
│   ├── latency_generator.py           # Latency benchmark generator
│   ├── throughput_generator.py        # Throughput benchmark generator
│   ├── cache_benchmarks.py            # Cache latency benchmarks
│   ├── branch_predictor_generator.py  # Branch predictor tests
│   ├── common.py                      # Shared utilities
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

**254 benchmarks (112 latency + 125 throughput + 9 cache + 5 branch predictor + 1 structural hazard + 2 store buffer)**

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

### Cache & Instruction Fetch (Memory Subsystem)

| Benchmark | Latency | CPI | Notes |
| :--- | :--- | :--- | :--- |
| **SRAM Access** | 3 cyc | - | Internal RAM baseline |
| **Flash Cache Hit** | 4 cyc | - | Small RO data in cache |
| **Flash Cache Miss** | ~347 cyc | - | Large RO data evicting cache |
| **Unaligned Load** | 2 cyc | - | Hard 1-cycle penalty (vs 1 cyc aligned) |
| **Fetch Linear** | - | 1.0 | 4KB NOP block (Flash) |
| **Fetch Branchy** | - | 3.0 | 4KB Jump to next (Flash) |

### Cache Replacement Policy Logic

To determine the cache eviction policy (LRU vs Pseudo-LRU/Random), we use a **"5-pointer Thrash"** test on the 4-way set associative cache:

1.  **Prime**: Access 4 lines ($P0 \to P1 \to P2 \to P3$) mapping to the same cache set. This fills the set. If the policy is LRU, $P0$ (accessed first) becomes the Least Recently Used line.
2.  **Thrash**: Access a 5th line ($P4$) mapping to the same set. This forces the eviction of one line.
3.  **Probe**: Measure the latency of accessing $P0$ again.
    - **High Latency (~Miss)**: $P0$ was evicted. This strongly suggests an **LRU** policy (as $P0$ was the oldest).
    - **Low Latency (~Hit)**: $P0$ survived. This suggests a **Random** or **Pseudo-LRU** policy.

**Experimental Results**:
- **Probe Latency**: Min 123 cycles, Avg 693 cycles.
- **Baseline Hit**: 4 cycles.
- **Conclusion**: The high latency (>> 4 cycles) and the fact that `min_cycles` is never a hit indicates that $P0$ is **consistently evicted**. This confirms the **LRU (Least Recently Used)** replacement policy.

### Structural Hazards Logic

To determine if the **Integer Divider** blocks the pipeline (preventing parallel execution of independent instructions), we measure a mixed sequence of **DIV** (10 cycles) and **ADD** (1 cycle):

-   **Hypothesis A (Non-Blocking)**: DIV executes in background. Total = max(DIV, ADD) ≈ 10 cycles.
-   **Hypothesis B (Blocking)**: DIV stalls the pipeline. Total = DIV + ADD ≈ 11 cycles.

**Experimental Results**:
-   **DIV + ADD Sequence**: 11 cycles.
-   **Conclusion**: The divider on the ESP32-C6 is **BLOCKING**. It stalls the main pipeline until the division is complete, preventing instruction overlap.

### Store Buffer Logic

To characterize the **Store Buffer**, we perform two tests:

1.  **Store Burst**: 100 consecutive `SW` instructions to a buffer.
    -   **Result**: 104 cycles for 100 instructions (~1.04 CPI).
    -   **Conclusion**: Validates that the ESP32-C6 has a **Write Buffer** capable of absorbing bursty stores at near 1 cycle/instruction rate, preventing pipeline stalls.
2.  **Store-to-Load Forwarding**: Pairs of `SW` followed immediately by `LW` to the same address.
    -   **Result**: 3 cycles per pair.
    -   **Conclusion**: **Efficient Forwarding**. Data can be read back from the store buffer with minimal latency (~3 cycles total for the pair), significantly faster than waiting for a cache write-back and re-read.

### Detailed Results

[View full detailed results in RESULTS.md](RESULTS.md)

## License

MIT License
