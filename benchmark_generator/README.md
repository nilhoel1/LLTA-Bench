# Benchmark Generator

This folder contains Python tools for generating and parsing ESP32-C6 instruction latency benchmarks.

## Tools

### `generate_latency_benchmarks.py`

Generates C header files containing latency benchmarks from extracted ISA definitions.

**Usage:**

```bash
# From project root - uses default paths
python3 benchmark_generator/generate_latency_benchmarks.py

# Or with explicit paths
python3 benchmark_generator/generate_latency_benchmarks.py \
    --input isa_extraction/output/esp32c6_instructions.json \
    --output esp32c6_benchmark/main/generated_benchmarks.h
```

**Options:**
- `--input, -i`: Input JSON file with instruction definitions
  (default: `isa_extraction/output/esp32c6_instructions.json`)
- `--output, -o`: Output header file path
  (default: `esp32c6_benchmark/main/generated_benchmarks.h`)
- `--warmup`: Number of warmup iterations (default: 100)
- `--iterations`: Number of measurement iterations (default: 1000)
- `--repeats`: Number of times to repeat measurement (default: 5)
- `--chain-length`: Length of dependency chain (default: 100)

### `parse_benchmark_results.py`

Parses benchmark results from ESP32-C6 serial output and generates reports.

**Usage:**

```bash
# Capture output from monitor, then parse
python3 benchmark_generator/parse_benchmark_results.py \
    --input benchmark_output.txt \
    --output report.json \
    --csv results.csv
```

**Options:**
- `--input, -i`: Input file with captured benchmark output (required)
- `--output, -o`: Output JSON report file
- `--csv`: Output CSV file
- `--quiet, -q`: Suppress table output

## Workflow

1. **Extract ISA** (from project root):
   ```bash
   ./config.sh genDefs
   python3 isa_extraction/extract_esp32_isa.py
   ```

2. **Generate benchmarks**:
   ```bash
   python3 benchmark_generator/generate_latency_benchmarks.py
   ```

3. **Build and run** (from esp32c6_benchmark/):
   ```bash
   cd esp32c6_benchmark
   idf.py build flash monitor | tee ../benchmark_output.txt
   ```

4. **Parse results**:
   ```bash
   python3 benchmark_generator/parse_benchmark_results.py \
       --input benchmark_output.txt \
       --output report.json
   ```

## Methodology

The generator uses the **dependency chain** methodology from the uops.info paper
(ASPLOS '19). Each instruction's output register becomes the input for the next
instruction, forcing sequential execution. The cycle count divided by chain
length gives per-instruction latency.

### Skipped Instructions

Some instructions cannot be measured with this methodology:
- **Atomics**: Need separate address/data registers
- **Branches/Jumps**: Change control flow
- **Stores**: Don't produce register output
- **Byte/Half loads**: Can't chain with self-referential pointers
- **SP instructions**: Corrupt stack pointer
- **RV64 hints**: Illegal on RV32 ESP32-C6

See the class docstring in `generate_latency_benchmarks.py` for detailed
measurement strategies for these instruction categories.
