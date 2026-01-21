#!/usr/bin/env python3
"""
branch_predictor_generator.py

Specialized benchmark generator to determine branch predictor direction policy.

Experiments:
- Test A: Backward Loop (BNE jumping backwards) - BTFN predicts taken
- Test B: Forward Skip (BNE jumping forwards, taken) - BTFN predicts not-taken
- Test C: Forward Not-Taken (baseline)
- Test D: Backward Not-Taken - BTFN predicts taken (misprediction)
- Warmup Detection: Track cycles across iterations for dynamic BTB detection

Expected Results:
- Static BTFN: A=1, B=3, C=1, D=3 cycles
- Dynamic BTB: All start ~3, then converge to 1 after warmup
- No Predictor: All taken=3, all not-taken=1
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

try:
    from .common import BenchmarkConfig, LATENCY_TYPE_MAP
except ImportError:
    from common import BenchmarkConfig, LATENCY_TYPE_MAP


@dataclass
class BranchPredictorConfig:
    """Configuration for branch predictor benchmarks."""
    warmup_iterations: int = 100
    measurement_iterations: int = 5000
    repeat_count: int = 10
    loop_iterations: int = 100  # For backward loop test


class BranchPredictorBenchmarkGenerator:
    """
    Generates specialized benchmarks to determine branch predictor direction policy.
    
    Key insight: In RISC-V:
    - Backward branches use labels like "1b" (1 backward)
    - Forward branches use labels like "1f" (1 forward)
    
    Static BTFN predicts:
    - Backward branch = Taken
    - Forward branch = Not-Taken
    """
    
    def __init__(self, config: BranchPredictorConfig):
        self.config = config
        self.generated_benchmarks = []
    
    def _escape_asm(self, asm: str) -> str:
        """Escape assembly string for C string literal."""
        return asm.replace("\\", "\\\\").replace('"', '\\"')
    
    def generate_test_a_backward_taken(self) -> str:
        """
        Test A: Backward Loop (Taken Branch Jumping Backwards)
        
        This creates a loop where BNE jumps backwards to repeat.
        If BTFN: predicted taken → ~1 cycle per branch
        If mispredicted: ~3 cycles per branch
        """
        func = f'''
/* 
 * Test A: Backward Taken Branch (Loop)
 * BNE jumps backwards to repeat the loop.
 * Static BTFN should predict TAKEN for backward branches.
 * Expected: ~1 cycle if predicted correctly
 */
static int bench_branch_backward_taken(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end, elapsed;
    uint64_t min_cycles = UINT64_MAX;
    uint64_t total_cycles = 0;
    
    const int loop_count = {self.config.loop_iterations};
    
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        register int32_t counter __asm__("a0") = loop_count;
        
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        __asm__ volatile (
            "1:                     \\n"  /* Loop start label */
            "    addi %0, %0, -1    \\n"  /* Decrement counter */
            "    bne  %0, zero, 1b  \\n"  /* Backward branch if counter != 0 */
            : "+r"(counter)
            :
            : "memory"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        elapsed = end - start;
        /* Normalize to cycles per branch (loop_count branches executed) */
        uint32_t cycles_per_branch = elapsed / loop_count;
        
        if (cycles_per_branch < min_cycles) min_cycles = cycles_per_branch;
        total_cycles += cycles_per_branch;
    }}
    
    result->min_cycles = (uint32_t)min_cycles;
    result->avg_cycles = (uint32_t)(total_cycles / total_iterations);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append(("BRANCH_BACKWARD_TAKEN", "branch_backward_taken"))
        return func
    
    def generate_test_b_forward_taken(self) -> str:
        """
        Test B: Forward Taken Branch (Skip)
        
        BNE jumps forward over some NOPs.
        If BTFN: predicted NOT-taken → misprediction → ~3 cycles
        If correctly predicted: ~1 cycle
        """
        func = f'''
/*
 * Test B: Forward Taken Branch (Skip)
 * BNE jumps forward, skipping instructions.
 * Static BTFN should predict NOT-TAKEN for forward branches.
 * If branch is taken, this is a misprediction → ~3 cycles
 */
static int bench_branch_forward_taken(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end, elapsed;
    uint64_t min_cycles = UINT64_MAX;
    uint64_t total_cycles = 0;
    
    /* We'll measure multiple forward-taken branches in sequence */
    const int branch_count = {self.config.loop_iterations};
    
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        /* Setup: a0 != a1, so BNE is taken */
        register uint32_t a0_val __asm__("a0") = 1;
        register uint32_t a1_val __asm__("a1") = 0;
        register int32_t counter __asm__("a2") = branch_count;
        (void)a0_val; (void)a1_val;
        
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        /* 
         * Each iteration: forward taken branch jumps over NOPs 
         * We unroll a few to reduce loop overhead
         */
        __asm__ volatile (
            "1:                         \\n"  /* Outer loop */
            "    bne  a0, a1, 2f        \\n"  /* Forward taken branch */
            "    nop                    \\n"  /* Skipped */
            "    nop                    \\n"  /* Skipped */
            "2:                         \\n"  /* Target */
            "    bne  a0, a1, 3f        \\n"  /* Forward taken branch */
            "    nop                    \\n"  /* Skipped */
            "    nop                    \\n"  /* Skipped */
            "3:                         \\n"  /* Target */
            "    bne  a0, a1, 4f        \\n"  /* Forward taken branch */
            "    nop                    \\n"  /* Skipped */
            "    nop                    \\n"  /* Skipped */
            "4:                         \\n"  /* Target */
            "    bne  a0, a1, 5f        \\n"  /* Forward taken branch */
            "    nop                    \\n"  /* Skipped */
            "    nop                    \\n"  /* Skipped */
            "5:                         \\n"  /* Target */
            "    addi %0, %0, -4        \\n"  /* 4 branches per iteration */
            "    bgt  %0, zero, 1b      \\n"  /* Loop back (backward, predicted taken) */
            : "+r"(counter)
            :
            : "a0", "a1", "memory"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        elapsed = end - start;
        /* Normalize: branch_count forward branches + (branch_count/4) backward branches */
        /* We care about the forward branches, so approximate */
        uint32_t total_branches = branch_count;
        uint32_t cycles_per_branch = elapsed / total_branches;
        
        if (cycles_per_branch < min_cycles) min_cycles = cycles_per_branch;
        total_cycles += cycles_per_branch;
    }}
    
    result->min_cycles = (uint32_t)min_cycles;
    result->avg_cycles = (uint32_t)(total_cycles / total_iterations);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append(("BRANCH_FORWARD_TAKEN", "branch_forward_taken"))
        return func
    
    def generate_test_c_forward_not_taken(self) -> str:
        """
        Test C: Forward Not-Taken Branch (Fall-through)
        
        BNE with equal operands falls through (not taken).
        If BTFN: predicted NOT-taken → correct → ~1 cycle
        """
        func = f'''
/*
 * Test C: Forward Not-Taken Branch (Baseline)
 * BNE falls through (not taken) since a0 == a0.
 * Static BTFN should predict NOT-TAKEN for forward branches.
 * This should always be ~1 cycle (correctly predicted).
 */
static int bench_branch_forward_not_taken(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end, elapsed;
    uint64_t min_cycles = UINT64_MAX;
    uint64_t total_cycles = 0;
    
    const int branch_count = {self.config.loop_iterations};
    
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        register int32_t counter __asm__("a2") = branch_count;
        
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        __asm__ volatile (
            "1:                         \\n"
            "    bne  a0, a0, 2f        \\n"  /* Never taken: a0 == a0 */
            "2:                         \\n"
            "    bne  a0, a0, 3f        \\n"
            "3:                         \\n"
            "    bne  a0, a0, 4f        \\n"
            "4:                         \\n"
            "    bne  a0, a0, 5f        \\n"
            "5:                         \\n"
            "    addi %0, %0, -4        \\n"
            "    bgt  %0, zero, 1b      \\n"
            : "+r"(counter)
            :
            : "memory"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        elapsed = end - start;
        uint32_t cycles_per_branch = elapsed / branch_count;
        
        if (cycles_per_branch < min_cycles) min_cycles = cycles_per_branch;
        total_cycles += cycles_per_branch;
    }}
    
    result->min_cycles = (uint32_t)min_cycles;
    result->avg_cycles = (uint32_t)(total_cycles / total_iterations);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append(("BRANCH_FORWARD_NOT_TAKEN", "branch_forward_not_taken"))
        return func
    
    def generate_test_d_backward_not_taken(self) -> str:
        """
        Test D: Backward Not-Taken Branch
        
        BNE with equal operands targets a backward label but doesn't jump.
        If BTFN: predicted TAKEN → misprediction → ~3 cycles
        
        This is tricky to set up since we need a backward label reference
        that won't actually be jumped to.
        """
        func = f'''
/*
 * Test D: Backward Not-Taken Branch
 * BNE references a backward label but falls through (not taken).
 * Static BTFN should predict TAKEN for backward branches.
 * If not taken, this is a misprediction → ~3 cycles
 */
static int bench_branch_backward_not_taken(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end, elapsed;
    uint64_t min_cycles = UINT64_MAX;
    uint64_t total_cycles = 0;
    
    const int branch_count = {self.config.loop_iterations};
    
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        register int32_t counter __asm__("a2") = branch_count / 4;
        
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        /*
         * Structure: jump forward, then have backward-referring BNE that doesn't take
         * 
         * Label 1 is placed before some nops, then we have BNE referencing 1b
         * but with equal operands, so it won't jump.
         */
        __asm__ volatile (
            "    j 10f                  \\n"  /* Skip to start of test */
            "1:  nop                    \\n"  /* Backward target (never reached) */
            "    j 99f                  \\n"  /* Safety exit */
            "10:                        \\n"  /* Test start */
            "11:                        \\n"  /* Loop start */
            "    bne  a0, a0, 1b        \\n"  /* Backward not-taken: a0 == a0 */
            "12: nop                    \\n"
            "    bne  a0, a0, 12b       \\n"  /* Backward not-taken */
            "13: nop                    \\n"
            "    bne  a0, a0, 13b       \\n"  /* Backward not-taken */
            "14: nop                    \\n"
            "    bne  a0, a0, 14b       \\n"  /* Backward not-taken */
            "    addi %0, %0, -1        \\n"
            "    bgt  %0, zero, 11b     \\n"  /* Loop control (taken) */
            "99:                        \\n"  /* Exit */
            : "+r"(counter)
            :
            : "memory"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        elapsed = end - start;
        uint32_t cycles_per_branch = elapsed / branch_count;
        
        if (cycles_per_branch < min_cycles) min_cycles = cycles_per_branch;
        total_cycles += cycles_per_branch;
    }}
    
    result->min_cycles = (uint32_t)min_cycles;
    result->avg_cycles = (uint32_t)(total_cycles / total_iterations);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append(("BRANCH_BACKWARD_NOT_TAKEN", "branch_backward_not_taken"))
        return func
    
    def generate_warmup_detection(self) -> str:
        """
        Warmup Detection Test
        
        Track per-iteration cycles to detect if prediction improves over time.
        If dynamic BTB: first iterations show ~3 cycles, later ~1 cycle
        If static: consistent throughout
        
        Reports first 10 iteration timings for analysis.
        """
        func = f'''
/*
 * Warmup Detection: Dynamic BTB Detector
 * 
 * Measures per-iteration cycle counts to detect dynamic prediction learning.
 * Dynamic BTB: early iterations ~3 cycles, later ~1 cycle
 * Static BTFN: consistent throughout
 * 
 * Reports statistics that show warmup behavior.
 */
static int bench_branch_warmup_detection(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end;
    
    const int branch_count = 100;  /* Fixed for consistency */
    
    /* Track first vs last iteration groups */
    uint64_t early_total = 0;  /* First 10 iterations */
    uint64_t late_total = 0;   /* Last 10 iterations */
    uint64_t min_late = UINT64_MAX;
    
    const int early_count = 10;
    const int late_start = total_iterations > 20 ? total_iterations - 10 : 10;
    
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        /* Setup for forward taken branch */
        register uint32_t a0_val __asm__("a0") = 1;
        register uint32_t a1_val __asm__("a1") = 0;
        register int32_t counter __asm__("a2") = branch_count;
        (void)a0_val; (void)a1_val;
        
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        __asm__ volatile (
            "1:                         \\n"
            "    bne  a0, a1, 2f        \\n"  /* Forward taken */
            "    nop                    \\n"
            "2:                         \\n"
            "    addi %0, %0, -1        \\n"
            "    bgt  %0, zero, 1b      \\n"
            : "+r"(counter)
            :
            : "a0", "a1", "memory"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        uint32_t elapsed = end - start;
        uint32_t cycles_per_branch = elapsed / branch_count;
        
        /* Track early vs late performance */
        if (rep < early_count) {{
            early_total += cycles_per_branch;
        }}
        if (rep >= late_start) {{
            late_total += cycles_per_branch;
            if (cycles_per_branch < min_late) min_late = cycles_per_branch;
        }}
    }}
    
    /* 
     * Interpretation:
     * - If early_avg >> late_avg: Dynamic BTB detected (learning curve)
     * - If early_avg ≈ late_avg: Static predictor or no learning
     */
    uint32_t early_avg = early_total / early_count;
    uint32_t late_count = total_iterations - late_start;
    uint32_t late_avg = late_total / late_count;
    
    /* Pack early vs late into min (early) and avg (late) for comparison */
    result->min_cycles = early_avg;  /* Early iterations avg */
    result->avg_cycles = late_avg;   /* Late iterations avg */
    result->max_cycles = (early_avg > late_avg) ? (early_avg - late_avg) : 0; /* Delta */
    result->total_iterations = total_iterations;
    result->status = 0;
    return 0;
}}
'''
        self.generated_benchmarks.append(("BRANCH_WARMUP_DETECTION", "branch_warmup_detection"))
        return func
    
    def generate_all_benchmarks(self) -> List[str]:
        """Generate all branch predictor test functions."""
        functions = [
            self.generate_test_a_backward_taken(),
            self.generate_test_b_forward_taken(),
            self.generate_test_c_forward_not_taken(),
            self.generate_test_d_backward_not_taken(),
            self.generate_warmup_detection(),
        ]
        return functions
    
    def generate_descriptor_entry(self, name: str, func_name: str) -> str:
        """Generate a benchmark descriptor entry."""
        return f'''    {{
        .instruction_name = "{name}",
        .asm_syntax = "Branch Predictor Test",
        .latency_type = LAT_TYPE_BRANCH,
        .bench_type = BENCH_TYPE_LATENCY,
        .category = BENCH_CAT_CONTROL,
        .run_benchmark = bench_{func_name}
    }}'''
    
    def generate_all_descriptors(self) -> List[str]:
        """Generate descriptor entries for all benchmarks."""
        descriptors = []
        for name, func_name in self.generated_benchmarks:
            descriptors.append(self.generate_descriptor_entry(name, func_name))
        return descriptors


def generate_branch_predictor_header(output_path: str, config: BranchPredictorConfig) -> None:
    """Generate a standalone header file for branch predictor experiments."""
    from datetime import datetime
    
    generator = BranchPredictorBenchmarkGenerator(config)
    functions = generator.generate_all_benchmarks()
    descriptors = generator.generate_all_descriptors()
    
    timestamp = datetime.now().isoformat()
    
    header = f'''/**
 * @file branch_predictor_benchmarks.h
 * @brief Branch Predictor Direction Experiment for ESP32-C6
 *
 * Generated: {timestamp}
 * Generator: branch_predictor_generator.py
 *
 * Experiments to determine if ESP32-C6 uses:
 * - Static BTFN (Backward-Taken, Forward-Not-Taken) prediction
 * - Dynamic BTB (Branch Target Buffer) with learning
 *
 * Expected Results (Static BTFN):
 * - Test A (Backward Taken): ~1 cycle (correct prediction)
 * - Test B (Forward Taken): ~3 cycles (misprediction)
 * - Test C (Forward Not-Taken): ~1 cycle (correct prediction)
 * - Test D (Backward Not-Taken): ~3 cycles (misprediction)
 *
 * Expected Results (Dynamic BTB):
 * - All tests: ~3 cycles initially, ~1 cycle after warmup
 */

#ifndef BRANCH_PREDICTOR_BENCHMARKS_H
#define BRANCH_PREDICTOR_BENCHMARKS_H

#include "benchmark_interface.h"
#include <limits.h>

/*
 * =============================================================================
 * Benchmark Configuration
 * =============================================================================
 */

const benchmark_config_t BENCHMARK_CONFIG = {{
    .warmup_iterations = {config.warmup_iterations},
    .measurement_iterations = {config.measurement_iterations},
    .repeat_count = {config.repeat_count},
    .chain_length = {config.loop_iterations}
}};

const char *BENCHMARK_SET_NAME = "ESP32-C6 Branch Predictor Direction Experiment";

/*
 * =============================================================================
 * Benchmark Function Implementations
 * =============================================================================
 */

{"".join(functions)}

/*
 * =============================================================================
 * Benchmark Descriptors ({len(descriptors)} tests)
 * =============================================================================
 */

const benchmark_descriptor_t BENCHMARKS[] = {{
{",".join(descriptors)}
}};

const size_t BENCHMARK_COUNT = sizeof(BENCHMARKS) / sizeof(BENCHMARKS[0]);

#endif /* BRANCH_PREDICTOR_BENCHMARKS_H */
'''
    
    with open(output_path, 'w') as f:
        f.write(header)
    
    print(f"Generated branch predictor benchmarks: {output_path}")
    print(f"  Tests: {len(generator.generated_benchmarks)}")
    for name, _ in generator.generated_benchmarks:
        print(f"    - {name}")


if __name__ == "__main__":
    import argparse
    
    # Use absolute paths based on script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    default_output = project_root / "esp32c6_benchmark" / "main" / "generated_benchmarks.h"
    
    parser = argparse.ArgumentParser(
        description="Generate branch predictor direction experiment benchmarks"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(default_output),
        help="Output header file path"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup iterations (default: 100)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
        help="Measurement iterations (default: 5000)"
    )
    parser.add_argument(
        "--loop-iterations",
        type=int,
        default=100,
        help="Branch iterations per measurement (default: 100)"
    )
    
    args = parser.parse_args()
    
    config = BranchPredictorConfig(
        warmup_iterations=args.warmup,
        measurement_iterations=args.iterations,
        loop_iterations=args.loop_iterations
    )
    
    generate_branch_predictor_header(args.output, config)
