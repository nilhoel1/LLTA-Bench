/**
 * @file benchmark_interface.h
 * @brief Interface definition for ESP32-C6 instruction benchmarks
 *
 * This header defines the interface that generated benchmark headers must implement.
 * The main.c relies on these definitions to run benchmarks in a modular way.
 */

#ifndef BENCHMARK_INTERFACE_H
#define BENCHMARK_INTERFACE_H

#include <stdint.h>
#include <stddef.h>
#include "esp_cpu.h"  /* For esp_cpu_get_cycle_count() */

/**
 * @brief Benchmark types supported by the framework
 */
typedef enum {
    BENCH_TYPE_LATENCY = 0,
    BENCH_TYPE_THROUGHPUT,
    BENCH_TYPE_PORT_USAGE,
    BENCH_TYPE_UNKNOWN
} benchmark_type_t;

/**
 * @brief Latency type categories for instructions
 */
typedef enum {
    LAT_TYPE_ARITHMETIC = 0,
    LAT_TYPE_LOAD,
    LAT_TYPE_STORE,
    LAT_TYPE_LOAD_STORE,
    LAT_TYPE_BRANCH,
    LAT_TYPE_JUMP,
    LAT_TYPE_MULTIPLY,
    LAT_TYPE_ATOMIC,
    LAT_TYPE_SYSTEM,
    LAT_TYPE_UNKNOWN
} latency_type_t;

/**
 * @brief Result of a single benchmark measurement
 */
typedef struct {
    uint64_t min_cycles;
    uint64_t max_cycles;
    uint64_t avg_cycles;
    uint64_t total_iterations;
    int32_t  status;  // 0 = success, negative = error
} benchmark_result_t;

/**
 * @brief Descriptor for a single benchmark
 */
typedef struct {
    const char *instruction_name;    // LLVM enum name (e.g., "ADD", "ADDI")
    const char *asm_syntax;          // Assembly syntax (e.g., "add a0, a0, a0")
    latency_type_t latency_type;     // Category of instruction
    benchmark_type_t bench_type;     // Type of benchmark (latency, throughput, etc.)

    /**
     * @brief Function pointer to run the benchmark
     * @param iterations Number of times to run the inner loop
     * @param result Pointer to store the result
     * @return 0 on success, negative on error
     */
    int (*run_benchmark)(uint32_t iterations, benchmark_result_t *result);
} benchmark_descriptor_t;

/**
 * @brief Configuration for benchmark execution
 */
typedef struct {
    uint32_t warmup_iterations;      // Iterations for cache warmup
    uint32_t measurement_iterations; // Iterations for actual measurement
    uint32_t repeat_count;           // Number of times to repeat measurement
    uint32_t chain_length;           // Length of dependency chain (for latency)
} benchmark_config_t;

/*
 * =============================================================================
 * The following symbols MUST be defined by the generated benchmark header
 * =============================================================================
 */

/**
 * @brief Array of benchmark descriptors (defined in generated header)
 */
extern const benchmark_descriptor_t BENCHMARKS[];

/**
 * @brief Number of benchmarks in the array (defined in generated header)
 */
extern const size_t BENCHMARK_COUNT;

/**
 * @brief Default configuration (defined in generated header)
 */
extern const benchmark_config_t BENCHMARK_CONFIG;

/**
 * @brief Name/version of the generated benchmark set
 */
extern const char *BENCHMARK_SET_NAME;

/*
 * =============================================================================
 * Helper macros for benchmark implementation
 * =============================================================================
 */

/**
 * @brief Read cycle counter using ESP-IDF API
 * Note: rdcycle is not accessible in user mode on ESP32-C6,
 * so we use the ESP-IDF provided function which reads mcycle.
 */
#define READ_CYCLE_COUNTER() ((uint32_t)esp_cpu_get_cycle_count())

/**
 * @brief Memory barrier to prevent instruction reordering
 */
#define MEMORY_BARRIER() __asm__ volatile ("fence" ::: "memory")

/**
 * @brief Compiler barrier to prevent optimization across this point
 */
#define COMPILER_BARRIER() __asm__ volatile ("" ::: "memory")

/**
 * @brief Macro to define a latency benchmark function
 *
 * @param name Function name suffix
 * @param setup_code Code to run before measurement (can initialize registers)
 * @param chain_asm Assembly for one iteration of the dependency chain
 * @param chain_count Number of instructions in the chain per iteration
 */
#define DEFINE_LATENCY_BENCHMARK(name, setup_code, chain_asm, chain_count)     \
    static int bench_latency_##name(uint32_t iterations,                       \
                                    benchmark_result_t *result) {              \
        uint32_t start, end, elapsed;                                          \
        uint64_t total = 0;                                                    \
        uint64_t min_cycles = UINT64_MAX;                                      \
        uint64_t max_cycles = 0;                                               \
                                                                               \
        setup_code;                                                            \
        COMPILER_BARRIER();                                                    \
                                                                               \
        for (uint32_t rep = 0; rep < iterations; rep++) {                      \
            COMPILER_BARRIER();                                                \
            start = READ_CYCLE_COUNTER();                                      \
            __asm__ volatile (                                                 \
                chain_asm                                                      \
                ::: "a0", "a1", "a2", "a3", "a4", "a5", "memory"               \
            );                                                                 \
            end = READ_CYCLE_COUNTER();                                        \
            COMPILER_BARRIER();                                                \
                                                                               \
            elapsed = end - start;                                             \
            total += elapsed;                                                  \
            if (elapsed < min_cycles) min_cycles = elapsed;                    \
            if (elapsed > max_cycles) max_cycles = elapsed;                    \
        }                                                                      \
                                                                               \
        result->min_cycles = min_cycles / (chain_count);                       \
        result->max_cycles = max_cycles / (chain_count);                       \
        result->avg_cycles = total / (iterations * (chain_count));             \
        result->total_iterations = iterations;                                 \
        result->status = 0;                                                    \
        return 0;                                                              \
    }

/**
 * @brief Macro to create a benchmark descriptor entry
 */
#define BENCHMARK_ENTRY(llvm_name, asm_str, lat_type, bench_func) {            \
    .instruction_name = llvm_name,                                             \
    .asm_syntax = asm_str,                                                     \
    .latency_type = lat_type,                                                  \
    .bench_type = BENCH_TYPE_LATENCY,                                          \
    .run_benchmark = bench_func                                                \
}

#endif /* BENCHMARK_INTERFACE_H */
