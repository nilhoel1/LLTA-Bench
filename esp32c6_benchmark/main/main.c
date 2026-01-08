/**
 * @file main.c
 * @brief ESP32-C6 Instruction Latency Benchmark Runner
 * 
 * This file provides a modular framework for running instruction benchmarks
 * on the ESP32-C6 (RISC-V). It reads benchmark definitions from a generated
 * header file and executes them, printing results over serial.
 * 
 * DO NOT MODIFY THIS FILE - benchmark specifics should be in generated headers.
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"

/* Include the benchmark interface definition */
#include "benchmark_interface.h"

/* Include the generated benchmarks - this is the only file that changes */
#include "generated_benchmarks.h"

static const char *TAG = "BENCH";

/*
 * =============================================================================
 * Utility Functions
 * =============================================================================
 */

/**
 * @brief Convert latency type enum to string
 */
static const char* latency_type_to_string(latency_type_t type) {
    switch (type) {
        case LAT_TYPE_ARITHMETIC: return "arithmetic";
        case LAT_TYPE_LOAD:       return "load";
        case LAT_TYPE_STORE:      return "store";
        case LAT_TYPE_LOAD_STORE: return "load_store";
        case LAT_TYPE_BRANCH:     return "branch";
        case LAT_TYPE_JUMP:       return "jump";
        case LAT_TYPE_MULTIPLY:   return "multiply";
        case LAT_TYPE_ATOMIC:     return "atomic";
        case LAT_TYPE_SYSTEM:     return "system";
        case LAT_TYPE_UNKNOWN:
        default:                  return "unknown";
    }
}

/**
 * @brief Convert benchmark type enum to string
 */
static const char* benchmark_type_to_string(benchmark_type_t type) {
    switch (type) {
        case BENCH_TYPE_LATENCY:    return "latency";
        case BENCH_TYPE_THROUGHPUT: return "throughput";
        case BENCH_TYPE_PORT_USAGE: return "port_usage";
        case BENCH_TYPE_UNKNOWN:
        default:                    return "unknown";
    }
}

/**
 * @brief Print a separator line
 */
static void print_separator(void) {
    printf("--------------------------------------------------------------------------------\n");
}

/**
 * @brief Print benchmark header information
 */
static void print_header(void) {
    printf("\n");
    print_separator();
    printf("ESP32-C6 Instruction Benchmark Results\n");
    printf("Benchmark Set: %s\n", BENCHMARK_SET_NAME);
    printf("Total Benchmarks: %zu\n", BENCHMARK_COUNT);
    printf("Configuration:\n");
    printf("  Warmup Iterations:      %" PRIu32 "\n", BENCHMARK_CONFIG.warmup_iterations);
    printf("  Measurement Iterations: %" PRIu32 "\n", BENCHMARK_CONFIG.measurement_iterations);
    printf("  Repeat Count:           %" PRIu32 "\n", BENCHMARK_CONFIG.repeat_count);
    printf("  Chain Length:           %" PRIu32 "\n", BENCHMARK_CONFIG.chain_length);
    print_separator();
    printf("\n");
}

/**
 * @brief Print result in JSON format for easy parsing
 */
static void print_result_json(const benchmark_descriptor_t *bench, 
                               const benchmark_result_t *result) {
    printf("{\"instruction\":\"%s\",\"asm\":\"%s\",\"type\":\"%s\","
           "\"latency_type\":\"%s\",\"min_cycles\":%" PRIu64 ","
           "\"max_cycles\":%" PRIu64 ",\"avg_cycles\":%" PRIu64 ","
           "\"iterations\":%" PRIu64 ",\"status\":%d}\n",
           bench->instruction_name,
           bench->asm_syntax,
           benchmark_type_to_string(bench->bench_type),
           latency_type_to_string(bench->latency_type),
           result->min_cycles,
           result->max_cycles,
           result->avg_cycles,
           result->total_iterations,
           (int)result->status);
}

/**
 * @brief Print result in human-readable format
 */
static void print_result_human(size_t index, 
                                const benchmark_descriptor_t *bench,
                                const benchmark_result_t *result) {
    printf("[%3zu/%3zu] %-20s | %-25s | %s\n",
           index + 1, BENCHMARK_COUNT,
           bench->instruction_name,
           bench->asm_syntax,
           latency_type_to_string(bench->latency_type));
    
    if (result->status == 0) {
        printf("          Cycles: min=%" PRIu64 " max=%" PRIu64 " avg=%" PRIu64 "\n",
               result->min_cycles, result->max_cycles, result->avg_cycles);
    } else {
        printf("          ERROR: status=%d\n", (int)result->status);
    }
}

/*
 * =============================================================================
 * Benchmark Execution
 * =============================================================================
 */

/**
 * @brief Run warmup iterations for a benchmark
 */
static void run_warmup(const benchmark_descriptor_t *bench) {
    benchmark_result_t dummy_result;
    
    if (bench->run_benchmark != NULL) {
        bench->run_benchmark(BENCHMARK_CONFIG.warmup_iterations, &dummy_result);
    }
}

/**
 * @brief Run a single benchmark with multiple repetitions
 */
static int run_benchmark_with_stats(const benchmark_descriptor_t *bench,
                                     benchmark_result_t *final_result) {
    benchmark_result_t results[BENCHMARK_CONFIG.repeat_count];
    uint64_t sum_min = 0, sum_max = 0, sum_avg = 0;
    uint64_t global_min = UINT64_MAX, global_max = 0;
    int status = 0;
    
    if (bench->run_benchmark == NULL) {
        final_result->status = -1;
        return -1;
    }
    
    /* Run warmup */
    run_warmup(bench);
    
    /* Run multiple repetitions */
    for (uint32_t rep = 0; rep < BENCHMARK_CONFIG.repeat_count; rep++) {
        status = bench->run_benchmark(BENCHMARK_CONFIG.measurement_iterations, 
                                       &results[rep]);
        if (status != 0) {
            final_result->status = status;
            return status;
        }
        
        sum_min += results[rep].min_cycles;
        sum_max += results[rep].max_cycles;
        sum_avg += results[rep].avg_cycles;
        
        if (results[rep].min_cycles < global_min) {
            global_min = results[rep].min_cycles;
        }
        if (results[rep].max_cycles > global_max) {
            global_max = results[rep].max_cycles;
        }
    }
    
    /* Compute final statistics */
    final_result->min_cycles = global_min;
    final_result->max_cycles = global_max;
    final_result->avg_cycles = sum_avg / BENCHMARK_CONFIG.repeat_count;
    final_result->total_iterations = BENCHMARK_CONFIG.measurement_iterations * 
                                     BENCHMARK_CONFIG.repeat_count;
    final_result->status = 0;
    
    return 0;
}

/**
 * @brief Run all benchmarks and print results
 */
static void run_all_benchmarks(void) {
    benchmark_result_t result;
    int successes = 0;
    int failures = 0;
    
    print_header();
    
    /* Print JSON header marker for parsing */
    printf("=== BEGIN_RESULTS_JSON ===\n");
    
    for (size_t i = 0; i < BENCHMARK_COUNT; i++) {
        const benchmark_descriptor_t *bench = &BENCHMARKS[i];
        
        /* Small delay to prevent watchdog issues */
        vTaskDelay(pdMS_TO_TICKS(1));
        
        /* Run the benchmark */
        int status = run_benchmark_with_stats(bench, &result);
        
        /* Print JSON result */
        print_result_json(bench, &result);
        
        if (status == 0) {
            successes++;
        } else {
            failures++;
        }
    }
    
    printf("=== END_RESULTS_JSON ===\n\n");
    
    /* Print summary */
    print_separator();
    printf("Benchmark Run Complete\n");
    printf("  Successful: %d\n", successes);
    printf("  Failed:     %d\n", failures);
    printf("  Total:      %zu\n", BENCHMARK_COUNT);
    print_separator();
    
    /* Also print human-readable format */
    printf("\n=== HUMAN READABLE RESULTS ===\n\n");
    
    for (size_t i = 0; i < BENCHMARK_COUNT; i++) {
        const benchmark_descriptor_t *bench = &BENCHMARKS[i];
        
        vTaskDelay(pdMS_TO_TICKS(1));
        run_benchmark_with_stats(bench, &result);
        print_result_human(i, bench, &result);
    }
}

/*
 * =============================================================================
 * Main Entry Point
 * =============================================================================
 */

void app_main(void) {
    /* Wait for serial to be ready */
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    ESP_LOGI(TAG, "ESP32-C6 Instruction Benchmark Starting...");
    ESP_LOGI(TAG, "CPU Frequency: %d MHz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    
    /* Disable interrupts for more accurate measurements */
    /* Note: We re-enable briefly for watchdog feeding via vTaskDelay */
    
    /* Run all benchmarks */
    run_all_benchmarks();
    
    ESP_LOGI(TAG, "All benchmarks complete. Halting.");
    
    /* Infinite loop - benchmark complete */
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
