#!/usr/bin/env python3
"""
cache_benchmarks.py

Generates cache latency benchmarks for ESP32-C6.
Measures latency for:
1. SRAM (Internal RAM)
2. Flash Cache Hit (Small .rodata)
3. Flash Cache Miss (Large .rodata > Cache Size)

Methodology:
- Uses pointer chasing (Linked List) to defeat prefetching.
- Nodes are padded to 32 bytes (Likely cache line size, or at least ample padding).
- Random permutation of links to ensure unpredictable access pattern.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from .common import BenchmarkConfig
except ImportError:
    from common import BenchmarkConfig


@dataclass
class CacheBenchmarkConfig:
    """Configuration for cache benchmarks."""
    warmup_iterations: int = 10
    measurement_iterations: int = 1000
    repeat_count: int = 5
    sram_node_count: int = 256       # 256 * 32B = 8KB (Fits in SRAM)
    flash_hit_node_count: int = 256  # 256 * 32B = 8KB (Should fit in Cache)
    # ESP32-C6 Cache is likely 16KB or 32KB.
    # 32KB / 32B = 1024 lines.
    # To force misses, we need > 32KB. Let's aim for 64KB or 128KB.
    # 4096 * 32B = 128KB
    flash_miss_node_count: int = 4096 


class CacheBenchmarkGenerator:
    """Generates C code for cache latency benchmarks."""

    def __init__(self, config: CacheBenchmarkConfig):
        self.config = config
        self.generated_benchmarks: List[Tuple[str, str]] = []

    def get_definitions(self) -> str:
        """Return C definitions required for these benchmarks."""
        return """
/* 
 * Cache Benchmarking Structures 
 * Node size: 32 bytes (padded) to match typical cache line size and avoid false sharing/prefetch hits within line.
 */
typedef struct Node {
    struct Node *next;
    uint32_t padding[7]; // 4 bytes * 7 = 28 bytes + 4 bytes ptr = 32 bytes
} __attribute__((aligned(32))) node_t;
"""

    def _generate_permutation(self, size: int) -> List[int]:
        """Generate a random cycle permutation of indices 0..size-1."""
        indices = list(range(size))
        random.shuffle(indices)
        
        # We need a single cycle covering all nodes.
        # Simple shuffle might create multiple disjoint cycles.
        # Strategy: 
        # 1. Create a list of nodes 0..N-1
        # 2. Shuffle them.
        # 3. Link shuffled[i] -> shuffled[i+1], and shuffled[N-1] -> shuffled[0]
        # This guarantees one Hamiltonian cycle.
        
        perm = list(range(size))
        random.shuffle(perm)
        
        # next_indices[i] tells which index 'i' points to.
        next_indices = [0] * size
        for i in range(size - 1):
            next_indices[perm[i]] = perm[i+1]
        next_indices[perm[size-1]] = perm[0]
        
        return next_indices

    def generate_sram_test(self) -> str:
        """
        Generates SRAM latency test.
        Allocates nodes in Heap (SRAM), links them randomly at runtime, then measures traversal.
        """
        func_name = "sram_latency"
        func = f"""
/*
 * SRAM Latency Benchmark
 * Allocates linked list in SRAM (heap), initializes random pointer chase, measures latency.
 */
static int bench_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end;
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;
    
    const uint32_t node_count = {self.config.sram_node_count};
    
    // Allocate nodes in SRAM
    node_t *nodes = (node_t *)malloc(sizeof(node_t) * node_count);
    if (!nodes) return -1; // Out of memory
    
    // linear congruent generator for pseudo-random linking on device to save code space
    // Creating a full random permutation on device is complex, 
    // so we use a simple stride with a prime number coprime to node_count, or just naive random.
    // Better: use a large prime stride for simplicity in C.
    // For 256 nodes, stride 167 (prime) covers all if gcd(256, 167)=1.
    // But we want random-like behavior.
    
    // Let's implement the 'shuffled array' trick in C for initialization
    // Actually, simply linking i -> (i * PRIME + OFFSET) % N might be enough to defeat stride prefetcher?
    // Let's just do a simple prime stride that wraps around.
    
    uint32_t *indices = (uint32_t *)malloc(sizeof(uint32_t) * node_count);
    if (!indices) {{ free(nodes); return -1; }}
    
    for (uint32_t i = 0; i < node_count; i++) indices[i] = i;
    
    // Fisher-Yates shuffle
    // We use a simple LCG for randomness: x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    uint32_t rand_state = 12345;
    for (uint32_t i = node_count - 1; i > 0; i--) {{
        rand_state = (rand_state * 1103515245 + 12345) & 0x7FFFFFFF;
        uint32_t j = rand_state % (i + 1);
        uint32_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }}
    
    // Link nodes based on shuffled indices
    // perm[i] -> perm[i+1]
    for (uint32_t i = 0; i < node_count - 1; i++) {{
        nodes[indices[i]].next = &nodes[indices[i+1]];
    }}
    nodes[indices[node_count-1]].next = &nodes[indices[0]];
    
    node_t *current = &nodes[indices[0]];
    
    free(indices); // Done with indices
    
    // Warmup
    for (int i = 0; i < {self.config.warmup_iterations}; i++) {{
        for (int j = 0; j < node_count; j++) {{
             current = current->next;
        }}
    }}
    
    // Measurement
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        // Unroll slightly? 
        // No, we want pure latency. Compiler might optimize.
        // Let's rely on volatility or barrier.
        // Actually, just a simple loop.
        
        #pragma GCC unroll 0
        for (int i = 0; i < node_count; i++) {{
            current = current->next;
        }}
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        uint32_t elapsed = end - start;
        if (elapsed < min_cycles) min_cycles = elapsed;
        total_cycles += elapsed;
    }}
    
    // Calculate cycles per load (latency)
    // We did (total_iterations * node_count) loads.
    // Average per rep = total_cycles / total_iterations
    // Per load = Average per rep / node_count
    
    result->min_cycles = (uint32_t)(min_cycles / node_count);
    result->avg_cycles = (uint32_t)((total_cycles / total_iterations) / node_count);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    
    // Prevent optimization of 'current' (unlikely needed due to dependencies, but safe)
    __asm__ volatile ("" : : "r" (current));
    
    free(nodes);
    
    return 0;
}}
"""
        self.generated_benchmarks.append(("CACHE_SRAM", func_name))
        return func


    def _generate_flash_array(self, name: str, node_count: int, array_name: str) -> str:
        """Helper to generate a const array of Nodes in .rodata with pre-calculated links."""
        
        links = self._generate_permutation(node_count)
        
        # We need to output a C array initialization.
        # node_t flash_array[] = {
        #    { &flash_array[target_index], {0} },
        #    ...
        # };
        
        # Since we can't easily perform address arithmetic in a const initializer in C without knowing the base,
        # WE MUST USE INDEX-BASED linking if we were doing it dynamically, OR
        # better: strictly rely on the linker resolving `&array[i]`.
        # Yes, `&array[i]` is a valid constant expression for static initialization.
        
        lines = []
        lines.append(f"/* {name} Data: {node_count} nodes ({node_count * 32 / 1024} KB) */")
        lines.append(f"static const node_t {array_name}[{node_count}] = {{")
        
        # Generate lines. To avoid massive strings, we might want to chunk this?
        # But this function returns a string.
        # For 4096 nodes, this is 4096 lines.
        
        for i in range(node_count):
            target = links[i]
            # Point to the next node in the cycle
            lines.append(f"    {{ (node_t*)&{array_name}[{target}], {{0}} }},")
            
        lines.append("};")
        return "\n".join(lines)

    def generate_flash_hit_test(self) -> Tuple[str, str]:
        """
        Generates Flash Hit Latency test (Small Read-Only Data).
        Returns (C Function Implementation, Array Definition).
        """
        node_count = self.config.flash_hit_node_count
        array_name = "flash_hit_nodes"
        func_name = "flash_hit_latency"
        
        array_def = self._generate_flash_array("Flash Hit", node_count, array_name)
        
        func = f"""
/*
 * Flash Hit Latency Benchmark
 * Traverses a linked list in Flash (.rodata) that fits in cache ({node_count*32} bytes).
 */
static int bench_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end;
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;
    
    const node_t *current = &{array_name}[0];
    const uint32_t node_count = {node_count};
    
    // Warmup (Load into cache)
    for (int i = 0; i < {self.config.warmup_iterations}; i++) {{
        for (int j = 0; j < node_count; j++) {{
             current = current->next;
        }}
    }}
    
    // Measurement
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        // Main loop
        for (int i = 0; i < node_count; i++) {{
            current = current->next;
        }}
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        uint32_t elapsed = end - start;
        if (elapsed < min_cycles) min_cycles = elapsed;
        total_cycles += elapsed;
    }}
    
    result->min_cycles = (uint32_t)(min_cycles / node_count);
    result->avg_cycles = (uint32_t)((total_cycles / total_iterations) / node_count);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    
    __asm__ volatile ("" : : "r" (current));
    
    return 0;
}}
"""
        self.generated_benchmarks.append(("CACHE_FLASH_HIT", func_name))
        return func, array_def

    def generate_flash_miss_test(self) -> Tuple[str, str]:
        """
        Generates Flash Miss Latency test (Large Read-Only Data).
        Returns (C Function Implementation, Array Definition).
        """
        node_count = self.config.flash_miss_node_count
        array_name = "flash_miss_nodes"
        func_name = "flash_miss_latency"
        
        array_def = self._generate_flash_array("Flash Miss", node_count, array_name)
        
        func = f"""
/*
 * Flash Miss Latency Benchmark
 * Traverses a linked list in Flash (.rodata) significantly larger than cache ({node_count*32/1024} KB).
 * Random access pattern ensures cache misses.
 */
static int bench_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end;
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;
    
    const node_t *current = &{array_name}[0];
    const uint32_t node_count = {node_count};
    
    // Warmup? Maybe not needed as much if we want cold misses,
    // but consistent "miss" latency implies steady state cache thrashing.
    // Traversing it once ensures we are in a steady state of eviction.
    for (int i = 0; i < 2; i++) {{
        for (int j = 0; j < node_count; j++) {{
             current = current->next;
        }}
    }}
    
    // Measurement
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        for (int i = 0; i < node_count; i++) {{
            current = current->next;
        }}
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        uint32_t elapsed = end - start;
        if (elapsed < min_cycles) min_cycles = elapsed;
        total_cycles += elapsed;
    }}
    
    result->min_cycles = (uint32_t)(min_cycles / node_count);
    result->avg_cycles = (uint32_t)((total_cycles / total_iterations) / node_count);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    
    __asm__ volatile ("" : : "r" (current));
    
    return 0;
}}
"""
        self.generated_benchmarks.append(("CACHE_FLASH_MISS", func_name))
        return func, array_def

    def generate_unaligned_load_test(self, offset: int) -> str:
        """
        Generates Unaligned Load Latency test.
        Allocates a buffer and performs loads at specific offsets.
        """
        func_name = f"unaligned_load_{offset}"
        
        # Massive unrolling in Python to avoid C loop overhead
        asm_instr = '"lw %0, 0(%1) \\n"\n'
        asm_block = asm_instr * 100
        
        func = f"""
/*
 * Unaligned Load Benchmark (Offset {offset})
 * Loads a word from an address with specific offset.
 * Offset 31 crosses a cache line boundary (bytes 31, 32, 33, 34).
 */
static int bench_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end;
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;
    
    // Allocate buffer on stack, aligned to 64 bytes
    uint8_t buffer[128] __attribute__((aligned(64)));
    
    // Prevent compiler optimizing away the buffer
    for (int i=0; i<128; i++) buffer[i] = (uint8_t)i;
    
    uintptr_t base = (uintptr_t)&buffer[0];
    uintptr_t addr = base + {offset};
    uint32_t val;
    
    // Warmup
    for (int i=0; i<100; i++) {{
        asm volatile("lw %0, 0(%1)" : "=r"(val) : "r"(addr));
    }}
    
    // Measurement
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        // Massive unrolled block (100 loads)
        asm volatile(
            {asm_block}
            : "=r"(val) : "r"(addr)
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        uint32_t elapsed = end - start;
        if (elapsed < min_cycles) min_cycles = elapsed;
        total_cycles += elapsed;
    }}
    
    // Calculate per-load latency
    // Each iteration did 100 loads
    result->min_cycles = (uint32_t)(min_cycles / 100);
    result->avg_cycles = (uint32_t)((total_cycles / total_iterations) / 100);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    
    __asm__ volatile ("" : : "r" (val));
    
    return 0;
}}
"""
        name = f"UNALIGNED_LOAD_{offset}"
        if offset == 0:
            name += "_ALIGNED"
        elif offset == 31:
            name += "_CROSS_LINE"
            
        self.generated_benchmarks.append((name, func_name))
        return func

    def generate_fetch_linear_test(self) -> str:
        """
        Generates Linear Instruction Fetch Throughput test (1000 NOPs).
        Executes from Flash (.flash.text) with compressed instructions disabled.
        """
        func_name = "fetch_linear"
        
        func = f"""
/*
 * Linear Instruction Fetch Benchmark
 * Executes 1000 NOPs linearly from Flash.
 * Measures CPI (Cycles Per Instruction). Ideal: 1.0 CPI.
 */
static int bench_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end;
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;
    
    // Warmup
    for (int i=0; i<100; i++) {{
         asm volatile(
            ".option push \\n"
            ".option norvc \\n"
            ".rept 1000 \\n"
            "nop \\n"
            ".endr \\n"
            ".option pop \\n"
        );
    }}
    
    // Measurement
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        asm volatile(
            ".option push \\n"
            ".option norvc \\n"
            ".rept 1000 \\n"
            "nop \\n"
            ".endr \\n"
            ".option pop \\n"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        uint32_t elapsed = end - start;
        if (elapsed < min_cycles) min_cycles = elapsed;
        total_cycles += elapsed;
    }}
    
    // Calculate 1000 * CPI
    // Result stores avg_cycles per iteration. User can divide by 1000.
    
    result->min_cycles = (uint32_t)min_cycles;
    result->avg_cycles = (uint32_t)(total_cycles / total_iterations);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    
    return 0;
}}
"""
        self.generated_benchmarks.append(("FETCH_LINEAR_1000_NOP", func_name))
        return func

    def generate_fetch_branchy_test(self) -> str:
        """
        Generates Branchy Instruction Fetch Throughput test (1000 Jumps).
        Executes from Flash (.flash.text) with compressed instructions disabled.
        Uses 'j .+4' to jump to next instruction (safe for 32-bit mode).
        """
        func_name = "fetch_branchy"
        
        func = f"""
/*
 * Branchy Instruction Fetch Benchmark
 * Executes 1000 unconditional jumps (j .+4) linearly from Flash.
 * Disables prefetching efficiency. Measures CPI.
 */
static int bench_{func_name}(uint32_t total_iterations, benchmark_result_t *result) {{
    uint32_t start, end;
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX;
    
    // Warmup
    for (int i=0; i<100; i++) {{
         asm volatile(
            ".option push \\n"
            ".option norvc \\n"
            ".rept 1000 \\n"
            "j .+4 \\n"
            ".endr \\n"
            ".option pop \\n"
        );
    }}
    
    // Measurement
    COMPILER_BARRIER();
    
    for (uint32_t rep = 0; rep < total_iterations; rep++) {{
        COMPILER_BARRIER();
        start = READ_CYCLE_COUNTER();
        
        asm volatile(
            ".option push \\n"
            ".option norvc \\n"
            ".rept 1000 \\n"
            "j .+4 \\n"
            ".endr \\n"
            ".option pop \\n"
        );
        
        end = READ_CYCLE_COUNTER();
        COMPILER_BARRIER();
        
        uint32_t elapsed = end - start;
        if (elapsed < min_cycles) min_cycles = elapsed;
        total_cycles += elapsed;
    }}
    
    result->min_cycles = (uint32_t)min_cycles;
    result->avg_cycles = (uint32_t)(total_cycles / total_iterations);
    result->max_cycles = 0;
    result->total_iterations = total_iterations;
    result->status = 0;
    
    return 0;
}}
"""
        self.generated_benchmarks.append(("FETCH_BRANCHY_1000_JMP", func_name))
        return func

    def generate_descriptor_entry(self, name: str, func_name: str) -> str:
        """Generate a benchmark descriptor entry."""
        asm_syntax = "Pointer Chasing"
        if name.startswith("UNALIGNED"):
            asm_syntax = "Unaligned Load"
        elif name.startswith("FETCH"):
            asm_syntax = "Instruction Fetch"
            
        return f"""    {{
        .instruction_name = "{name}",
        .asm_syntax = "{asm_syntax}",
        .latency_type = LAT_TYPE_LOAD,
        .bench_type = BENCH_TYPE_LATENCY,
        .category = BENCH_CAT_MEMORY,
        .run_benchmark = bench_{func_name}
    }}"""

    def generate_all_benchmarks(self) -> Tuple[str, List[str]]:
        """
        Matches standard generator interface roughly.
        Returns (concatenated_c_code, list_of_descriptors).
        """
        c_code_parts = []
        descriptors = []
        
        # 1. Definitions
        c_code_parts.append(self.get_definitions())
        
        # 2. SRAM
        c_code_parts.append(self.generate_sram_test())
        
        # 3. Flash Hit
        hit_func, hit_array = self.generate_flash_hit_test()
        c_code_parts.append(hit_array)
        c_code_parts.append(hit_func)
        
        # 4. Flash Miss
        miss_func, miss_array = self.generate_flash_miss_test()
        c_code_parts.append(miss_array)
        c_code_parts.append(miss_func)
        
        # 5. Unaligned Loads (Task B)
        c_code_parts.append(self.generate_unaligned_load_test(0))   # Aligned
        c_code_parts.append(self.generate_unaligned_load_test(1))   # Simple Unaligned
        c_code_parts.append(self.generate_unaligned_load_test(31))  # Cross-Line
        
        # 6. Instruction Fetch (Task A)
        c_code_parts.append(self.generate_fetch_linear_test())
        c_code_parts.append(self.generate_fetch_branchy_test())
        
        # Generate descriptors
        for name, func_name in self.generated_benchmarks:
            descriptors.append(self.generate_descriptor_entry(name, func_name))
            
        return "\n".join(c_code_parts), descriptors
