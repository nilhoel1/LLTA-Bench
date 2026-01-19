"""
benchmark_generator package

Provides latency and throughput benchmark generation for ESP32-C6.
"""

from .common import BenchmarkConfig, Instruction, LATENCY_TYPE_MAP
from .latency_generator import LatencyBenchmarkGenerator
from .throughput_generator import ThroughputBenchmarkGenerator

__all__ = [
    'BenchmarkConfig',
    'Instruction',
    'LATENCY_TYPE_MAP',
    'LatencyBenchmarkGenerator',
    'ThroughputBenchmarkGenerator',
]
