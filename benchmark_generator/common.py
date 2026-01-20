#!/usr/bin/env python3
"""
common.py

Shared data classes and constants for benchmark generators.
"""

from dataclasses import dataclass
from pathlib import Path


# Default paths relative to project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "isa_extraction" / "output" / "esp32c6_instructions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "esp32c6_benchmark" / "main" / "generated_benchmarks.h"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark generation."""
    warmup_iterations: int = 100
    measurement_iterations: int = 1000
    repeat_count: int = 5
    chain_length: int = 100  # For latency benchmarks
    independent_count: int = 8  # For throughput benchmarks


@dataclass
class Instruction:
    """Represents an instruction to benchmark."""
    llvm_enum_name: str
    opcode_hex: str
    test_asm: str
    latency_type: str
    tablegen_entry: dict = None  # Full tablegen record from LLVM
    setup_code_override: str = None # Optional C setup code
    name_suffix: str = ""        # Suffix for benchmark name (e.g. "_LOW_LAT")


# Mapping from JSON latency_type to C enum
LATENCY_TYPE_MAP = {
    "arithmetic": "LAT_TYPE_ARITHMETIC",
    "load": "LAT_TYPE_LOAD",
    "store": "LAT_TYPE_STORE",
    "load_store": "LAT_TYPE_LOAD_STORE",
    "branch": "LAT_TYPE_BRANCH",
    "jump": "LAT_TYPE_JUMP",
    "multiply": "LAT_TYPE_MULTIPLY",
    "atomic": "LAT_TYPE_ATOMIC",
    "system": "LAT_TYPE_SYSTEM",
    "unknown": "LAT_TYPE_UNKNOWN",
}
