#!/usr/bin/env python3
"""
parse_benchmark_results.py

Parses benchmark results from ESP32-C6 serial output and generates reports.

Usage:
    # Capture output from ESP32-C6 and save to file, then:
    python parse_benchmark_results.py --input results.txt --output report.json

    # Or pipe directly from idf.py monitor:
    idf.py monitor | tee results.txt
    # Then: python parse_benchmark_results.py --input results.txt
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional
import re


@dataclass
class BenchmarkResult:
    """Parsed benchmark result."""
    instruction: str
    asm: str
    type: str
    latency_type: str
    min_cycles: int
    max_cycles: int
    avg_cycles: int
    iterations: int
    status: int


def parse_results(input_file: str) -> list:
    """Parse JSON results from benchmark output."""
    results = []
    in_json_section = False

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()

            if line == "=== BEGIN_RESULTS_JSON ===":
                in_json_section = True
                continue
            elif line == "=== END_RESULTS_JSON ===":
                in_json_section = False
                continue

            if in_json_section and line.startswith('{'):
                try:
                    data = json.loads(line)
                    result = BenchmarkResult(
                        instruction=data.get('instruction', ''),
                        asm=data.get('asm', ''),
                        type=data.get('type', ''),
                        latency_type=data.get('latency_type', ''),
                        min_cycles=data.get('min_cycles', 0),
                        max_cycles=data.get('max_cycles', 0),
                        avg_cycles=data.get('avg_cycles', 0),
                        iterations=data.get('iterations', 0),
                        status=data.get('status', -1)
                    )
                    results.append(result)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:50]}...")

    return results


def generate_summary(results: list) -> dict:
    """Generate summary statistics from results."""
    if not results:
        return {"error": "No results to summarize"}

    successful = [r for r in results if r.status == 0]
    failed = [r for r in results if r.status != 0]

    # Helper to calculate stats for a subset of results
    def calc_stats(subset):
        if not subset:
            return {}
        by_type = {}
        for r in subset:
            if r.latency_type not in by_type:
                by_type[r.latency_type] = []
            by_type[r.latency_type].append(r)
        
        type_stats = {}
        for lat_type, type_results in by_type.items():
            avg = sum(r.avg_cycles for r in type_results) / len(type_results)
            type_stats[lat_type] = {
                "count": len(type_results),
                "avg_cycles": round(avg, 2),
                "min_cycles": min(r.min_cycles for r in type_results),
                "max_cycles": max(r.max_cycles for r in type_results)
            }
        return type_stats

    latency_subset = [r for r in successful if r.type == "latency"]
    throughput_subset = [r for r in successful if r.type == "throughput"]
    
    return {
        "total_benchmarks": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "latency_stats": calc_stats(latency_subset),
        "throughput_stats": calc_stats(throughput_subset)
    }


def print_table(results: list):
    """Print results as a formatted table."""
    if not results:
        print("No results to display")
        return

    # Header
    print(f"{'Instruction':<20} {'Assembly':<30} {'Bench Type':<12} {'Lat Type':<12} {'Min':<6} {'Avg':<6} {'Max':<6} {'Status':<8}")
    print("-" * 115)

    # Sort by instruction name, then type
    sorted_results = sorted(results, key=lambda r: (r.instruction, r.type))

    for r in sorted_results:
        status_str = "OK" if r.status == 0 else f"ERR({r.status})"
        # Shorten bench type for display
        btype = "LAT" if r.type == "latency" else "THR" if r.type == "throughput" else r.type[:3].upper()
        
        print(f"{r.instruction:<20} {r.asm:<30} {btype:<12} {r.latency_type:<12} {r.min_cycles:<6} {r.avg_cycles:<6} {r.max_cycles:<6} {status_str:<8}")


def export_csv(results: list, output_file: str):
    """Export results to CSV format."""
    with open(output_file, 'w') as f:
        f.write("instruction,asm,bench_type,latency_type,min_cycles,avg_cycles,max_cycles,iterations,status\n")
        for r in results:
            asm_escaped = r.asm.replace('"', '""')
            f.write(f'{r.instruction},"{asm_escaped}",{r.type},{r.latency_type},{r.min_cycles},{r.avg_cycles},{r.max_cycles},{r.iterations},{r.status}\n')


def export_json(results: list, summary: dict, output_file: str):
    """Export results to JSON format."""
    output = {
        "summary": summary,
        "results": [asdict(r) for r in results]
    }
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Parse ESP32-C6 benchmark results"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file with captured benchmark output"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress table output"
    )

    args = parser.parse_args()

    # Parse results
    print(f"Parsing results from: {args.input}")
    results = parse_results(args.input)

    if not results:
        print("Error: No results found in input file")
        print("Make sure the file contains lines between '=== BEGIN_RESULTS_JSON ===' and '=== END_RESULTS_JSON ==='")
        sys.exit(1)

    print(f"Found {len(results)} benchmark results")

    # Generate summary
    summary = generate_summary(results)

    # Print summary
    print("\n=== Summary ===")
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    
    if summary.get('latency_stats'):
        print("\n--- Latency (cycles) ---")
        for lat_type, stats in summary['latency_stats'].items():
            print(f"  {lat_type:<12}: {stats['count']:<3} instrs, avg={stats['avg_cycles']:>6.2f}")
            
    if summary.get('throughput_stats'):
        print("\n--- Throughput (cycles/instr) ---")
        for lat_type, stats in summary['throughput_stats'].items():
            print(f"  {lat_type:<12}: {stats['count']:<3} instrs, avg={stats['avg_cycles']:>6.2f}")

    # Print table
    if not args.quiet:
        print("\n=== Results Table ===\n")
        print_table(results)

    # Export
    if args.output:
        export_json(results, summary, args.output)
        print(f"\nJSON report written to: {args.output}")

    if args.csv:
        export_csv(results, args.csv)
        print(f"CSV report written to: {args.csv}")


if __name__ == "__main__":
    main()
