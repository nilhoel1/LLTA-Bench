#!/usr/bin/env python3
import json
import os
import sys

# Define the support logic from the generators

def check_latency_support(instr):
    asm = instr.get("test_asm", "").lower().strip()
    lat_type = instr.get("latency_type", "")
    
    # 1. Skip patterns
    skip_patterns = [
        "ecall", "ebreak", "mret", "sret", "dret", "wfi",
        "fence", "sfence", "unimp", "c.unimp", ".insn", "c.nop", "nop",
    ]
    for p in skip_patterns:
        if p in asm:
            return False, "Pattern skip"
            
    # 2. Dangerous patterns
    dangerous_patterns = [
        "addi16sp", "addi4spn", ", sp,", " sp,", ",sp)", "(sp)",
        " zero,", ",zero,", "slli64", "srai64", "srli64",
    ]
    for p in dangerous_patterns:
        if p in asm:
            return False, "Dangerous pattern"

    # 3. Type specific checks
    parts = asm.split()
    mnemonic = parts[0] if parts else ""

    if lat_type in ["arithmetic", "multiply"]:
        return True, "Arithmetic/Multiply"
        
    elif lat_type == "load":
        if "lb" in asm or "lh" in asm:
            return False, "Load sub-word"
        return True, "Load word"
        
    elif lat_type == "store":
        store_pairs = ["sw", "sh", "sb", "c.sw"] # c.sh/c.sb not in map
        if mnemonic in store_pairs:
            return True, "Store supported"
        return False, "Store unsupported"
        
    elif lat_type == "load_store":
        # Simplified check based on generator
        if "lw" in asm and "sw" not in asm:
            if "lb" in asm or "lh" in asm: return False, "Load sub-word"
            return True, "Load portion"
        elif "sw" in asm:
            if mnemonic == "c.sw": return True, "Store portion"
        elif "sh" in asm or "sb" in asm:
            # logic says _gen_store_chain, which checks store_load_pairs.
            # store_load_pairs has sw, sh, sb, c.sw
            if mnemonic in ["sh", "sb"]: return True, "Store portion"
        return False, "LoadStore complex"

    elif lat_type == "branch":
        supported = ["beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"]
        if mnemonic in supported:
            return True, "Branch"
        return False, "Branch unsupported"
        
    elif lat_type == "jump":
        if mnemonic in ["jal", "c.j", "c.jal"]:
            return True, "Jump"
        return False, "Jump unsupported"
        
    elif lat_type == "atomic":
        if "lr." in asm or "sc." in asm:
            return False, "LR/SC"
        if mnemonic.startswith("amo"):
            return True, "Atomic"
        return False, "Atomic other"
        
    return False, "Unknown/System"

def check_throughput_support(instr):
    asm = instr.get("test_asm", "").lower().strip()
    parts = asm.split()
    mnemonic = parts[0] if parts else ""
    
    # distinct import path handling to make sure we can import the module from tools/ or root
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from benchmark_generator.throughput_generator import SUPPORTED_MNEMONICS
    
    if mnemonic in SUPPORTED_MNEMONICS:
        return True, "Supported"
    return False, "Not in allowlist"

def main():
    json_path = "isa_extraction/output/esp32c6_instructions.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found")
        sys.exit(1)
        
    with open(json_path, 'r') as f:
        instructions = json.load(f)
        

    missing_throughput = []
    missing_latency = [] # Supported by throughput but not latency
    missing_both = [] # Supported by neither
    covered_both = [] 
    
    print(f"{'Instruction':<20} | {'Latency Status':<20} | {'Throughput Status':<20}")
    print("-" * 70)
    
    for instr in instructions:
        name = instr.get("llvm_enum_name")
        lat_ok, lat_reason = check_latency_support(instr)
        tp_ok, tp_reason = check_throughput_support(instr)
        
        if lat_ok and tp_ok:
            covered_both.append(instr)
        elif lat_ok and not tp_ok:
            missing_throughput.append(instr)
            print(f"{name:<20} | {lat_reason:<20} | {tp_reason:<20}")
        elif not lat_ok and tp_ok:
            missing_latency.append(instr)
            print(f"{name:<20} | {lat_reason:<20} | {tp_reason:<20}")
        else:
            missing_both.append(instr)
            # Optional: Uncomment to see completely unsupported instructions
            # print(f"{name:<20} | {lat_reason:<20} | {tp_reason:<20}")
            
    print("-" * 70)
    print(f"Total instructions in ISA: {len(instructions)}")
    print(f"Covered by Both:           {len(covered_both)}")
    print(f"Latency Only (TP Gap):     {len(missing_throughput)}")
    print(f"Throughput Only (Lat Gap): {len(missing_latency)}")
    print(f"Covered by Neither:        {len(missing_both)}")
    
    # Dump missing categories to files for inspection
    with open("gap_throughput.json", "w") as f:
        json.dump(missing_throughput, f, indent=2)
    with open("gap_latency.json", "w") as f:
        json.dump(missing_latency, f, indent=2)
    with open("gap_neither.json", "w") as f:
        json.dump(missing_both, f, indent=2)

if __name__ == "__main__":
    main()
