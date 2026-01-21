
import json
import collections

def generate_tables():
    with open('report.json', 'r') as f:
        data = json.load(f)

    results = data['results']
    
    # Analyze and Merge Data
    # Map instruction -> {latency: ..., throughput: ..., category: ..., asm: ...}
    instr_map = collections.defaultdict(dict)
    
    for r in results:
        instr = r['instruction']
        bench_type = r.get('bench_type', r.get('type')) # Handle both keys if present, JSON uses 'type' for "latency" but CSV has field 'bench_type' which is clearer. In JSON, 'type' is 'latency' or 'throughput'.
        
        # Check JSON structure from view_file
        # JSON: "type": "latency" or "throughput"
        # JSON: "latency_type": "arithmetic", etc.
        
        b_type = r['type'] # 'latency' or 'throughput'
        cat = r['latency_type']
        
        if 'category' not in instr_map[instr]:
             instr_map[instr]['category'] = cat
        if 'asm' not in instr_map[instr]:
             instr_map[instr]['asm'] = r['asm']
             
        # Extract metrics
        # For latency, we want min_cycles (or avg if it varies? README uses min mostly, but average for some?)
        # README methodology says "Total cycles / chain length". This equates to avg_cycles in the JSON.
        # But wait, looking at json: "avg_cycles": 1.0. "min_cycles": 1.
        # For variable latency instructions (like branches), avg might be more informative?
        # The current README says "3-4" for branches.
        # JSON for BGE: min 1, max 12, avg 4.
        
        val = r['min_cycles']
        
        # Special handling based on existing README style
        if cat == 'branch' or cat == 'jump':
             # For branches, might define range?
             # Let's start with min_cycles, but maybe check avg.
             pass
        
        instr_map[instr][b_type] = val

    # --- Generate Detailed Results Table ---
    # Header: | Instruction | Assembly | Type | Latency | Throughput |
    detailed_rows = []
    sorted_instrs = sorted(instr_map.keys())
    
    for instr in sorted_instrs:
        info = instr_map[instr]
        asm = f"`{info.get('asm', '')}`"
        cat = info.get('category', 'unknown')
        lat = info.get('latency', '-')
        thr = info.get('throughput', '-')
        
        detailed_rows.append(f"| {instr} | {asm} | {cat} | {lat} | {thr} |")
        
    detailed_table = "| Instruction | Assembly | Type | Latency | Throughput |\n|-------------|----------|------|---------|------------|\n" + "\n".join(detailed_rows)
    
    # --- Generate Summary Table ---
    # We need to manually aggregate consistent with the README categories.
    # Group by category logic from current README
    
    # Categories:
    # Arithmetic: ADD...
    # Shifts
    # Compressed ALU
    # Upper Immediate
    # Multiply (low)
    # Multiply (high)
    # Division
    # Sign/Zero Extend
    # Word Load
    # Atomic
    # Branch (not-taken) ?? Wait, data says Latency 1 for BEQ.
    # Jump
    
    # Let's create a Helper to gather stats for a list of instrs
    def get_stats(instr_list):
        lats = []
        thrus = []
        for i in instr_list:
            if i in instr_map:
                l = instr_map[i].get('latency', None)
                t = instr_map[i].get('throughput', None)
                if l is not None: lats.append(l)
                if t is not None: thrus.append(t)
        
        def fmt_range(lst):
            if not lst: return "-"
            mn, mx = min(lst), max(lst)
            if mn == mx: return str(mn)
            return f"{mn}-{mx}"
            
        return fmt_range(lats), fmt_range(thrus)

    # Define groups
    groups = [
        ("Arithmetic", ["ADD", "ADDI", "SUB", "AND", "ANDI", "OR", "ORI", "XOR", "XORI", "SLT", "SLTI", "SLTIU", "SLTU"]),
        ("Shifts", ["SLL", "SLLI", "SRA", "SRAI", "SRL", "SRLI"]),
        ("Compressed ALU", ["C_ADD", "C_ADDI", "C_AND", "C_ANDI", "C_OR", "C_XOR", "C_SUB", "C_MV", "C_LI", "C_SLLI", "C_SRAI", "C_SRLI"]),
        ("Upper Immediate", ["LUI", "AUIPC"]),
        ("Multiply (low)", ["MUL"]),
        ("Multiply (high)", ["MULH", "MULHSU", "MULHU"]),
        ("Division", ["DIV", "DIVU", "REM", "REMU"]),
        ("Sign/Zero Extend", ["SEXT_B", "SEXT_H", "ZEXT_H_RV32", "ZEXT_H_RV64"]), # Note: JSON has ZEXT_H_RV32/64
        ("Word Load", ["LW", "C_LW"]), # Note: JSON has LW, C_LW
        ("Atomic (AMO)", [k for k in sorted_instrs if k.startswith("AMO") and k.endswith("_W")]), # Simplified match
        ("Branch (not-taken)", ["BEQ", "BNE"]), 
        ("Branch (taken/complex)", ["BGE", "BGEU", "BLT", "BLTU", "C_BEQZ", "C_BNEZ"]), # Rename based on values?
        ("Jump (direct)", ["C_J", "C_JAL", "JAL"]),
    ]

    summary_rows = []
    
    # Handle specific names adjustments (e.g. ZEXT.H in readme vs ZEXT_H_RV32 in data)
    # The README lists instructions with dots (SEXT.H), data has underscores (SEXT_H).
    # I should map them for display or just use the names in the data? 
    # The current README summary table uses nice names "SEXT.B", "C.ADD".
    # The detailed table uses "SEXT_B".
    # I should try to format the list of instructions in the summary table nicely.
    
    for cat_name, items in groups:
        # Check for actual keys in map (handle underscores)
        real_items = []
        display_items = []
        
        for item in items:
            # Try exact match
            if item in instr_map:
                real_items.append(item)
                display_items.append(item.replace("_", "."))
                continue
            
            # Try underscore version
            u_item = item.replace(".", "_")
            if u_item in instr_map:
                real_items.append(u_item)
                display_items.append(item) # Keep dot for display
                continue
                
            # Try RV32/64 suffix for ZEXT
            if item == "ZEXT.H" or item == "ZEXT_H":
                if "ZEXT_H_RV32" in instr_map: real_items.append("ZEXT_H_RV32"); display_items.append("ZEXT.H")
                # ignore RV64 for summary brevity if same?
                continue
                
            # Special case for AMO wildcard
            # already filtered the list for AMO generally, but let's just use the gathered list
        
        if cat_name == "Atomic (AMO)":
             real_items = [k for k in instr_map if k.startswith("AMO") and "_W" in k and "AQ" not in k and "RL" not in k] # Base AMOs
             # Actually, let's just use a representative sample or aggregate all.
             all_amos = [k for k in instr_map if k.startswith("AMO")]
             l_str, t_str = get_stats(all_amos)
             # Update display
             display_str = "AMOADD.W, AMOSWAP.W, etc."
             summary_rows.append(f"| **{cat_name}** | {display_str} | {l_str} | {t_str} |")
             continue

        if not real_items:
            continue
            
        l_str, t_str = get_stats(real_items)
        display_str = ", ".join(display_items)
        
        # Override for Division Throughput asterisk
        if cat_name == "Division":
            t_str += "*"
            
        summary_rows.append(f"| **{cat_name}** | {display_str} | {l_str} | {t_str} |")

    summary_table = "| Category | Instructions | Latency | Throughput |\n|----------|-------------|---------|------------|\n" + "\n".join(summary_rows)


    # --- Write to RESULTS.md ---
    with open('RESULTS.md', 'w') as f:
        f.write("# LLTA-Bench Results\n\n")
        f.write("**Target**: ESP32-C6 @ 160MHz\n")
        f.write("**Generated**: Auto-generated from `report.json`\n\n")
        
        f.write("## Results Summary\n\n")
        f.write(summary_table + "\n\n")
        
        f.write("*Division is not pipelined, so throughput ≈ latency.\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write(detailed_table + "\n")

    print(f"Successfully generated RESULTS.md with {len(results)} benchmarks.")

if __name__ == "__main__":
    generate_tables()
