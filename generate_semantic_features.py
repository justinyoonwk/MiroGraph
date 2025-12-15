import argparse

def format_semantic_context(u, v, bridges_str, ir, u_stats, v_stats):
    """
    Generate the input prompt for the Semantic Synthesizer Agent.
    Includes concept definitions, bridge concepts, and scale context (citations/papers).
    """
    lines = []
    lines.append(f"Task: Evaluate semantic synergy for Pair: ({u}, {v}) to reach {ir} citations.")
    lines.append(f"Target Threshold (IR): {ir}")
    
    # Helper to format stats
    def get_info(stats):
        if not stats:
            return "(No data available - Niche or New Concept)"
        # Avoid division by zero
        paper_count = max(1, stats.get('num', 0))
        avg_cit = stats.get('ct_Y', 0) / paper_count
        return f"Avg Citations per paper: {avg_cit:.1f}, Total Papers: {stats.get('num', 0)}"

    # 1. Concept Context (Scale & Difficulty)
    lines.append("\n[Concept Context]")
    lines.append(f"  - Concept '{u}': {get_info(u_stats)}")
    lines.append(f"  - Concept '{v}': {get_info(v_stats)}")
    lines.append("  * Note: Use these stats to judge if the concepts are Major (mainstream) or Minor (niche).")

    # 2. Structural Bridges
    if bridges_str and bridges_str.strip():
        lines.append(f"\nContext Concepts (Related terms found in graph): [{bridges_str}]")
    else:
        lines.append("\nContext Concepts: None provided (Pure semantic reasoning required).")
    
    # 3. Instructions
    lines.append("\nInstructions:")
    lines.append("1. Define the core nature of both concepts.")
    lines.append(f"2. Assessing their scale and the provided bridges, determine if a combination is semantically sound.")
    lines.append(f"3. Crucially, judge if this combination is novel and broad enough to likely achieve at least {ir} citations.")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate input for Semantic Synthesizer Agent.")
    parser.add_argument("--u", type=str, required=True, help="Source concept")
    parser.add_argument("--v", type=str, required=True, help="Target concept")
    parser.add_argument("--bridges", type=str, default="", help="Comma-separated bridge concepts")
    parser.add_argument("--ir", type=int, default=10, help="Impact Range")
    
    # Optional stats arguments (passed as string representation or simplified dict logic could be added)
    # For CLI simplicity in standalone testing, we might skip complex dict passing or use JSON.
    # But this script is mostly imported as a module. 
    # If run as main, we print a placeholder or require simplified inputs.
    
    args = parser.parse_args()
    
    # For CLI testing, we use dummy stats
    dummy_stats = {'ct_Y': 100, 'num': 10} 
    
    llm_input = format_semantic_context(args.u, args.v, args.bridges, args.ir, dummy_stats, dummy_stats)
    
    print("\n" + "="*50)
    print(" [Generated Input for Semantic Synthesizer Agent]")
    print("="*50)
    print(llm_input)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()