import os
import argparse
import json
import random
import pandas as pd
import torch
from openai import OpenAI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

# Custom Modules
import generate_path_features as graph_module
import generate_trend_features as trend_module
import generate_semantic_features as semantic_module

# ---------------------------------------------------------
# 1. Pydantic Models (Reasoning Only)
# ---------------------------------------------------------
class GraphAnalystOutput(BaseModel):
    reasoning: str = Field(..., description="Qualitative assessment based on topology and edge weights.")

class TrendForecasterOutput(BaseModel):
    reasoning: str = Field(..., description="Qualitative forecast based on citation growth metrics.")

class SemanticSynthesizerOutput(BaseModel):
    reasoning: str = Field(..., description="Qualitative analysis of feasibility, novelty, and scale fit.")

# ---------------------------------------------------------
# 2. OpenAI & Pydantic Helpers
# ---------------------------------------------------------
def serialize_pydantic(obj):
    if hasattr(obj, "model_dump_json"): return obj.model_dump_json()
    return obj.json()

def get_gpt_structured_response(client, system_prompt, user_input, response_model, model="gpt-4o"):
    if not client:
        # Mocking
        if response_model == GraphAnalystOutput: 
            obj = response_model(reasoning="Mock Graph Analysis: Structural potential analysis...")
        elif response_model == TrendForecasterOutput: 
            obj = response_model(reasoning="Mock Trend Analysis: Growth momentum analysis...")
        else: 
            obj = response_model(reasoning="Mock Semantic Analysis: Feasibility check...")
        return serialize_pydantic(obj)
    try:
        completion = client.beta.chat.completions.parse(
            model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
            response_format=response_model,
        )
        return serialize_pydantic(completion.choices[0].message.parsed)
    except Exception as e:
        print(f"❌ GPT API Error: {e}")
        return None

def safe_json_parse(json_str):
    try:
        cleaned = json_str.strip()
        if "```json" in cleaned: cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned: cleaned = cleaned.split("```")[1].split("```")[0]
        return json.loads(cleaned)
    except: return None

def load_prompt(base_dir, filename):
    path = os.path.join(base_dir, "prompts", filename)
    with open(path, "r", encoding="utf-8") as f: return f.read()

# ---------------------------------------------------------
# 3. GDF Generator (Modified for Gephi +1 smoothing)
# ---------------------------------------------------------
def generate_gdf_string(G):
    """
    Convert NetworkX graph to Gephi GDF format string.
    Nodes: name, label, citation_count
    Edges: node1, node2, weight (citation_weight + 1)
    """
    lines = []
    
    # 1. Node Definition
    lines.append("nodedef>name VARCHAR,label VARCHAR,citation_count INT")
    
    for node in G.nodes():
        # Handle citation count (default to 0 if missing)
        cit_count = G.nodes[node].get('citation_count', 0)
        # Escape names with single quotes
        safe_name = f"'{node}'"
        lines.append(f"{safe_name},{safe_name},{cit_count}")
    
    # 2. Edge Definition
    lines.append("edgedef>node1 VARCHAR,node2 VARCHAR,weight DOUBLE")
    
    for u, v, data in G.edges(data=True):
        # 원본 데이터(0 포함) 가져오기
        raw_weight = data.get('citation_weight', 0)
        
        # [수정됨] Gephi 시각화를 위해 1을 더해서 출력 (0인 경우 엣지가 안 보이는 문제 방지)
        smooth_weight = float(raw_weight) + 1.0
        
        safe_u = f"'{u}'"
        safe_v = f"'{v}'"
        lines.append(f"{safe_u},{safe_v},{smooth_weight}")
        
    return "\n".join(lines)

# ---------------------------------------------------------
# 4. Local Model Logic
# ---------------------------------------------------------
LOCAL_MODEL = None
LOCAL_TOKENIZER = None

def load_local_model(model_name):
    global LOCAL_MODEL, LOCAL_TOKENIZER
    print(f"\n📥 Loading Local Model: {model_name}...")
    try:
        LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
        print("✅ Local Model Loaded!")
    except Exception as e:
        print(f"❌ Load Error: {e}"); exit(1)

def get_local_response(system_prompt, user_input):
    if LOCAL_MODEL is None: return json.dumps({"error": "Model not loaded"})
    system_prompt += "\n\nIMPORTANT: Output ONLY a valid JSON object."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    prompt_text = LOCAL_TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = LOCAL_TOKENIZER(prompt_text, return_tensors="pt").to(LOCAL_MODEL.device)
    with torch.no_grad():
        outputs = LOCAL_MODEL.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True, top_p=0.9)
    return LOCAL_TOKENIZER.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# ---------------------------------------------------------
# 5. Data Helpers
# ---------------------------------------------------------
def load_solutions_df(base_dir):
    path = os.path.join(base_dir, "data_pair_solution", "unconnected_2021_pair_solution_connected_2024_clean.parquet")
    if not os.path.exists(path): raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path)

def get_random_pair(df, id_to_concept):
    sample = df.sample(1).iloc[0]
    cols = df.columns
    if 'v1' in cols and 'v2' in cols: u_r, v_r = sample['v1'], sample['v2']
    elif 'source' in cols: u_r, v_r = sample['source'], sample['target']
    elif 'u' in cols: u_r, v_r = sample['u'], sample['v']
    else: u_r, v_r = sample.iloc[0], sample.iloc[1]
    
    u_name = id_to_concept.get(int(u_r), str(u_r)) if str(u_r).isdigit() else str(u_r)
    v_name = id_to_concept.get(int(v_r), str(v_r)) if str(v_r).isdigit() else str(v_r)
    return u_name, v_name

def get_ground_truth_count(df, u_name, v_name, name_to_id):
    if df is None: return None
    u_id = name_to_id.get(u_name)
    v_id = name_to_id.get(v_name)
    if u_id is None or v_id is None: return None

    v1_col = 'v1' if 'v1' in df.columns else ('source' if 'source' in df.columns else df.columns[0])
    v2_col = 'v2' if 'v2' in df.columns else ('target' if 'target' in df.columns else df.columns[1])
    sol_col = 'solution' if 'solution' in df.columns else df.columns[-1]

    row = df[((df[v1_col] == u_id) & (df[v2_col] == v_id)) | 
             ((df[v1_col] == v_id) & (df[v2_col] == u_id))]
    
    if not row.empty:
        return int(row[sol_col].values[0])
    return 0 

# ---------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=str, default=None)
    parser.add_argument("--v", type=str, default=None)
    parser.add_argument("--ir", type=int, default=10)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--openai_key", type=str, default=None)
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--local_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Init
    print("\n[Init] Loading Resources...")
    try:
        G, id_to_concept = graph_module.load_graph_data(base_dir, args.year)
        concept_to_id, df_cit = trend_module.load_trend_data(base_dir, args.year)
        name_to_id = {val: key for key, val in id_to_concept.items()}
        solutions_df = load_solutions_df(base_dir)
        print(f"Loaded {len(solutions_df)} pairs from dataset.")
    except Exception as e: print(f"❌ Init Error: {e}"); return

    # 2. Select Pair
    u, v, paths, G_sub = None, None, None, None
    if args.u and args.v:
        u, v = args.u, args.v
        paths, G_sub, msg = graph_module.bfs_path_explorer(G, u, v)
        if not paths: print(f"❌ Terminating: {msg}"); return
    else:
        print("🎲 Sampling pair...")
        for i in range(100):
            try:
                u_c, v_c = get_random_pair(solutions_df, id_to_concept)
                if u_c == v_c or u_c not in G or v_c not in G: continue
                p, s, _ = graph_module.bfs_path_explorer(G, u_c, v_c)
                if p:
                    u, v, paths, G_sub = u_c, v_c, p, s
                    break
            except: continue
        if not u: print("❌ Failed to find pair."); return

    # 3. Setup Clients
    gpt_client = OpenAI(api_key=args.openai_key) if args.openai_key else None
    if LOCAL_MODEL is None: load_local_model(args.local_model)

    print(f"\n🚀 STARTING ANALYSIS: {u} -- {v} (IR >= {args.ir})")

    # --- Phase 1: Context ---
    print("\n[Phase 1] Context Generation & Logging...")
    degs = dict(G_sub.degree())
    bridges = sorted([n for n in G_sub.nodes() if n not in (u, v)], key=lambda n: degs[n], reverse=True)[:5]
    bridges_str = ", ".join(bridges)
    
    u_stats = trend_module.get_concept_stats(df_cit, name_to_id.get(u))
    v_stats = trend_module.get_concept_stats(df_cit, name_to_id.get(v))
    u_avg = (u_stats['ct_Y']/max(1,u_stats['num'])) if u_stats else 0
    v_avg = (v_stats['ct_Y']/max(1,v_stats['num'])) if v_stats else 0
    
    input_graph = graph_module.format_for_llm(u, v, paths, G_sub, args.ir)
    input_trend = trend_module.format_trend_context(u, v, u_stats, v_stats, args.year, args.ir)
    input_semantic = semantic_module.format_semantic_context(u, v, bridges_str, args.ir, u_stats, v_stats)

    def log_io(agent_name, content, is_input=True):
        prefix = "🔍 [INPUT LOG]" if is_input else "📝 [OUTPUT LOG]"
        print(f"\n{prefix} {agent_name}:")
        print("─"*50)
        print(content.strip())
        print("─"*50)

    log_io("Graph Analyst", input_graph)
    
    # 🌟 [GDF Output] +1 Smoothing Applied Here
    print("\n👇 [Gephi GDF Output] Copy below to save as .gdf file 👇")
    print("=" * 60)
    print(generate_gdf_string(G_sub))
    print("=" * 60)
    
    log_io("Trend Forecaster", input_trend)
    log_io("Semantic Synthesizer", input_semantic)

    # --- Phase 2: Experts (Reasoning Only) ---
    print("\n[Phase 2] Experts Deliberation (Qualitative Only)...")
    def get_sys(f): return load_prompt(base_dir, f).replace("{IR}", str(args.ir))
    
    res_graph = get_gpt_structured_response(gpt_client, get_sys("graph_analyst.txt"), input_graph, GraphAnalystOutput, args.gpt_model)
    log_io("Graph Analyst", str(res_graph), is_input=False)

    res_trend = get_gpt_structured_response(gpt_client, get_sys("trend_forecaster.txt"), input_trend, TrendForecasterOutput, args.gpt_model)
    log_io("Trend Forecaster", str(res_trend), is_input=False)

    sys_sem = load_prompt(base_dir, "semantic_synthesizer.txt").replace("{IR}", str(args.ir))
    res_semantic = get_gpt_structured_response(gpt_client, sys_sem, input_semantic, SemanticSynthesizerOutput, args.gpt_model)
    log_io("Semantic Synthesizer", str(res_semantic), is_input=False)

    if not all([res_graph, res_trend, res_semantic]): print("❌ Experts failed."); return

    # --- Phase 3: Evaluator (DRAFT) ---
    print(f"\n[Phase 3] Impact Evaluator (Drafting)...")
    base_evidence = f"""
    TARGET: ({u}, {v}), IR >= {args.ir}
    GOAL: Cumulative citations of papers combining (u, v) >= {args.ir} in next 3 years.
    CONTEXT: u_avg={u_avg:.1f}, v_avg={v_avg:.1f}
    
    [EXPERT REPORTS (No Scores, Reasoning Only)]
    1. GRAPH: {res_graph}
    2. TREND: {res_trend}
    3. SEMANTIC: {res_semantic}
    """
    
    log_io("Impact Evaluator (Draft)", base_evidence)
    res_draft = get_local_response(get_sys("impact_evaluator.txt"), base_evidence)
    log_io("Impact Evaluator (Draft)", res_draft, is_input=False)

    # --- Phase 4: Skeptic ---
    print("\n[Phase 4] Skeptic Review...")
    res_skeptic = "{}"
    if gpt_client:
        try:
            sk_in = f"EVIDENCE:\n{base_evidence}\n\nDRAFT DECISION:\n{res_draft}"
            log_io("Skeptic Agent", sk_in)
            sk_comp = gpt_client.chat.completions.create(
                model=args.gpt_model,
                messages=[{"role": "system", "content": load_prompt(base_dir, "skeptic_agent.txt")}, 
                          {"role": "user", "content": sk_in}],
                response_format={"type": "json_object"}
            )
            res_skeptic = sk_comp.choices[0].message.content
            log_io("Skeptic Agent", res_skeptic, is_input=False)
        except: pass
    
    # --- Phase 5: Evaluator (FINAL) ---
    print(f"\n[Phase 5] Impact Evaluator (Final Decision)...")
    final_input = f"""
    {base_evidence}
    [YOUR DRAFT DECISION] {res_draft}
    [SKEPTIC CRITIQUE] {res_skeptic}
    [INSTRUCTION] Review Skeptic's critique. Revise if flawed, stick if robust. Output FINAL decision JSON.
    """
    log_io("Impact Evaluator (Final)", final_input)
    res_final = get_local_response(get_sys("impact_evaluator.txt"), final_input)
    log_io("Impact Evaluator (Final)", res_final, is_input=False)

    # --- Report ---
    print("\n" + "="*60 + "\n📢 FINAL REPORT\n" + "="*60)
    
    eval_json = safe_json_parse(res_final)
    prediction = eval_json.get('final_prediction', 0) if eval_json else 0
    lbl = "HIGH-IMPACT (1) 🌟" if prediction == 1 else "Low-Impact (0)"
    
    print(f"\n🔹 Prediction: {lbl}")
    if eval_json:
        print(f"   Confidence: {eval_json.get('confidence_score', 0.0):.2f}")
        print(f"   Rationale: {eval_json.get('decision_rationale')}")

    # --- GROUND TRUTH CHECK ---
    print("-" * 60)
    actual_cit = get_ground_truth_count(solutions_df, u, v, name_to_id)
    
    if actual_cit is not None:
        is_high_impact = actual_cit >= args.ir
        label_str = "HIGH-IMPACT (1)" if is_high_impact else "Low-Impact (0)"
        
        print(f"🎯 Ground Truth (2024):")
        print(f"   - Actual 3-Year Citations: {actual_cit}")
        print(f"   - Target Threshold (IR):   {args.ir}")
        print(f"   - Class Label:             {label_str}")
        
        is_correct = (prediction == 1) == is_high_impact
        result_icon = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"\n🏆 Evaluation: {result_icon}")
    else:
        print("⚠️ Ground Truth not found in dataset.")

    print("-" * 60)
    skep = safe_json_parse(res_skeptic)
    if skep:
        print(f"Skeptic Score: {skep.get('skepticism_score', 'N/A')}")

    print("="*60 + "\n")

if __name__ == "__main__":
    main()