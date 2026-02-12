# Bridging Concepts: Multi-Agent Reasoning over Scientific Concept Graphs for High-Impact Research Prediction

This is an official implementation of MiroGraph submitted to SIGIR 2026, Short Paper Track. DO NOT DISTRIBUTE.


## Requirements
- Python 3.11
- `pip install -r requirements.txt` (standard data/ML/LLM deps)
- Set your OpenAI API key via `OPENAI_API_KEY` (recommended: create a `.env` file in the repo root):

```bash
cat > .env <<'EOF'
OPENAI_API_KEY=your-api-key
EOF
```


## Reproducing Results (Example: SIGIR Test Split)
Run `multiagent_scoring.py` on `dataset/sigir/test/*` to create score caches, then run `classifier_designer.py` to generate predictions and the best classifier code under `results/`.

```bash
python multiagent_scoring.py \
  --train_csv dataset/sigir/test/train_data_10.csv \
  --test_csv  dataset/sigir/test/test_data_10.csv \
  --concept_graph dataset/sigir/test/concept_graph.csv \
  --ir 10 --model gpt-5-mini --max_workers 100 \
  --scores_cache_train results/metallm_sigir_test/train_scores.jsonl \
  --scores_cache_test  results/metallm_sigir_test/test_scores.jsonl \
  --keep_test_labels

python classifier_designer.py \
  --train_scores_cache results/metallm_sigir_test/train_scores.jsonl \
  --test_scores_cache  results/metallm_sigir_test/test_scores.jsonl \
  --output results/metallm_sigir_test/meta_code_test_predictions.jsonl \
  --best_code_out results/metallm_sigir_test/best_code.py \
  --num_candidates 10 --seed 42
```


## Eval Split
Eval uses `dataset/sigir/eval/*` and typically reuses the train cache from the Test split; outputs go to `results/metallm_sigir_eval/`.

```bash
python multiagent_scoring.py \
  --train_csv dataset/sigir/test/train_data_10.csv \
  --test_csv  dataset/sigir/eval/test_data_10.csv \
  --concept_graph dataset/sigir/eval/concept_graph_test.csv \
  --ir 10 --model gpt-5-mini --max_workers 100 \
  --scores_cache_train results/metallm_sigir_test/train_scores.jsonl \
  --reuse_train_cache \
  --scores_cache_test  results/metallm_sigir_eval/test_scores.jsonl \
  --keep_test_labels

python classifier_designer.py \
  --train_scores_cache results/metallm_sigir_test/train_scores.jsonl \
  --test_scores_cache  results/metallm_sigir_eval/test_scores.jsonl \
  --output results/metallm_sigir_eval/meta_code_test_predictions.jsonl \
  --best_code_in results/metallm_sigir_test/best_code.py \
  --seed 42
```


## Ablation (no_structure / no_semantic / no_trend / no_designer)
Use `ablation_study.py` with modes `{no_structure,no_semantic,no_trend,no_designer}`; outputs are written under `results/`.

```bash
python ablation_study.py \
  --mode no_semantic \
  --train_scores_cache results/metallm_sigir_test/train_scores.jsonl \
  --test_scores_cache  results/metallm_sigir_test/test_scores.jsonl \
  --output results/metallm_sigir_test/ablation_no_semantic.jsonl \
  --num_candidates 10 --seed 42
```


## No-Score Ablation (Agents output Yes/No only)
Use `ablation_study_no_score.py` to run the no-score variant; outputs are written under `results/`.

```bash
python ablation_study_no_score.py \
  --train_csv dataset/sigir/test/train_data_10.csv \
  --test_csv  dataset/sigir/test/test_data_10.csv \
  --concept_graph dataset/sigir/test/concept_graph.csv \
  --ir 10 --model gpt-5-mini --max_workers 100 \
  --num_candidates 10 --seed 42 \
  --output results/metallm_sigir_test/ablation_no_score_fresh.jsonl
```
