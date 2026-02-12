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


## Abstract Structurer & Hierarchical Concept Extraction
This step ingests BibTeX files from `conference/{SIGIR,KDD,CIKM}`, structures abstracts via LLM, extracts hierarchical concepts, and writes venue-specific outputs.

```bash
# SIGIR (2000–2024)
python pipeline/concept_extracting_pipeline/run_conference.py --sigir --output-dir output

# KDD (1995–2024)
python pipeline/concept_extracting_pipeline/run_conference.py --kdd --output-dir output

# CIKM (2001–2024)
python pipeline/concept_extracting_pipeline/run_conference.py --cikm --output-dir output
```

Outputs go to `<output-dir>`: final CSV/PKL at `{sigir_2000_2024_concepts|kdd_1995_2024_concepts|cikm_2001_2024_concepts}.{csv,pkl}` plus checkpoints `temp_{venue}_all_structured.pkl` and `temp_{venue}_all_extracted.pkl`. Intermediate `structured_abstracts.pkl` and `extracted_concepts.pkl` are also written when `processing.save_intermediate_results` is true.

Optional: add `--batch-size N` to adjust throughput/checkpoint frequency (default 50).


## Concept Clustering
This step clusters extracted concepts and maps the resulting clusters back to papers to obtain a consistent concept vocabulary.

```bash
python pipeline/concept_clustering_pipeline/preprocess_data.py \
  --input pipeline/data/input_concepts.csv \
  --output pipeline/data/input_concepts_processed.csv

python pipeline/concept_clustering_pipeline/main.py \
  --input pipeline/data/input_concepts_processed.csv \
  --venue SIGIR --min-frequency 2
```


## Dataset Construction
Use scripts in `pipeline/graph_building_pipeline/` to build **Scientific Concept Graphs** and link prediction datasets; outputs are written under `dataset/{venue}/{test,eval}/`.
Test uses a graph snapshot up to 2018 and predicts edges from 2019–2021, while Eval uses a graph snapshot up to 2021 and predicts edges from 2022–2024.

Example (from `final_papers_with_concepts.csv`):

```bash
python pipeline/graph_building_pipeline/build_concept_graph.py \
  --input output/sigir/final_papers_with_concepts.csv --venue sigir \
  --output-dir dataset/sigir/test --max-year 2018 --cache citation_cache.json
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
