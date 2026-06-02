# MiroGraph

Code and datasets for the MiroGraph CIKM 2026 submission.

## Requirements

- Python 3.11
- `pip install -r requirements.txt`
- Set `OPENAI_API_KEY` in the environment.

## Repository Structure

- `concept_extraction_code/`: extracts hierarchical scientific concepts from venue BibTeX files.
- `scg_dataset_pipeline_code/`: clusters extracted concepts, builds temporal Scientific Concept Graphs, and creates transductive/inductive prediction datasets.
- `mirograph_model_code/`: runs MiroGraph multi-agent scoring, classifier design, baselines, and ablations.
- `dataset/`: public CIKM, SIGIR, and KDD graph/dataset files.

## Model Defaults

Defaults are defined in `mirograph_model_code/mirograph_config.py`.

- Agent scoring / direct judging: `gpt-4o`
- Feature schema design and structural feature code generation: `gpt-5.4`
- Classifier code generation: `gpt-5.4`

These defaults can be overridden with CLI flags or environment variables:

- `MIROGRAPH_SCORING_MODEL`
- `MIROGRAPH_CODE_MODEL`

## Feature Design

Run this once per dataset/setting before multi-agent scoring.

```bash
python mirograph_model_code/feature_designer.py \
  --dataset SIGIR \
  --setting test \
  --ir 5 \
  --output_schema results/feature_designs/sigir_test_ir5_schema.json \
  --output_structure_code results/feature_designs/sigir_test_ir5_structure.py
```

## Test / Transductive

Test uses the `transductive` split. Both train and test scoring contexts use the 2023 graph.

Expected files:

- `dataset/<DATASET>/transductive/concept_graph_2023.csv`
- `dataset/<DATASET>/transductive/transductive_train_ir<IR>.csv`
- `dataset/<DATASET>/transductive/transductive_test_ir<IR>.csv`

```bash
python mirograph_model_code/multiagent_scoring.py \
  --experiment test \
  --dataset SIGIR \
  --ir 5 \
  --feature_schema results/feature_designs/sigir_test_ir5_schema.json \
  --structure_code results/feature_designs/sigir_test_ir5_structure.py \
  --output_dir results/sigir_test_ir5

python mirograph_model_code/classifier_designer.py \
  --train_scores_cache results/sigir_test_ir5/train_scores.jsonl \
  --test_scores_cache results/sigir_test_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/sigir_test_ir5_schema.json \
  --output results/sigir_test_ir5/meta_code_test_predictions.jsonl \
  --best_code_out results/sigir_test_ir5/best_code.py
```

## Eval / Inductive

Eval trains on the `inductive` train file and evaluates on the same held-out transductive test file used by Test.
Train scoring uses the 2021 graph. Eval target scoring uses the 2023 transductive graph.

Expected files:

- `dataset/<DATASET>/inductive/concept_graph_2021.csv`
- `dataset/<DATASET>/inductive/inductive_train_ir<IR>.csv`
- `dataset/<DATASET>/transductive/concept_graph_2023.csv`
- `dataset/<DATASET>/transductive/transductive_test_ir<IR>.csv`

```bash
python mirograph_model_code/multiagent_scoring.py \
  --experiment eval \
  --dataset SIGIR \
  --ir 5 \
  --feature_schema results/feature_designs/sigir_test_ir5_schema.json \
  --structure_code results/feature_designs/sigir_test_ir5_structure.py \
  --output_dir results/sigir_eval_ir5

python mirograph_model_code/classifier_designer.py \
  --train_scores_cache results/sigir_eval_ir5/train_scores.jsonl \
  --test_scores_cache results/sigir_eval_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/sigir_test_ir5_schema.json \
  --output results/sigir_eval_ir5/meta_code_eval_predictions.jsonl \
  --best_code_out results/sigir_eval_ir5/best_code.py
```

Metrics are accuracy and macro-F1. The classifier designer splits the training score cache using a 70:15 train/validation weight ratio, then retrains the selected classifier on the full training cache before predicting test/eval samples.

## Ablations

Supported ablations are `no_structure`, `no_semantic`, `no_trend`, `single_score`, and the MLP classifier ablation.

```bash
python mirograph_model_code/ablation_study.py \
  --modes no_structure,no_semantic,no_trend \
  --train_scores_cache results/sigir_test_ir5/train_scores.jsonl \
  --test_scores_cache results/sigir_test_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/sigir_test_ir5_schema.json \
  --output results/sigir_test_ir5/ablation.jsonl
```

```bash
python mirograph_model_code/single_score_scoring.py \
  --experiment test \
  --dataset SIGIR \
  --ir 5 \
  --output_dir results/sigir_test_ir5_single_score

python mirograph_model_code/ablation_study.py \
  --mode single_score \
  --train_scores_cache results/sigir_test_ir5_single_score/train_scores.jsonl \
  --test_scores_cache results/sigir_test_ir5_single_score/test_scores.jsonl \
  --output results/sigir_test_ir5_single_score/ablation_single_score.jsonl
```

```bash
python mirograph_model_code/ablation_study.py \
  --classifier_backend mlp \
  --train_scores_cache results/sigir_test_ir5/train_scores.jsonl \
  --test_scores_cache results/sigir_test_ir5/test_scores.jsonl \
  --feature_schema results/feature_designs/sigir_test_ir5_schema.json \
  --output results/sigir_test_ir5/ablation_mlp.jsonl
```

## Single-LLM Baseline

```bash
python mirograph_model_code/simple_prediction.py \
  --test_file dataset/SIGIR/transductive/transductive_test_ir5.csv \
  --ir 5 \
  --model gpt-4o-mini \
  --output results/sigir_test_ir5/simple_gpt4o_mini.jsonl
```
