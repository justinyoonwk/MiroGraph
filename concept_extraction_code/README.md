# Concept Extraction Code

This folder extracts hierarchical scientific concepts from venue BibTeX files.

`run_conference.py` reads BibTeX files from `conference/{SIGIR,KDD,CIKM}`, structures abstracts with an LLM, extracts concepts, and writes venue-specific outputs. Run commands from this folder.

```bash
# SIGIR
python run_conference.py --sigir --output-dir output

# KDD
python run_conference.py --kdd --output-dir output --batch-size 100

# CIKM
python run_conference.py --cikm --output-dir output
```

Outputs:

- SIGIR: `output/sigir_2000_2025_concepts.{csv,pkl}`
- KDD: `output/kdd_1995_2025_concepts.{csv,pkl}`
- CIKM: `output/cikm_1993_2025_concepts.{csv,pkl}`
