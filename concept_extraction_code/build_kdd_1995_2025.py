import argparse
import glob
import json
import logging
import os
import re
from pathlib import Path

import bibtexparser
import pandas as pd

from src import AbstractStructurer, ConceptExtractor
from src.utils import Config, get_tqdm, setup_logging


TARGET_COLUMNS = [
    "year",
    "title",
    "level_0_concepts",
    "level_1_concepts",
    "level_2_concepts",
    "level_3_concepts",
    "level_4_concepts",
    "level_5_concepts",
]


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[{}]", "", text)
    return text.replace("\n", " ").strip()


def parse_kdd_2025_bibtex(bib_dir):
    papers = []
    bib_files = sorted(glob.glob(str(Path(bib_dir) / "KDD2025V*.bib")))
    for bib_file in bib_files:
        with open(bib_file, "r", encoding="utf-8") as handle:
            parser = bibtexparser.bparser.BibTexParser(common_strings=True)
            database = bibtexparser.load(handle, parser=parser)

        for entry in database.entries:
            title = clean_text(entry.get("title", ""))
            abstract = clean_text(entry.get("abstract", ""))
            if not title or not abstract:
                continue

            papers.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "year": 2025,
                    "venue": "KDD",
                    "bib_id": entry.get("ID", ""),
                    "source_bib": os.path.basename(bib_file),
                }
            )

    df = pd.DataFrame(papers)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["bib_id", "title"]).reset_index(drop=True)
    return df


def configure_processor(processor, max_workers):
    processor.config.config.setdefault("api", {})
    processor.config.config["api"]["model_name"] = "gpt-4o-mini"
    processor.config.config["api"]["max_workers"] = max_workers
    processor.config.config.setdefault("processing", {})
    processor.config.config["processing"]["save_intermediate_results"] = False
    processor.client.model_name = "gpt-4o-mini"


def valid_concept_response(response):
    if not response:
        return False
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return False
    concepts = parsed.get("hierarchical_concepts")
    return isinstance(concepts, dict) and any(concepts.get(f"level_{i}") for i in range(6))


def has_any_concept(row):
    return any(bool(row.get(f"level_{i}_concepts")) for i in range(6))


def retry_structure_batch(structurer, batch, retries, logger):
    result = structurer.structure_abstracts(batch.copy())
    for attempt in range(1, retries + 1):
        invalid = result["status"] != "valid"
        if not invalid.any():
            break
        logger.info(
            "Retrying abstract structuring for %s invalid papers (attempt %s/%s)",
            int(invalid.sum()),
            attempt,
            retries,
        )
        retry_result = structurer.structure_abstracts(batch.loc[result.index[invalid]].copy())
        result = pd.concat(
            [result.loc[~invalid], retry_result],
            axis=0,
        ).sort_index()
    return result


def retry_extract_batch(extractor, batch, retries, logger):
    result = extractor.extract_concepts(batch.copy())
    for attempt in range(1, retries + 1):
        invalid = ~result["c_list"].apply(valid_concept_response) | ~result.apply(has_any_concept, axis=1)
        if not invalid.any():
            break
        logger.info(
            "Retrying concept extraction for %s invalid papers (attempt %s/%s)",
            int(invalid.sum()),
            attempt,
            retries,
        )
        retry_result = extractor.extract_concepts(batch.loc[result.index[invalid]].copy())
        result = pd.concat(
            [result.loc[~invalid], retry_result],
            axis=0,
        ).sort_index()
    return result


def process_in_batches(df, step_name, save_path, batch_size, config, process_batch):
    logger = logging.getLogger(__name__)
    processed = pd.DataFrame()
    start_idx = 0

    if save_path.exists():
        processed = pd.read_pickle(save_path)
        start_idx = len(processed)
        logger.info("Resuming %s from %s/%s", step_name, start_idx, len(df))

    if start_idx >= len(df):
        return processed

    batch_range = range(start_idx, len(df), batch_size)
    total_batches = (len(df) - start_idx + batch_size - 1) // batch_size
    for start in get_tqdm(config, batch_range, total=total_batches, desc=f"{step_name} batches"):
        end = min(start + batch_size, len(df))
        logger.info("Processing %s batch rows %s:%s", step_name, start, end)
        batch = df.iloc[start:end].copy()
        result_batch = process_batch(batch)
        processed = pd.concat([processed, result_batch], ignore_index=True)
        processed.to_pickle(save_path)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Build KDD 1995-2025 concept CSV.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--bib-dir", default="conference/KDD")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--existing-csv", default="output/kdd_1995_2024_complete_fixed.csv")
    parser.add_argument("--output-csv", default="output/kdd_1995_2025.csv")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--validation-retries", type=int, default=2)
    args = parser.parse_args()

    config = Config(args.config)
    config.config.setdefault("api", {})
    config.config["api"]["model_name"] = "gpt-4o-mini"
    config.config["api"]["max_workers"] = args.max_workers
    setup_logging(config)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_2025 = parse_kdd_2025_bibtex(args.bib_dir)
    logger.info("Loaded %s KDD 2025 papers from %s", len(df_2025), args.bib_dir)
    if len(df_2025) == 0:
        raise RuntimeError("No KDD 2025 papers found.")

    structurer = AbstractStructurer(args.config)
    configure_processor(structurer, args.max_workers)
    structured_path = output_dir / "temp_kdd_2025_structured.pkl"
    structured_all = process_in_batches(
        df_2025,
        "Structuring",
        structured_path,
        args.batch_size,
        config,
        lambda batch: retry_structure_batch(structurer, batch, args.validation_retries, logger),
    )

    structured_valid = structurer.get_valid_papers(structured_all)
    invalid_structured = len(structured_all) - len(structured_valid)
    logger.info("Valid structured papers: %s/%s", len(structured_valid), len(structured_all))
    if invalid_structured:
        logger.warning("Dropping %s papers with invalid structured abstracts", invalid_structured)

    extractor = ConceptExtractor(args.config)
    configure_processor(extractor, args.max_workers)
    extracted_path = output_dir / "temp_kdd_2025_extracted.pkl"
    extracted = process_in_batches(
        structured_valid.reset_index(drop=True),
        "Extraction",
        extracted_path,
        args.batch_size,
        config,
        lambda batch: retry_extract_batch(extractor, batch, args.validation_retries, logger),
    )

    final_2025 = extracted[TARGET_COLUMNS].copy()
    final_2025["year"] = pd.to_numeric(final_2025["year"], errors="coerce")
    final_2025 = final_2025.sort_values(["year", "title"]).reset_index(drop=True)
    final_2025.to_pickle(output_dir / "kdd_2025_concepts.pkl")
    final_2025.to_csv(output_dir / "kdd_2025_concepts.csv", index=False)

    existing = pd.read_csv(args.existing_csv)
    combined = pd.concat([existing[TARGET_COLUMNS], final_2025], ignore_index=True)
    combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    combined = combined.sort_values(["year", "title"]).reset_index(drop=True)
    combined.to_csv(args.output_csv, index=False)

    logger.info("Saved %s rows to %s", len(combined), args.output_csv)
    logger.info("KDD 2025 rows added: %s", len(final_2025))


if __name__ == "__main__":
    main()
