import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export final cluster name lists and summary statistics")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    names = pickle.load(open(output_dir / "final_cluster_names.pkl", "rb"))
    concept_to_clusters = pickle.load(open(output_dir / "concept_to_clusters.pkl", "rb"))
    papers = pd.read_csv(output_dir / "final_papers_with_concepts.csv")

    cluster_to_keywords = defaultdict(list)
    for concept, cluster_ids in concept_to_clusters.items():
        for cluster_id in cluster_ids:
            cluster_to_keywords[int(cluster_id)].append(concept)

    paper_counts = Counter()
    for value in papers["assigned_concepts"].fillna(""):
        concepts = {c.strip() for c in str(value).split(",") if c.strip()}
        for concept in concepts:
            paper_counts[concept] += 1

    rows = []
    for cluster_id, name in sorted(names.items()):
        keywords = sorted(cluster_to_keywords.get(int(cluster_id), []))
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_name": name,
                "keyword_count": len(keywords),
                "paper_count": paper_counts.get(name, 0),
                "keywords_preview": "; ".join(keywords[:20]),
            }
        )

    all_names = pd.DataFrame(rows)
    all_names.to_csv(output_dir / "all_cluster_names.csv", index=False, encoding="utf-8")

    unique_rows = []
    for name, group in all_names.groupby("cluster_name", dropna=False):
        unique_rows.append(
            {
                "cluster_name": name,
                "cluster_id_count": len(group),
                "paper_count": int(paper_counts.get(name, 0)),
                "cluster_ids": ",".join(str(v) for v in sorted(group["cluster_id"].tolist())),
            }
        )

    unique_stats = pd.DataFrame(unique_rows).sort_values(
        ["paper_count", "cluster_id_count", "cluster_name"],
        ascending=[False, False, True],
    )
    unique_stats.to_csv(output_dir / "unique_cluster_name_stats.csv", index=False, encoding="utf-8")

    duplicate_groups = unique_stats[unique_stats["cluster_id_count"] > 1]
    print("final_cluster_ids", len(names))
    print("unique_cluster_names", all_names["cluster_name"].nunique())
    print("duplicate_name_groups", len(duplicate_groups))
    print("cluster_ids_with_duplicate_name", int(duplicate_groups["cluster_id_count"].sum()))
    print("mapped_source_keywords", len(concept_to_clusters))
    print("paper_rows", len(papers))
    print("papers_with_any_concept", int((papers["assigned_concepts"].fillna("").str.strip() != "").sum()))
    counts = all_names["keyword_count"].value_counts().sort_index()
    print("keyword_count_per_cluster_hist")
    for keyword_count, cluster_count in counts.items():
        print(f"{int(keyword_count)} {int(cluster_count)}")
    print("top_30_by_paper_count")
    print(unique_stats.head(30).to_string(index=False))
    print("top_30_duplicate_names")
    print(duplicate_groups.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
