#!/usr/bin/env python3
"""
Pipeline: BERTopic-based topic modeling + semantic labeling + simple analytics.

Usage:
  python src/pipeline.py \
    --citations_csv data/sample_citations.csv \
    --editorial_csv data/sample_editorial.csv \
    --out_dir outputs
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

DEFAULT_LABELS = [
    "virtual reality", "computer graphics", "geometric modelling", "shape analysis",
    "geometry processing", "augmented reality", "visual analytics", "computer vision",
    "3d user interfaces", "real time rendering", "medical visualization", "visualization",
    "computational geography", "computational topology", "topological data analysis"
]

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_citations(path):
    df = pd.read_csv(path)
    for col in ["Title", "Abstract"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    if "Year" not in df.columns:
        df["Year"] = np.nan
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    return df

def load_editorial(path):
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Expect optional "Expertise" column with comma-separated labels
    return df

def fit_bertopic(abstracts, stop_lang="english"):
    vectorizer_model = CountVectorizer(stop_words=stop_lang)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, calculate_probabilities=False)
    topics, _ = topic_model.fit_transform(abstracts)
    return topic_model, topics

def label_topics(topic_model, labels):
    info = topic_model.get_topic_info()
    valid = info[info["Topic"] != -1]["Topic"].tolist()
    topic_keywords = {
        t: " ".join([w for w, _ in topic_model.get_topic(t)])
        for t in valid
    }
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    topic_embeds = embedder.encode(list(topic_keywords.values()), convert_to_numpy=True)
    label_embeds = embedder.encode(labels, convert_to_numpy=True)
    sim = cosine_similarity(topic_embeds, label_embeds)
    best = [labels[idx] for idx in np.argmax(sim, axis=1)]
    return dict(zip(valid, best)), sim

def plot_topic_trends(df, out_path):
    # Expect 'Year' and 'topic_label'
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    filtered = df[df["topic_label"].notna() & (df["topic_label"] != "outlier")]
    if filtered.empty or filtered["Year"].dropna().empty:
        print("No valid Year/topic_label data to plot trends.")
        return
    pivot = (
        filtered.groupby(["Year", "topic_label"]).size()
        .reset_index(name="Count")
        .pivot(index="Year", columns="topic_label", values="Count")
        .fillna(0).astype(int)
    )
    plt.figure(figsize=(16, 8))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.title("Topic Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_papers_vs_editors_heatmap(df, editorial_df, out_path):
    # Build paper counts by topic_label
    topic_labels = df["topic_label"].dropna().unique()
    paper_counts = df["topic_label"].value_counts().reindex(topic_labels).fillna(0).astype(int)

    # Basic editorial counts per label from "Expertise" column (comma-separated)
    if editorial_df is not None and "Expertise" in editorial_df.columns:
        editor_label_counts = {lbl: 0 for lbl in topic_labels}
        for _, row in editorial_df.iterrows():
            labels = str(row.get("Expertise", "")).lower().split(",")
            labels = [l.strip() for l in labels if l.strip()]
            for lbl in labels:
                if lbl in editor_label_counts:
                    editor_label_counts[lbl] += 1
        editors = np.array([editor_label_counts.get(lbl, 0) for lbl in topic_labels])
    else:
        # Fallback: simulate counts (0–4) if editorial_df not provided
        rng = np.random.default_rng(42)
        editors = rng.integers(0, 5, size=len(topic_labels))

    # Build a matrix (2 x N)
    papers = paper_counts.values
    matrix = np.vstack([papers, editors])

    plt.figure(figsize=(16, 4))
    # Manual heatmap using imshow to avoid seaborn dependency
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks([0, 1], ["papers", "editors"])
    plt.xticks(range(len(topic_labels)), topic_labels, rotation=45, ha="right")
    plt.title("Papers vs Editorial Expertise")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--citations_csv", required=True, help="Path to citations CSV with Title, Abstract, Year")
    ap.add_argument("--editorial_csv", default=None, help="Optional path to editorial board CSV with 'Expertise' column")
    ap.add_argument("--out_dir", default="outputs", help="Directory to write outputs")
    ap.add_argument("--labels_json", default=None, help="Optional JSON file with an array of label strings")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = load_citations(args.citations_csv)
    editorial_df = load_editorial(args.editorial_csv)

    labels = DEFAULT_LABELS
    if args.labels_json and os.path.exists(args.labels_json):
        try:
            import json
            with open(args.labels_json, "r") as f:
                labels = json.load(f)
            if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
                raise ValueError("labels_json must contain a JSON array of strings")
        except Exception as e:
            print(f"Warning: failed to load labels_json ({e}); falling back to defaults.", file=sys.stderr)

    topic_model, topics = fit_bertopic(df["Abstract"].tolist())
    topic_to_label, _ = label_topics(topic_model, labels)

    df["topic"] = topics
    df["topic_label"] = df["topic"].map(topic_to_label).fillna("outlier")

    # Save per-paper topics
    per_paper_path = os.path.join(args.out_dir, "abstracts_with_topics.csv")
    df.to_csv(per_paper_path, index=False)

    # Compute and save top-3 topics per year
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    filtered = df[df["topic_label"].notna() & (df["topic_label"] != "outlier")]
    if not filtered.empty and not filtered["Year"].dropna().empty:
        counts = (
            filtered.groupby(["Year", "topic_label"]).size()
            .reset_index(name="Count")
            .sort_values(["Year", "Count"], ascending=[True, False])
        )
        top3 = counts.groupby("Year").head(3).reset_index(drop=True)
        top3_path = os.path.join(args.out_dir, "top_topics_by_year.csv")
        top3.to_csv(top3_path, index=False)

    # Plots
    plot_topic_trends(df, os.path.join(args.out_dir, "topic_trends.png"))
    plot_papers_vs_editors_heatmap(df, editorial_df, os.path.join(args.out_dir, "papers_vs_editors_heatmap.png"))

    print(f"✅ Done. Outputs written to: {args.out_dir}")
    print(f"- {per_paper_path}")
    if os.path.exists(os.path.join(args.out_dir, 'top_topics_by_year.csv')):
        print(f"- {os.path.join(args.out_dir, 'top_topics_by_year.csv')}")
    print(f"- {os.path.join(args.out_dir, 'topic_trends.png')}")
    print(f"- {os.path.join(args.out_dir, 'papers_vs_editors_heatmap.png')}")

if __name__ == "__main__":
    main()
