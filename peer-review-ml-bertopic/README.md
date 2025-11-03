# Optimizing Peer Review with BERTopic & Embeddings

This repository contains a **sanitized, public-safe** version of a university project that
applies **transformer-based topic modeling (BERTopic)** and **semantic similarity** to
analyze research abstracts and align them with editorial expertise.

> **Note:** Real journal data is **not included** due to confidentiality. The repo includes
> sample/synthetic data and a reproducible pipeline demonstrating the **methods** and **outputs**.

## What this does
- Loads a CSV of paper abstracts (sample in `data/sample_citations.csv`)
- Fits **BERTopic** to extract topics
- Assigns **semantic labels** to topics using a SentenceTransformer
- Tracks **topic trends over time**
- (Optionally) Compares topics vs editorial expertise (sample in `data/sample_editorial.csv`)
- Saves outputs to `outputs/` (CSV + PNG charts)

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the pipeline with the sample data
python src/pipeline.py   --citations_csv data/sample_citations.csv   --editorial_csv data/sample_editorial.csv   --out_dir outputs
```

Outputs:
- `outputs/abstracts_with_topics.csv` – topic + label per paper
- `outputs/top_topics_by_year.csv` – top-3 topics per year
- `outputs/topic_trends.png` – line chart of topic counts over time
- `outputs/papers_vs_editors_heatmap.png` – heatmap comparing paper volume vs editor counts (using sample editorial data)

## Data format

**Citations CSV** (example in `data/sample_citations.csv`):
- `Title` (str)
- `Abstract` (str)
- `Year` (int)

**Editorial CSV** (optional, example in `data/sample_editorial.csv`):
- `EditorName` (str)
- `Expertise` (str) – comma-separated labels (e.g., `"virtual reality, computer graphics"`)

## Disclaimers
- This repo demonstrates **code and methodology only**.
- No confidential data, credentials, or private endpoints are included.
- Replace `data/sample_*` with your own (non-confidential) data to reproduce.

## Tech Stack
Python, BERTopic, sentence-transformers, scikit-learn, pandas, numpy, matplotlib

---

*Last updated: 2025-11-03*
