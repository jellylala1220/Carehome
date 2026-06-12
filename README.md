# Care Home Analysis Dashboard

This repository contains a Streamlit dashboard for care home observation analysis and NEWS2-focused reporting.

## Run Locally

Use the project virtual environment if it already exists:

```bash
source .venv/bin/activate
streamlit run app.py
```

If you are setting up the project from a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app is available at http://localhost:8501 by default.

## Current App

`app.py` is the main Streamlit entry point. The sidebar includes:

- Upload Data
- Care Home Analysis
- Batch Prediction
- Prediction Visualization
- Benchmark Grouping
- Regional Analysis
- Correlation Analysis

The app expects users to upload Excel data through the Streamlit interface. It does not automatically load the Phase 2 raw data folder.

## Input Data

The uploaded observation workbook should include the core columns used by the analysis:

- `Care Home ID`
- `Care Home Name`
- `Date/Time`
- `NEWS2 score`
- `Clinical concern?`
- `No of Beds`

Some analyses use additional physiological parameter columns when they are available.

## Repository Layout

```text
app.py                         # Streamlit dashboard
data_processor_simple.py       # Main processing and plotting functions used by app.py
data_processor.py              # Legacy/original processing module
prediction.py                  # Legacy standalone prediction script
requirements.txt               # Python dependencies for recreating the environment
Data/                          # Existing tracked sample/project workbooks
Coding/                        # Existing notebooks
docs/                          # Project workflow and version-management notes
```

Local-only folders such as `Stage 2 THA&LU/`, `phase2/raw/`, `Doc/`, and `Patent_Figures/` are intentionally ignored by Git unless a file is deliberately anonymised and force-added.

## Version Workflow

- Keep `main` as the stable Streamlit version.
- Use a feature branch for Phase 2 work, for example `codex/phase2-analysis`.
- Commit code, notebooks, documentation, and dependency changes.
- Keep raw Phase 2 data local by default.
- Save important stable milestones with Git tags, for example `v1.0-streamlit-phase1` or `v2.0-phase2`.

More detail is in `docs/VERSIONING.md`.

