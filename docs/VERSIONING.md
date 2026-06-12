# Versioning and Phase 2 Workflow

## Current Version Management

This project uses Git for local version control and GitHub as the remote repository:

- Remote: `https://github.com/jellylala1220/Carehome.git`
- Stable branch: `main`
- Working branches: use `codex/...` or `phase2/...` names for new work

The Streamlit app entry point is `app.py`. The app should remain runnable from the repository root with:

```bash
source .venv/bin/activate
streamlit run app.py
```

## What Belongs in Git

Commit these files:

- Python source files
- notebooks that document analysis decisions
- `requirements.txt`
- README and project documentation
- small anonymised example data when it is safe to share

Do not commit these by default:

- `.venv/`
- `__pycache__/`
- `.DS_Store`
- raw Phase 2 data
- private patent material
- generated charts or temporary exports

## Phase 2 Data Rule

Raw Phase 2 files should stay in a local-only folder such as:

```text
Stage 2 THA&LU/
phase2/raw/
phase2/private/
local_data/
```

These folders are ignored by Git. If a dataset has been anonymised and is intentionally safe to version, put it in a clearly named tracked location such as:

```text
Data/anonymised_phase2_sample.xlsx
```

Then add it deliberately:

```bash
git add Data/anonymised_phase2_sample.xlsx
```

## Suggested Day-to-Day Workflow

Before starting a new task:

```bash
git status
git switch main
git pull
git switch -c codex/phase2-task-name
```

During work:

```bash
streamlit run app.py
git status
git diff
```

When the app still works:

```bash
git add app.py data_processor_simple.py requirements.txt README.md docs/
git commit -m "Add phase2 task name"
```

After a stable milestone:

```bash
git switch main
git merge codex/phase2-task-name
git tag v2.0-phase2
git push origin main --tags
```

## Dependency Rule

Use `requirements.txt` as the source of truth for recreating the Python environment. The `.venv/` folder is local machine state and should not be committed.

If a new package is needed for Phase 2, install it locally and add it to `requirements.txt`.
