# AGENTS.md — survey_analysis

## Quick start

```bash
pip install -r requirements.txt
# system dependency (brew/apt):
brew install graphviz
streamlit run src/app.py
# or: bash run_app.sh
```

## Architecture

Flat Python project restructured into a `src/` package.

| File | Role |
|------|------|
| `src/app.py` | Main Streamlit entry point |
| `src/common.py` | Model file parser, data preprocessing |
| `src/helpers.py` | Data normalization, LabelEncoder, column rename utils, config loading |
| `src/regression.py` | OLS regression via statsmodels, composite scores, graphviz graphs |
| `src/sem.py` | SEM via semopy (Model/inspect), graphviz graphs |
| `src/quality.py` | Cronbach's alpha, Pearson correlation, EFA |
| `src/cli.py` | CLI entry point (was broken, now imports from `src.regression`) |
| `config.json` | Centralized app configuration (output path, thresholds) |
| `pyproject.toml` | Modern Python packaging metadata |
| `chart_plot.py` | Standalone script, not part of the app |

## Key workflows

1. Upload an Excel **model file** (sheets: Demographic_Variables, Independent_Variables, Mediator_Variables, Dependent_Variables, Relations, Parameters) and a **survey data file** (Google Forms export).
2. Column indices in model files are **1-based**; converted to 0-based internally (`src/helpers.py:129-137`).
3. Variable names are normalized: spaces→underscores, parentheses removed.
4. All output (graphs, Excel files) goes to `./output/`. Directory is **cleared once on first run** (`src/helpers.py:21`).
5. `Image.MAX_IMAGE_PIXELS = None` is set to bypass PIL image size limits.
6. App configuration lives in `config.json` at project root.

## SEM specifics

- Uses `semopy` library. Spec format: `=~` for measurement models, `~` for regressions, `~~` for covariances.
- First construct's estimate is fixed to 1 (semopy baseline convention).
- SEM stats (CFI, RMSEA, chi²) are interpreted against thresholds from the model Excel's Parameters sheet.

## Gotchas

- `test_analysis.py` and `streamlit_app copy.py` are legacy files kept for reference; only `src/` files are active.
- `survey_analysis_sem.py:724` was fixed — the `create_model_spec_graph_short` function now defines `edge_labels`.
- No test framework, no lint/typecheck config setup.
- No CI/CD, no pre-commit hooks.
- Debug `print()` calls are left in production code across all modules.

## Change rules

- When making changes, always keep other files intact. Do not modify, clean up, or delete any file unless explicitly asked.
