# OptoStack (BTP)

Screen **perovskite absorber + ETL + HTL** stacks for junction **Type I / II / III** and **optoelectronic suitability** — not PCE.

## Features

- Free-text absorber / ETL / HTL (library suggestions via datalist)
- Junction Types + suitability verdict (**YES** / **MARGINAL** / **NO**)
- **Predicted** badges only when ML/estimate is used (no “lookup” labels in the UI)
- **χ (electron affinity) hidden** in the web UI; used internally for physics Type
- Perovskite gate — non-perovskite / contact / 2D RP–DJ absorbers are blocked
- Family rules + Vegard blend (`perovskite_rules.py`) with formula parse for FA/MA/organics
- Material libraries for absorbers, ETLs, HTLs; deterministic ML when unknown
- Optional Azure/OpenAI LLM fill via CLI (`--llm`); **off by default** in the UI

## Project structure

```
btp/
├── app.py                 # Flask web UI
├── requirements.txt
├── .env.example           # optional LLM keys (do not commit .env)
├── docs/
│   ├── OPTOSTACK_FULL_REPORT.md   # ★ advisor-ready full report
│   ├── REPORT_README.txt          # how to present / zip the report
│   ├── report_figures/            # CV figures for the full report
│   ├── TOOL_FLOWCHART.md          # runtime + training Mermaid
│   ├── TOOL_WORKFLOW.md           # operator: how to use end-to-end
│   ├── TECHNICAL_WORKFLOW.md      # engineering: pipeline & deploy
│   ├── PROJECT_DEVELOPMENT_LOG.md # chronological build history
│   ├── DATASETS.md                # dataset catalog & provenance
│   └── WORKFLOW.md                # short index → above docs
├── scripts/               # pipeline, eval, dataset builders
│   ├── predict_stack.py   # main train + predict
│   ├── cross_validate_models.py
│   ├── perovskite_rules.py
│   ├── formula_estimator.py / formula_parse.py
│   ├── literature_bands.py
│   └── …                  # see scripts/README.md
├── data/
│   ├── perovskite_*.csv   # primary libraries & stacks (~1030 stacks)
│   ├── etl_material_library.csv / htl_material_library.csv
│   ├── models/            # committed joblibs + layer_lookup.json (fast cloud cold start)
│   ├── cross_validation_report.md
│   ├── figures/           # CV PNGs (also copied to docs/report_figures/)
│   └── raw/               # curated SCAPS/DFT tables
└── research paper/        # reference PDFs
```

## Datasets

| File | Role |
|------|------|
| `data/perovskite_stack_dataset.csv` | ~1030 stacks with Type labels (SCAPS pool + expansion + CdTe training-only) |
| `data/perovskite_absorber_library.csv` | ~1764 absorber Eg (+ χ with `chi_source`) |
| `data/etl_material_library.csv` / `htl_material_library.csv` | Contact Eg+χ |
| `data/layer_properties.csv` | Unified layer table (from enrich) |
| `data/perovskite_test_set_literature.csv` | Literature Eg validation set |
| `data/models/layer_lookup.json` | Runtime Eg/χ lookup |

**Verified vs estimated:** stack Types and many Eg values come from peer-reviewed SCAPS/DFT tables. Absorber χ may be literature (`literature_SCAPS`) or estimated (`ml_plus_family_prior`, …) — estimated χ is for **screening only**. Full provenance: [docs/DATASETS.md](docs/DATASETS.md).

## Pipeline

### Prediction (runtime)

1. **Perovskite gate** — block non-absorber / non-perovskite formulas  
2. Resolve layers from **library lookup** (Eg; χ internally)  
3. If complete → **physics Type** from band edges (`compute_from_Eg_chi`)  
4. Else **formula estimator** (family/Vegard + ML) for missing Eg/χ → physics Type when possible  
5. Else **Type-ML** from names + Eg  
6. Optional `--llm` fills gaps, then recompute  
7. Map Types → **suitability** (YES / MARGINAL / NO)

### Train / enrich

```bash
python scripts/enrich_chi_dataset.py   # χ + ETL/HTL libs + layer properties
python scripts/predict_stack.py --train
```

Produces `data/models/perovskite_eg_regressor.joblib`, `stack_type_classifier.joblib`, `layer_lookup.json`.

Detail: [docs/TOOL_WORKFLOW.md](docs/TOOL_WORKFLOW.md) · [docs/TECHNICAL_WORKFLOW.md](docs/TECHNICAL_WORKFLOW.md).

## Suitability rules

From absorber–ETL and absorber–HTL Types (**not** a PCE model):

| Verdict | Rule |
|---------|------|
| **YES** | Both interfaces Type I or Type II |
| **MARGINAL** | Exactly one Type III |
| **NO** | Both Type III (or Type missing → UNKNOWN) |

Type I/II are treated as acceptable for confinement/separation; Type III (broken gap) is usually not preferred for standard opto stacks.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ recommended. Copy `.env.example` → `.env` only if you want optional LLM assist — not required for library + ML screening. **Never commit `.env` or API keys.**

## Train / enrich

```bash
python scripts/enrich_chi_dataset.py
python scripts/predict_stack.py --train

# Optional rebuild / verify
python scripts/build_perovskite_dataset.py
python scripts/verify_perovskite_dataset.py
```

## Run

### Web UI

```bash
python app.py
```

Serves on **`0.0.0.0:$PORT`** (default **7860**). Open **http://127.0.0.1:7860**. First run trains models if missing. Enter any formulas → **Predict Type & suitability**.

### CLI

```bash
# Known stack → physics Type
python scripts/predict_stack.py --absorber K2TiI6 --etl TiO2 --htl MoO3

# Less-known absorber → estimator / Type-ML
python scripts/predict_stack.py --absorber Cs2AgBiBr6 --etl TiO2 --htl MoO3

# Optional LLM fill
python scripts/predict_stack.py --absorber K2GeI6 --etl TiO2 --htl MoO3 --llm

python scripts/predict_stack.py --list-materials
```

## Evaluation / accuracy

**Advisor-facing package:** [`docs/OPTOSTACK_FULL_REPORT.md`](docs/OPTOSTACK_FULL_REPORT.md) ([`docs/REPORT_README.txt`](docs/REPORT_README.txt) for presentation order).

| Report | What it measures |
|--------|------------------|
| [`docs/OPTOSTACK_FULL_REPORT.md`](docs/OPTOSTACK_FULL_REPORT.md) | Full overview + CV scorecard + workflows + examples |
| [`data/cross_validation_report.md`](data/cross_validation_report.md) | Whole-tool GroupKFold CV (Eg + Type) |
| [`data/perovskite_test_set_literature_accuracy_report.md`](data/perovskite_test_set_literature_accuracy_report.md) | Tool Eg vs literature test set |

```bash
python scripts/cross_validate_models.py
python scripts/eval_literature_test_set.py
```

**Caveats (read these):**

- **Library vs predicted:** in-library materials can look near-perfect (self-consistent lookup); judge **unseen** / `predicted` rows for true generalization.
- **Type vs Eg:** high Type accuracy on random splits can reflect name memorization; prefer GroupKFold leave-absorber-out in the CV report. Literature test set scores **Eg only** (no Type GT).
- Suitability (YES/MARGINAL/NO) is a deterministic Type rule — not a supervised model.

## Deployment

`python app.py` binds **`0.0.0.0`** on **`PORT`** (default **7860**). Local URL: http://127.0.0.1:7860.

```bash
# Example: gunicorn
pip install gunicorn
gunicorn -b 0.0.0.0:7860 "app:app"
# Or: gunicorn -b 0.0.0.0:$PORT "app:app"  after export PORT=7860
```

Docker sketch:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
ENV PORT=7860
CMD gunicorn -b 0.0.0.0:${PORT} "app:app"
```

Do not bake secrets into images; mount `.env` at runtime if using LLM.

## How to interpret results

| Signal | Trust for pre-DFT triage? |
|--------|---------------------------|
| Physics method + known contacts, no `predicted` badges | Highest — band-edge Type from curated values |
| `predicted` Eg/Type or low confidence / OOD caution | Screening only — verify before heavy DFT/SCAPS |
| **YES** suitability | Both junctions look Type I/II — worth deeper sim |
| **MARGINAL** / **NO** | At least one broken-gap interface — redesign contacts or check χ/Eg assumptions |
| Blocked / not perovskite | Outside project scope |

Use OptoStack to **rank and reject** stacks quickly; confirm promising YES cases with literature χ/Eg and device simulation.

## Limitations

- Not a PCE, stability, or defect predictor  
- Stack training absorbers with full SCAPS χ are few; Type-ML generalizes names+Eg  
- Estimated χ and free-text formulas can be wrong far from the library  
- 2D RP/DJ and non-perovskites are intentionally blocked as absorbers  
- UI hides χ; CLI/JSON may still expose internal fields for debugging  

## Docs

- [docs/OPTOSTACK_FULL_REPORT.md](docs/OPTOSTACK_FULL_REPORT.md) — **full advisor report** (start here for demos)  
- [docs/REPORT_README.txt](docs/REPORT_README.txt) — how to present / share the report  
- [docs/TOOL_FLOWCHART.md](docs/TOOL_FLOWCHART.md) — runtime + training Mermaid flowcharts  
- [docs/TOOL_WORKFLOW.md](docs/TOOL_WORKFLOW.md) — **operator** end-to-end use (inputs, outputs, suitability, disclaimers)  
- [docs/TECHNICAL_WORKFLOW.md](docs/TECHNICAL_WORKFLOW.md) — **engineering** pipeline, modules, Types, artifacts, deploy  
- [docs/PROJECT_DEVELOPMENT_LOG.md](docs/PROJECT_DEVELOPMENT_LOG.md) — chronological build / fix history  
- [docs/DATASETS.md](docs/DATASETS.md) — datasets & verification  
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — short index of the workflow docs  
- [scripts/README.md](scripts/README.md) — script index  

## License / citation

No project LICENSE file is included. Cite the source DOIs listed in [docs/DATASETS.md](docs/DATASETS.md) when using curated tables or reporting results.
