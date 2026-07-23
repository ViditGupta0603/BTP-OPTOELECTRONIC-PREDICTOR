# Prediction pipeline — test guide

Project overview: [README.md](../README.md) · datasets: [DATASETS.md](DATASETS.md)

## 1. Install + train (once)

```bash
pip install -r requirements.txt
python scripts/enrich_chi_dataset.py   # χ + ETL/HTL libraries + layer lookup
python scripts/predict_stack.py --train
```

Trains:
- `data/models/perovskite_eg_regressor.joblib` — Eg from formula (~1748 absorbers)
- `data/models/stack_type_classifier.joblib` — Type from names+Eg (~726+ stacks)
- `data/models/layer_lookup.json` — known Eg/χ (absorber + ETL + HTL libraries)

Libraries:
- `data/perovskite_absorber_library.csv` — absorbers with Eg + χ (`chi_source`)
- `data/etl_material_library.csv` / `data/htl_material_library.csv` — contacts

**Eg stability:** library Eg is preferred over ML; ML/formula estimator is deterministic (rounded); LLM is **off by default** (UI never enables it; CLI needs `--llm`). Optional `--llm` uses formula-rules-only prompts (no web). The **web UI** shows **`predicted` badges only** when ML/estimate is used, and **hides χ**; library-sourced values are not labeled “lookup”.

## 2. Predict Type (no LLM)

```bash
# Known materials → physics Type from Eg+χ
python scripts/predict_stack.py --absorber K2TiI6 --etl TiO2 --htl MoO3

# Same with different contacts from the expanded pool
python scripts/predict_stack.py --absorber CsPb0.625Zn0.375IBr2 --etl SnO2 --htl NiO
python scripts/predict_stack.py --absorber Cs3Sb2Br9 --etl WS2 --htl CuI

# New-ish formula → ML Eg + Type-ML (if contacts known)
python scripts/predict_stack.py --absorber Cs2AgBiBr6 --etl TiO2 --htl MoO3

# List known layer names
python scripts/predict_stack.py --list-materials
```

## 3. Predict with LLM fill (Azure OpenAI)

Copy `.env.example` → `.env` and set your key:

```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://shopsifu.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

```bash
python scripts/llm_literature_assist.py --material K2GeI6 --role absorber
# or force Azure:
python scripts/llm_literature_assist.py --material K2GeI6 --provider azure

python scripts/predict_stack.py --absorber K2GeI6 --etl TiO2 --htl MoO3 --llm
```

## 4. What you get

| `method` | Meaning |
|----------|---------|
| `compute_from_Eg_chi` | Best — Type from band edges |
| `ml_type_from_names_and_Eg` | Type-ML (no χ); fields `predicted_absorber_*_type` |

## Decision order

1. Lookup Eg+χ → **compute** Type  
2. Else ML Eg (if needed) + **Type-ML** from names+Eg  
3. If `--llm` → fill missing χ/Eg → recompute Type when complete  

## Unknown / random absorbers (formula rules)

When the absorber is **not** in `layer_lookup`, Eg/χ come from
`scripts/formula_estimator.py`, which blends:

1. **Perovskite family rules** in `scripts/perovskite_rules.py` (taxonomy + Vegard end-members)
2. **ML formula estimator** (composition features)

Families: 3D ABX₃ (incl. mixed A / mixed halide), halide double A₂B′B″X₆,
vacancy-ordered A₂BX₆, 0D A₃B₂X₉, oxide perovskites, 2D RP/DJ (blocked),
contact ETL/HTL and non-perovskites (GaAs, CdTe, BeSiP₂, …) blocked as absorbers.

UI shows `confidence` / `OOD caution` when the estimate is out-of-distribution.
Same perovskite formula → same Eg/χ regardless of role.

See also suitability rules and evaluation caveats in [README.md](../README.md). 
