# OptoStack — technical workflow (engineering)

Pipeline architecture, modules, data flow, Eg/χ resolution, Anderson Type physics, model artifacts, libraries, and deploy.

Related: [TOOL_WORKFLOW.md](TOOL_WORKFLOW.md) (operators) · [PROJECT_DEVELOPMENT_LOG.md](PROJECT_DEVELOPMENT_LOG.md) · [DATASETS.md](DATASETS.md) · [scripts/README.md](../scripts/README.md)

---

## 1. Architecture overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Flask UI   │────▶│  predict_stack   │────▶│  literature_bands   │
│  app.py     │     │  (orchestrator)  │     │  junction_type +    │
└─────────────┘     └────────┬─────────┘     │  suitability        │
                             │               └─────────────────────┘
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐
│ layer_lookup  │  │ formula_estimate│  │ Type-ML joblib   │
│ + CSV libs    │  │ + perovskite_   │  │ (names + Eg)     │
│               │  │   rules + parse │  │                  │
└───────────────┘  └─────────────────┘  └──────────────────┘
```

**Runtime entry points**

| Entry | Role |
|-------|------|
| `app.py` | Flask UI; always calls `predict_stack(..., use_llm=False)`; binds `0.0.0.0:$PORT` |
| `scripts/predict_stack.py` | Train + CLI predict; optional `--llm` |
| `scripts/enrich_chi_dataset.py` | Offline χ fill + ETL/HTL libraries + `layer_properties.csv` |

---

## 2. Module map

| Module | Responsibility |
|--------|----------------|
| **`predict_stack.py`** | Orchestration: normalize names, perovskite gate, literature stack hit, layer resolve, ML Type fallback, train Eg/Type models, build `layer_lookup.json` |
| **`literature_bands.py`** | `Layer` (Eg, χ), `junction_type` (Anderson I/II/III), CBO/VBO, `optoelectronic_suitability`, `stack_row` |
| **`formula_estimator.py`** | Deterministic unknown-formula Eg/χ: family/Vegard prior × ML blend; organic HTL priors (P3HT, Spiro, …) |
| **`perovskite_rules.py`** | Taxonomy, eligibility, family Eg/χ priors, Vegard end-members, non-perovskite / contact denylists, 2D RP/DJ block |
| **`formula_parse.py`** | `normalize_formula_text` (the one unicode fold), FA/MA/organic cations, parenthesis groups, **halide-sensitive** aliases (`FAPbBr3` ≠ `FAPbI3` prefix) |
| **`enrich_chi_dataset.py`** | Enrich absorber χ (`chi_source`), write ETL/HTL libs, unify `layer_properties.csv` |
| **`llm_literature_assist.py`** | Optional CLI LLM Eg/χ (domain-context prompts; no web); unused by UI |

Supporting: `build_perovskite_dataset.py`, `verify_*`, `eval_*`, `benchmark_predictions.py`, `iterative_accuracy_loop.py`.

---

## 3. Data flow (predict)

```
absorber, etl, htl (raw strings)
        │
        ▼
normalize_formula_text  ← unicode fold: ₀-₉ / ⁰-⁹ / full-width / dashes /
        │                 no-break + zero-width space / interpunct → ASCII
        ▼
normalize_material_name  ← formula_parse.canonicalize_material_alias
        │
        ▼
check_absorber_perovskite  ← perovskite_rules + contact role index
        │
   ineligible? ──▶ blocked result (not_perovskite)
        │
        ▼
lookup_literature_stack  (exact triple in perovskite_stack_dataset)
        │
   hit? ──▶ literature_stack_row result (+ caveats)
        │
        ▼
resolve_layer / ml_estimate_eg_chi for missing Eg/χ
        │
        ▼
all χ present? ──yes──▶ Layer×3 → stack_row → compute_from_Eg_chi
        │                        → optoelectronic_suitability
        no
        ▼
ml_type(absorber, partner, Eg…) → ml_type_from_names_and_Eg
        │
        ▼
degenerate HTL caveat (PEDOT…); indirect-gap notes
```

**Field labels:** `_src_kind` maps internal sources → `lookup` | `predicted` | `user`. UI shows **only** `predicted` badges and **strips χ** / “lookup” wording (`app._ui_notes`, `_ui_result_json`).

---

## 4. Eg / χ resolution

### Sources (priority)

1. **User override** (CLI/API args `eg` / `chi`)
2. **Layer lookup** (`data/models/layer_lookup.json`), built from:
   - Verified experimental / lead-halide tables
   - `perovskite_absorber_library.csv` (Eg; χ with `chi_source`)
   - `etl_material_library.csv` / `htl_material_library.csv`
   - `layer_properties.csv`, selected raw SCAPS, opto literature layers
   - Contact overrides: device **TiO₂** preferred at **Eg=3.2 eV**, **χ=4.0** (vs some SCAPS rows at 3.4)
3. **Formula estimator** (`formula_estimator` / `ml_estimate_eg_chi`):
   - Family prior + Vegard from `perovskite_rules`
   - Composition ML (`formula_eg_chi_estimator.joblib` when trained)
   - Named organic HTL priors (e.g. P3HT Eg≈1.9)
4. **Optional LLM** (`--llm` only) — not in UI path

### χ policy

- χ is required for **physics** Type (`CBM = -χ`, `VBM = -(χ+Eg)`).
- Absorber χ may be `literature_SCAPS` or estimated (`ml_plus_family_prior`, …) — see [DATASETS.md](DATASETS.md).
- **Estimated χ = screening only.**
- UI **hides** χ; CLI may still print it.

### Eg stability

- Library / verified Eg wins over ML.
- Estimator is deterministic (rounded); same formula → same Eg/χ for perovskite families (role-independent).
- Contacts still use role-specific χ priors when estimating unknowns.

---

## 5. Junction Type physics (Anderson I / II / III)

Implemented in `literature_bands.junction_type`:

```text
CBM = -χ
VBM = -(χ + Eg)

Type III (broken):  VBM_a ≥ CBM_b  OR  VBM_b ≥ CBM_a
Type I  (straddle): one material's [VBM, CBM] contains the other
Type II (stagger):  otherwise (offset, no broken gap)
```

Historical note: Round 2 fixed a **label swap** so Type I = straddling and Type III = broken gap (matching the docstring and published CBM/VBM checks). Round 3 added stacks that exercise **all three** Types (e.g. Cs₂SnI₆/TiO₂/CFTS → HTL Type III).

### Suitability (`optoelectronic_suitability`)

| Verdict | Condition |
|---------|-----------|
| YES | Both interfaces ∈ {Type I, Type II} |
| MARGINAL | Exactly one Type III |
| NO / UNKNOWN | Both Type III, or missing Type |

---

## 6. Perovskite gate & contact roles

`check_absorber_perovskite` / `looks_like_perovskite_absorber`:

- Eligible families: 3D ABX₃, mixed A/halide, halide double A₂B′B″X₆, vacancy-ordered A₂BX₆, A₃B₂X₉, oxide perovskites, …
- Blocked: denylist (CZTS, GaAs, …), simple oxide semiconductors as absorber, known ETL/HTL materials in absorber slot, 2D RP/DJ markers, monolayer prefixes
- Contact libraries feed a **role index** so ZnO/TiO₂/… in the absorber field are blocked as misplaced contacts

**Role disclaimer:** pipeline does **not** reject ZnO-as-HTL or MoO₃-as-ETL; only absorber eligibility is gated. UI states operator responsibility for ETL/HTL assignment.

---

## 7. Model artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| Eg regressor | `data/models/perovskite_eg_regressor.joblib` | Absorber Eg from formula features (train on absorber library) |
| Type classifier | `data/models/stack_type_classifier.joblib` | Absorber–partner Type from names + Eg |
| Formula Eg/χ estimator | `data/models/formula_eg_chi_estimator.joblib` | Optional blended estimator (via `formula_estimator --train`) |
| Layer lookup | `data/models/layer_lookup.json` | Runtime Eg/χ cache |
| Train meta | `data/models/train_meta.json` | Training summary |

**Train**

```bash
python scripts/enrich_chi_dataset.py
python scripts/predict_stack.py --train
# optional:
python scripts/formula_estimator.py --train
```

Sklearn stack: `RandomForestRegressor` (Eg), `GradientBoostingClassifier` (Type), `ColumnTransformer` + `OneHotEncoder` / `StandardScaler`.

---

## 8. Primary data libraries

| File | Role |
|------|------|
| `data/perovskite_stack_dataset.csv` | ~726 labeled stacks (3 absorbers × pooled ETL×HTL) |
| `data/perovskite_absorber_library.csv` | ~1763 absorbers Eg (+ χ / `chi_source`) |
| `data/etl_material_library.csv` | ETL Eg+χ |
| `data/htl_material_library.csv` | HTL Eg+χ |
| `data/layer_properties.csv` | Unified layer table from enrich |
| `data/raw/*` | SCAPS/DFT source tables, verified lead halides |

Provenance and DOIs: [DATASETS.md](DATASETS.md).

---

## 9. Python libraries

From `requirements.txt`:

| Package | Use |
|---------|-----|
| `pandas`, `numpy` | Tables / arrays |
| `scikit-learn` | Eg/Type models, preprocessing |
| `joblib` | Model persistence |
| `flask` | Web UI |
| `matminer` | (dataset / composition tooling where used) |
| `openai`, `anthropic` | Optional CLI LLM only |

Python **3.10+** recommended.

---

## 10. Deploy bind (host / port)

`app.py` main:

```python
port = int(os.environ.get("PORT", 7860))
host = "0.0.0.0"
app.run(host=host, port=port, debug=False)
```

| Setting | Value |
|---------|-------|
| Default host | **`0.0.0.0`** (all interfaces) |
| Default port | **`7860`** (`PORT` env override) |
| Local URL | http://127.0.0.1:7860 |

Gunicorn example:

```bash
gunicorn -b 0.0.0.0:7860 "app:app"
# or: gunicorn -b 0.0.0.0:$PORT "app:app"
```

Do not bake secrets into images; mount `.env` only if using CLI LLM.

---

## 11. UI scrubbing contract

`app.py` enforces:

- Strip χ / electron-affinity notes and JSON keys
- Replace “lookup” wording with “library” in notes
- Surface **`predicted`** badges only (never “lookup” pills)
- Map method `compute_from_Eg_chi` → display pill **physics**
- Show ETL/HTL **role-assignment disclaimer** under the form

Pipeline may still return full χ in dicts for CLI/debug; UI presentation is intentionally narrower.

---

## 12. Evaluation hooks

| Script / report | Focus |
|-----------------|-------|
| `docs/OPTOSTACK_FULL_REPORT.md` | Advisor-facing overview + CV scorecard |
| `cross_validate_models.py` → `data/cross_validation_report.md` | Whole-tool GroupKFold (Eg + Type) |
| `eval_literature_test_set.py` | Tool Eg vs literature test set |
| `eval_browser_random_test.py` | Optional browser-sourced random perovskites |
| `benchmark_predictions.py` / `iterative_accuracy_loop.py` | Optional older eval loops |
| `verify_round2_fixes.py` / `verify_round3_fixes.py` | Regression gates for label/halide fixes |

Caveat: Stratified/random Type accuracy can look near-perfect via name memorization; prefer GroupKFold leave-absorber-out. Literature CSV may score **Eg only**.
