# OptoStack — project development log

Chronological record of major build and fix steps for OptoStack (BTP), from dataset curation through deploy and UI disclaimers. Outcomes are tied to current repo behavior and verification artifacts where available.

Related: [TOOL_WORKFLOW.md](TOOL_WORKFLOW.md) · [TECHNICAL_WORKFLOW.md](TECHNICAL_WORKFLOW.md) · [DATASETS.md](DATASETS.md)

---

## Phase 0 — Repository bootstrap

**What we did**

- Initial commit of the BTP / optoelectronic predictor workspace.
- Documented repository layout, early data-flow notes, and `.gitignore`.
- Established that the product goal is **band-alignment Type + suitability screening**, not PCE.

**Outcome**

- Clean repo skeleton for datasets under `data/`, scripts under `scripts/`, and docs under `docs/`.

---

## Phase 1 — Dataset curation (literature / SCAPS / DFT)

**What we did**

- Curated peer-reviewed SCAPS/DFT **layer tables** into `data/raw/` (K₂TiI₆ family, CsPbZnIBr₂, Cs₃Sb₂Br₉, K₂GeI₆, BeSiP₂ reference, Paper5 double perovskites, Pilania oxides, …).
- Built **`perovskite_stack_dataset.csv`**: expand a small set of absorbers with full Eg+χ across a **pooled ETL×HTL** contact grid → **~726** labeled stacks (Types from physics on literature χ).
- Built **`perovskite_absorber_library.csv`**: Paper5 + Pilania + K₂GeI₆ + later verified lead-halides → **~1763** absorbers with Eg.
- Explicitly **excluded** non-perovskite device papers (DSSC / GaInP / CdTe, etc.) and failed extracts (e.g. Ca₃NI₃).
- Kept broader `opto_literature_dataset*.csv` as optional/historical 2D-heavy tables — **not** the primary perovskite product path.
- Added verification scripts (`verify_perovskite_dataset.py`, research-paper / range audits).

**Outcome**

- A perovskite-first catalog with DOI provenance (see [DATASETS.md](DATASETS.md)).
- Clear separation: stack Types for Type-ML training vs large Eg library for absorption/composition learning.

---

## Phase 2 — ML / LLM pipeline + free-text ETL/HTL + suitability

**What we did**

- Implemented **`predict_stack.py`** as the orchestrator: lookup → physics Type → ML fallback.
- Trained **Eg regressor** and **stack Type classifier** (joblibs under `data/models/`).
- Added **`literature_bands.py`**: Anderson Types from Eg+χ, CBO/VBO, **optoelectronic suitability** (YES / MARGINAL / NO).
- Enabled **free-text** absorber/ETL/HTL (not only dropdown enums), with library datalists in the UI.
- Optional **LLM** assist (`llm_literature_assist.py`, `--llm`) for missing Eg/χ — later locked **off by default** in the UI for reproducibility.
- Flask (and earlier Streamlit-era) UI for interactive screening; deploy-oriented bind settings introduced with the full pipeline drop.

**Outcome**

- Operators can type arbitrary formulas; known layers use libraries; unknowns use ML.
- Suitability is a deterministic function of the two interface Types — not a black-box PCE score.

---

## Phase 3 — Eg stability, χ enrichment, ETL/HTL libraries

**What we did**

- Added **`enrich_chi_dataset.py`**: fill absorber/contact χ, tag `chi_source` (`literature_SCAPS`, `ml_plus_family_prior`, …), write **`etl_material_library.csv`** / **`htl_material_library.csv`**, unify **`layer_properties.csv`**.
- Hardened **Eg stability**: prefer library Eg over ML; deterministic rounding for estimates.
- Extended layer lookup build so free-text absorbers can still reach **physics Type** when contacts have χ and absorber χ is literature or estimated.

**Outcome**

- Contacts are first-class libraries, not only raw SCAPS dumps.
- Estimated χ is usable for screening but clearly marked as non-measured.

---

## Phase 4 — Formula rules, FA/MA parsing, family priors

**What we did**

- Introduced **`perovskite_rules.py`**: family taxonomy, Vegard-like halide shifts, eligibility, denylists.
- Introduced **`formula_estimator.py`**: blend family/Vegard prior with composition ML for unknowns.
- Introduced / expanded **`formula_parse.py`**: FA/MA/organic cations, parenthesis groups (`HC(NH2)2…`), unicode subscripts.
- Mapped shorthand aliases (`FAPbI3` → `HC(NH2)2PbI3`, `MAPbI3` → `CH3NH3PbI3`, Spiro aliases).

**Outcome**

- Unknown but “family-like” perovskites get structured priors instead of pure unconstrained ML.
- Classic hybrid lead-halides resolve to library rows when present.

---

## Phase 5 — Accuracy benchmarks & verified lead-halides

**What we did**

- Built literature / browser / iterative evaluation scripts and reports:
  - `data/perovskite_test_set_literature_accuracy_report.md`
  - `data/browser_random_accuracy_report.md`
  - `data/iterative_accuracy_report.md`
  - `data/perovskite_prediction_benchmark.md`
- Added **verified ABX₃ lead-halide** Eg points (`verified_lead_halide_perovskites.csv`) into the absorber library (CsPbX₃, MAPbX₃, FAPbI₃/Br₃, mixed-halide Vegard points, …).
- Documented evaluation caveats: in-library self-consistency can inflate metrics; Type GT may be missing on literature Eg sets; GroupKFold leave-absorber-out is harder than mixed holdout.

**Outcome**

- Measurable Eg MAE on held literature sets; operational Type reports for library/browser practical sets.
- Classic hybrids no longer “missing” from the absorber library.

---

## Phase 6 — Lookup vs predicted labels → hide χ / predicted-only UI

**What we did**

- Introduced internal **`field_labels`** / source kinds (`lookup` vs `predicted`).
- First UI iteration showed both lookup and predicted provenance.
- Product decision: **do not show “lookup” labels** to users; show **`predicted` badges only** when ML/estimate is used.
- **Hide χ** in the web UI (notes + JSON scrubbing); χ remains for physics Type internally and for CLI debug.
- Method display: `compute_from_Eg_chi` → **physics** pill; strip “chi” from user-facing method text.

**Branches / merges:** `ui/hide-chi-and-lookup-labels` (and subsequent UI polish).

**Outcome**

- Cleaner operator UX: trust cues are “is this predicted?” rather than “lookup vs predicted” jargon.
- χ remains scientifically necessary but not a UI distraction / overclaim surface.

---

## Phase 7 — Non-perovskite blocking (ZnO, CZTS, …) & eval fixes (P3HT / TiO₂ / A₂TiI₆)

**What we did**

- Strengthened **perovskite-only gate**: block CZTS/CIGS/CdTe/… and **contact materials** mistakenly entered as absorbers (ZnO, TiO₂, …).
- Fixed **P3HT** optical Eg ≈ **1.9 eV** (organic HTL priors — avoid generic oxide-like Eg prior).
- Normalized **TiO₂** device lookup to **Eg = 3.2 eV**, χ = 4.0 (compact anatase device convention; literature stack rows may still preserve paper-specific 3.4).
- Improved **A₂TiI₆** family lookup / aliases; Spiro shorthand.
- Indirect-gap caveats (e.g. Cs₂AgBiBr₆).
- Captured results in `data/post_eval_fix_test_report.md` (22/22 targeted checks at the time; Cs₂SnI₆–TiO₂ Type III → MARGINAL called out as physics, not Eg bug).

**Branches:** `fix/eval-report-p3ht-tio2-scope`, merged via `deploy/bind-host-port` line.

**Outcome**

- Scope boundaries enforceable in UI and API.
- Contact Eg priors match common PSC tables; organic HTLs no longer silently wrong.

---

## Phase 8 — Round 2: Type I/II/III label swap, FAPbBr₃ halide, PEDOT caveat

**Problems**

- Anderson Types were effectively **mislabeled** (straddling vs broken-gap swap) relative to the intended I/II/III convention — stacks looked “all Type I”.
- **FAPbBr₃** (and unicode `FAPbBr₃`) risked resolving via **iodide** alias/prefix → wrong Eg (~1.48 instead of ~2.23).
- **PEDOT:PSS** treated like a normal semiconductor HTL despite degenerate/metallic polymer behavior.

**What we did**

- Corrected `junction_type` ordering/labels: **Type I = straddling**, **Type II = staggered**, **Type III = broken**.
- Made FA/MA aliasing **exact / halide-sensitive** (no FAPbI₃ prefix eating FAPbBr₃).
- Added **degenerate HTL** caveat for PEDOT:PSS / PEDOT.
- Verification: `data/round2_fix_verification.md`, `scripts/verify_round2_fixes.py`.

**Branch:** `fix/round2-type-halide-pedot` → merged to main.

**Outcome**

- Types vary across realistic stacks (I and II observed; Anderson unit tests pass for I/II/III).
- FAPbBr₃ Eg ≈ 2.23; MAPbBr₃ ≈ 2.32.
- PEDOT stacks carry an explicit reliability warning.

---

## Phase 9 — Round 3: A₂BX₆ Cs₂TiI₆/Br₆ collision, Type I/III coverage, FAPbCl₃

**Problems**

- **Cs₂TiI₆** collided with bromide-like Eg (~1.80) — iodide/bromide identity broken.
- Need stacks that demonstrate **Type I and Type III**, not only Type II.
- **FAPbCl₃** should sit near ~2.9 eV.

**What we did**

- Exact-match resolve for A₂BX₆ halide identity; restored **Eg(I) < Eg(Br) < Eg(Cl)** across Ti/Sn/Pd/Pt series (Cs₂TiI₆ ≈ 1.58, Cs₂TiBr₆ ≈ 1.88, Cs₂TiCl₆ ≈ 2.9).
- Coverage stacks: e.g. Cs₂SnI₆/TiO₂/P3HT (Type I ETL), Cs₂SnI₆/TiO₂/CFTS (Type III HTL), MAPbI₃/TiO₂/Spiro (Type II/II), etc.
- FAPbCl₃ / `HC(NH2)2PbCl3` Eg ≈ 2.9.
- Verification: `data/round3_fix_verification.md`, `scripts/verify_round3_fixes.py`.

**Branch:** `fix/round3-a2bx6-halide-types` → merged to main.

**Outcome**

- Halide series monotonic and distinct; Types I, II, and III all appear in verification stacks.
- Chloride FA lead-halide no longer missing/wrong.

---

## Phase 10 — Deploy bind, README, GitHub push

**What we did**

- Flask serves on **`0.0.0.0`** with **`PORT`** (default **7860**) for cloud/VM deployability; local access via 127.0.0.1.
- Expanded root **README** (features, datasets, pipeline, suitability, evaluation caveats, Docker/gunicorn sketch).
- Pushed feature work through GitHub; **`deploy/bind-host-port`** used as the integration/deploy branch and merged (PR #1 and follow-on merges).

**Outcome**

- Runnable on localhost and bind-all hosts for deployment platforms.
- README is the operator-facing entry; dataset provenance lives in `docs/DATASETS.md`.

---

## Phase 11 — ETL/HTL role disclaimer

**What we did**

- Added explicit UI copy: OptoStack **does not validate** conventional ETL vs HTL usage; **the person is responsible** for correct role assignment.
- Merged on branch `ui/etl-htl-role-disclaimer`.

**Outcome**

- Prevents false confidence when users swap contacts or put absorbers/contacts in the wrong boxes (absorber gate still blocks many misplaced absorbers; contact-role swap is intentional operator burden).

---

## Phase 12 — Documentation pack (this deliverable)

**What we did**

- Authored:
  - `docs/TOOL_WORKFLOW.md` — full operator workflow
  - `docs/TECHNICAL_WORKFLOW.md` — engineering pipeline
  - `docs/PROJECT_DEVELOPMENT_LOG.md` — this history
- Pointed root README at all three; retained `docs/WORKFLOW.md` as a short pointer to the new set.

**Outcome**

- Single place for “how to use”, “how it works”, and “what we built, in order”.

---

## Timeline snapshot (git merges on main)

| Commit / merge | Theme |
|----------------|-------|
| Initial + layout docs | Bootstrap |
| Large pipeline + README + datasets | Phase 1–5 core |
| PR #1 `deploy/bind-host-port` | Deploy bind + early full stack |
| `fix/eval-report-p3ht-tio2-scope` | P3HT / TiO₂ / A₂TiI₆ / scope |
| `fix/round2-type-halide-pedot` | Type labels, FAPbBr₃, PEDOT |
| `fix/round3-a2bx6-halide-types` | Cs₂TiX₆, Type I/III, FAPbCl₃ |
| `ui/etl-htl-role-disclaimer` | Role responsibility copy |
| `docs/tool-technical-history-workflows` | This documentation pack |

---

## Current product posture (as of this log)

- **Perovskite-only** absorber screening with free-text contacts.
- **Physics Type** when Eg+χ complete; else **Type-ML**.
- **Suitability** from Types only.
- UI: **predicted badges only**, **χ hidden**, **LLM off**, **ETL/HTL disclaimer**.
- Regression scripts for Round 2 / Round 3 and post-eval P3HT/TiO₂/scope checks.
