# OptoStack — operator workflow (how to use the tool)

End-to-end guide for **users and operators**: what to enter, what happens, how to read results, suitability rules, and when to trust outputs before DFT/SCAPS.

Related: [OPTOSTACK_FULL_REPORT.md](OPTOSTACK_FULL_REPORT.md) (advisor report) · [TOOL_FLOWCHART.md](TOOL_FLOWCHART.md) (runtime + training Mermaid flowcharts) · [TECHNICAL_WORKFLOW.md](TECHNICAL_WORKFLOW.md) (engineering) · [PROJECT_DEVELOPMENT_LOG.md](PROJECT_DEVELOPMENT_LOG.md) (history) · [DATASETS.md](DATASETS.md) · [README.md](../README.md) · [cross-validation report](../data/cross_validation_report.md)

---

## 1. What OptoStack does (and does not)

**Does:** Screen a **perovskite absorber + ETL + HTL** stack for Anderson junction **Type I / II / III** at each interface and an **optoelectronic suitability** verdict (**YES** / **MARGINAL** / **NO**).

**Does not:** Predict PCE, stability, defects, mobility, or fabrication yield. It is a **pre-DFT / pre-device triage** tool for band-alignment screening.

---

## 2. Quick start

```bash
pip install -r requirements.txt
python scripts/enrich_chi_dataset.py   # once (or after library changes)
python scripts/predict_stack.py --train
python app.py
```

Open **http://127.0.0.1:7860** (server binds `0.0.0.0` on `PORT`, default **7860**).

CLI equivalent:

```bash
python scripts/predict_stack.py --absorber MAPbI3 --etl TiO2 --htl Spiro-OMeTAD
python scripts/predict_stack.py --list-materials
```

Optional Azure/OpenAI fill is **CLI-only** (`--llm`). The **web UI never enables LLM**.

---

## 3. Inputs

| Field | What to enter | Tips |
|-------|---------------|------|
| **Absorber** | Perovskite formula | ABX₃, A₂BB′X₆, A₂BX₆, A₃B₂X₉, oxide perovskites, alloys. Shortcuts: `FAPbI3`, `MAPbBr3`, unicode subscripts (`FAPbBr₃`) OK. |
| **ETL** | Electron-transport layer | Suggestions from datalist (`TiO2`, `SnO2`, `ZnO`, …). Free text allowed. |
| **HTL** | Hole-transport layer | Suggestions (`Spiro-OMeTAD`, `NiO`, `P3HT`, …). Free text allowed. |

### ETL / HTL role responsibility (important)

OptoStack **does not validate** whether a material is conventionally used as ETL or HTL. You can put ZnO in the HTL box or MoO₃ in the ETL box and the tool will still compute Types from the **roles you assigned**.

**The operator is responsible for correct role assignment.** Misplacing contacts will produce misleading Types and suitability.

### Perovskite-only absorber scope

Absorbers that are **not** perovskite / perovskite-inspired are **blocked**, including:

- Thin-film PV: **CZTS**, **CIGS**, **CdTe**, **GaAs**, **Si**, …
- Common **contact oxides/organics** mistakenly entered as absorber: **ZnO**, **TiO2**, **SnO2**, **MoO3**, **Spiro-OMeTAD**, …
- **2D RP/DJ** absorbers (PEA/BA/… markers) — out of scope for this screening gate

Blocked runs show a clear **not perovskite / blocked** message; no Type/suitability is claimed.

---

## 4. What happens when you click Predict

High-level order (details in [TECHNICAL_WORKFLOW.md](TECHNICAL_WORKFLOW.md)):

1. **Normalize** names (FA/MA aliases, unicode digits, dashes).
2. **Perovskite gate** — block ineligible absorbers.
3. If the exact triple exists in the curated **stack table** → return that **known stack** result.
4. Else resolve **Eg** (and internally **χ**) from libraries / formula estimator.
5. If Eg+χ complete for all three layers → **physics Type** from band edges (`compute_from_Eg_chi`).
6. Else fall back to **Type-ML** from names + Eg.
7. Map the two interface Types → **suitability** verdict.

Library Eg is preferred over ML. Same formula → same deterministic estimate when ML/rules are used.

---

## 5. How to read the outputs

### Method pills

| Pill | Meaning |
|------|---------|
| **physics** | Types from vacuum band edges (Eg + χ). Best path when values are complete. |
| **known stack** | Exact absorber/ETL/HTL row from the curated SCAPS stack table. |
| **ML Type** | Types from the name+Eg classifier (χ incomplete or unavailable). |
| **no LLM** / **LLM** | Whether optional LLM fill was used (UI: always no LLM). |
| **confidence** / **OOD caution** | Estimator uncertainty for unknown formulas. |

### Junction Types

| Type | Physics picture | Screening implication |
|------|-----------------|------------------------|
| **Type I** | Straddling (one gap contains the other) | Usually acceptable for confinement |
| **Type II** | Staggered offsets | Usually acceptable for separation |
| **Type III** | Broken gap (VBM of one ≥ CBM of the other) | Usually **not** preferred for standard opto stacks |

You get **Absorber–ETL** and **Absorber–HTL** Types separately.

### Suitability rules

| Verdict | Rule |
|---------|------|
| **YES** | Both interfaces Type I or Type II |
| **MARGINAL** | Exactly one interface is Type III |
| **NO** | Both Type III (or a Type missing → **UNKNOWN**) |

This is **not** a PCE model — only a band-alignment screen.

### `predicted` badges

- The UI shows a **`predicted`** badge **only** when ML / formula estimate was used for that field (Eg or Type).
- **Library-sourced** values are **not** labeled “lookup” in the UI (internal sources may still say `lookup`).
- If you see **`predicted`**, treat that number/Type as **screening-grade**, not literature-verified.

### What you do **not** see in the web UI

- **χ (electron affinity)** is used **internally** for physics Type but **hidden** in the UI (and scrubbed from displayed notes/JSON).
- CLI / raw JSON may still expose χ for debugging.

### Extra caveats you may see

- **Indirect-gap** absorbers (e.g. Cs₂AgBiBr₆, Cs₂TiX₆): YES carries an optical-absorption caveat.
- **PEDOT:PSS** (and similar degenerate/metallic polymer HTLs): Eg-based Anderson Type is **unreliable**; a caveat is attached even if Types look fine.

---

## 6. When to trust results before DFT / SCAPS

| Situation | Trust for pre-DFT triage? |
|-----------|---------------------------|
| **physics** / **known stack**, common contacts, **no** `predicted` badges | Highest — curated Eg (+ internal χ) → band-edge Types |
| Library Eg but any `predicted` Type or contact fill | Medium — check contact χ/Eg assumptions |
| `predicted` Eg and/or **ML Type**, low confidence, OOD caution | Screening only — verify before heavy DFT |
| **YES** with known lead-halide + TiO₂/Spiro-style contacts | Worth deeper sim / literature check |
| **MARGINAL** / **NO** | Redesign contacts or verify χ/Eg — do not treat as “bad PCE” |
| **Blocked / not perovskite** | Outside tool scope |
| PEDOT:PSS or highly doped polymers as HTL | Do not trust Eg-based Type alone |
| Operator swapped ETL/HTL roles | Invalid for device conclusions |

**Practical rule:** use OptoStack to **rank and reject** stacks quickly; confirm promising **YES** cases with literature Eg/χ and device simulation (SCAPS/DFT) before claiming device quality.

---

## 7. Disclaimers (read before publishing numbers)

1. **Not PCE / stability / defect prediction.**
2. **Perovskite absorbers only** — non-perovskites are blocked by design.
3. **ETL/HTL role is your responsibility** — the tool does not enforce conventional contact roles.
4. **Estimated χ** (when used internally) and **formula estimates** can be wrong far from the library.
5. **Stack training absorbers** with full original SCAPS χ are few; Type-ML generalizes from names+Eg on an expanded contact pool.
6. **In-library** materials can look near-perfect (self-consistent lookup); judge generalization on **unseen** / `predicted` cases.
7. Prefer **physics + no predicted badges** over ML Type when deciding what goes to DFT.

---

## 8. Example operator checklist

1. Confirm absorber is a perovskite formula (not CZTS/ZnO/…).
2. Assign ETL and HTL correctly for your intended architecture.
3. Prefer library suggestions when available.
4. Read method pill + Types + suitability + any caveats.
5. Note any **`predicted`** badges → plan literature/DFT follow-up.
6. For YES + physics + no predicted → shortlist for SCAPS/DFT.
7. For MARGINAL/NO → change contacts or verify band edges before DFT spend.

---

## 9. CLI / LLM (optional)

```bash
# Copy .env.example → .env with Azure/OpenAI keys first
python scripts/predict_stack.py --absorber K2GeI6 --etl TiO2 --htl MoO3 --llm
```

LLM is off by default and **not** available from the Flask form. Prefer library + formula estimator for reproducible screening.
