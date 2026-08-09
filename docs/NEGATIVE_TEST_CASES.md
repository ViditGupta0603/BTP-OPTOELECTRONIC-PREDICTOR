# OptoStack — Negative Test Cases (QA)

**Audience:** advisors, TAs, and students verifying that OptoStack **rejects** out-of-scope or nonsensical inputs instead of inventing junction Types or band gaps.  
**Companion:** [USER_MANUAL.md](USER_MANUAL.md) (how to run the UI).  
**Branch / code basis:** validation in `scripts/predict_stack.py` (`check_absorber_perovskite`, `validate_contact_material`), `scripts/perovskite_rules.py` (`NON_PEROVSKITE_ABSORBERS`, contact families), and blocked UI in `app.py`. Contact garbage rejection was merged from `fix/reject-garbage-etl-htl` (already on this line of work).

---

## Purpose of negative testing

Positive demos show a valid perovskite stack returning Type I/II/III and Eg. Negative testing confirms that:

1. Non-perovskite absorbers are **blocked** (perovskite-only screening).
2. Empty / whitespace / garbage ETL–HTL strings are **blocked** (no invented contact Eg).
3. Contacts placed in the **absorber** field are rejected with a role hint.
4. A blocked path shows a red-tinted banner and **does not** populate Absorber–ETL Type, Absorber–HTL Type, or Eg stats.

---

## How to run

### Web UI (preferred for checklist)

1. From the project root (venv activated): `python app.py`
2. Open [http://127.0.0.1:7860](http://127.0.0.1:7860)
3. Enter Absorber / ETL / HTL as listed; click **Predict Type**
4. Mark **Pass** only if the expected blocked UI appears (see cues below)

HTML `required` on the three fields may stop a fully blank submit in the browser. Use whitespace-only strings, or the CLI / Python API, for empty-field server checks.

### CLI

```powershell
cd C:\Users\vidit\Desktop\btp
.\.venv\Scripts\Activate.ps1
python scripts/predict_stack.py --absorber CdTe --etl TiO2 --htl NiO
```

Inspect printed JSON: expect `"blocked": true`, `"screening_blocked": true`, and **no** invented `absorber_etl_type` / Eg fields on the blocked path.

### Python one-liner

```powershell
python -c "import sys; sys.path.insert(0,'scripts'); from predict_stack import predict_stack; r=predict_stack('CdTe','TiO2','NiO',use_llm=False); print(r.get('blocked'), r.get('method'), r.get('message'))"
```

---

## Expected UI cues (blocked path)

| Cue | What you should see |
|-----|---------------------|
| Banner | Red-tinted panel: **Perovskite screening** or **Contact screening** |
| Pills | `blocked` plus `not perovskite` **or** `invalid contact` |
| Message | Contains phrases such as *not a perovskite absorber*, *screening is for perovskite stacks only*, *not a recognized ETL/HTL material*, or *is required* |
| Hint (optional) | e.g. *Try entering … in the ETL/HTL field*, or *Garbage or non-chemistry text is blocked…* |
| Types / Eg | The Type/Eg stat grid is **hidden** when `result.blocked` — no Type I/II/III and no Eg invented for the blocked path |
| Method (CLI/JSON) | `blocked_non_perovskite` or `blocked_invalid_contact` |

**Pass rule:** blocked banner + correct pill + no Type/Eg grid.  
**Fail rule:** Types or Eg appear for a case that must be blocked, or the tool accepts garbage as a “prediction.”

---

## Priority legend

| Priority | Meaning |
|----------|---------|
| **P0** | Must block for a credible demo / viva (non-perovskites, empty contacts, clear garbage) |
| **P1** | Important edge / taxonomy cases (misplaced contacts, precursors, residual formula-parse gaps) |

---

## Status of contact-garbage validation

| Item | Status |
|------|--------|
| `validate_contact_material` / `blocked_invalid_contact` | **Present** on current work (merged via `fix/reject-garbage-etl-htl`) — **does not** depend on an unmerged fix |
| Strings with **no** parseable element symbols (`xyz`, `asdf`, `!!`, `123`) | **Blocked** today |
| Strings that **look like** formulas because random letters map to element symbols (`ABCD` → B,C; `HELLO` → H,O) | **Blocked** today — full-string `_tokenises_as_written` + metal/anion plausibility (no silent drop of non-elements) |

---

## A. Empty / missing inputs

| ID | Absorber | ETL | HTL | Pri | Expected result | Notes | Pass? |
|----|----------|-----|-----|-----|-----------------|-------|-------|
| E01 | *(empty)* | TiO2 | NiO | P0 | Blocked — not perovskite | Message: does not match perovskite absorber families… Browser may refuse submit (`required`); use CLI or whitespace | ☐ |
| E02 | `   ` (spaces) | TiO2 | NiO | P0 | Blocked — not perovskite | Whitespace strips to empty absorber | ☐ |
| E03 | MAPbI3 | *(empty)* | NiO | P0 | Blocked — invalid contact | Message: *ETL is required…* | ☐ |
| E04 | MAPbI3 | TiO2 | *(empty)* | P0 | Blocked — invalid contact | Message: *HTL is required…* | ☐ |
| E05 | MAPbI3 | `   ` | NiO | P0 | Blocked — invalid contact | Whitespace-only ETL treated as empty | ☐ |
| E06 | MAPbI3 | TiO2 | `   ` | P0 | Blocked — invalid contact | Whitespace-only HTL treated as empty | ☐ |

---

## B. Non-perovskite absorbers

Canonical blocks from `DENYLIST_ABSORBERS` / `NON_PEROVSKITE_ABSORBERS`. Thin-film / III–V cases should mention intentional perovskite-only screening.

| ID | Absorber | ETL | HTL | Pri | Expected result | Notes | Pass? |
|----|----------|-----|-----|-----|-----------------|-------|-------|
| N01 | CdTe | TiO2 | NiO | P0 | Blocked — not perovskite | Thin-film chalcogenide; extra thin-film wording in message | ☐ |
| N02 | CZTS | TiO2 | NiO | P0 | Blocked — not perovskite | Same family of blocks | ☐ |
| N03 | CIGS | TiO2 | NiO | P0 | Blocked — not perovskite | Same | ☐ |
| N04 | Si | TiO2 | NiO | P0 | Blocked — not perovskite | Elemental semiconductor | ☐ |
| N05 | GaAs | TiO2 | NiO | P0 | Blocked — not perovskite | III–V | ☐ |
| N06 | Ge | TiO2 | NiO | P0 | Blocked — not perovskite | Elemental | ☐ |
| N07 | InP | TiO2 | NiO | P0 | Blocked — not perovskite | III–V | ☐ |
| N08 | BeSiP2 | TiO2 | NiO | P0 | Blocked — not perovskite | Non-perovskite semiconductor | ☐ |
| N09 | CdSe | TiO2 | NiO | P0 | Blocked — not perovskite | In denylist | ☐ |
| N10 | PbI2 | TiO2 | NiO | P0 | Blocked — not perovskite | Precursor, not ABX₃ absorber | ☐ |
| N11 | SnI2 | TiO2 | NiO | P1 | Blocked — not perovskite | Halide precursor denylist | ☐ |
| N12 | GaN | TiO2 | NiO | P1 | Blocked — not perovskite | In `perovskite_rules.NON_PEROVSKITE_ABSORBERS` | ☐ |

---

## C. Garbage / non-chemistry absorbers

| ID | Absorber | ETL | HTL | Pri | Expected result | Notes | Pass? |
|----|----------|-----|-----|-----|-----------------|-------|-------|
| G01 | xyz | TiO2 | NiO | P0 | Blocked — not perovskite | No perovskite family match | ☐ |
| G02 | asdf | TiO2 | NiO | P0 | Blocked — not perovskite | Random text | ☐ |
| G03 | !! | TiO2 | NiO | P0 | Blocked — not perovskite | Punctuation only | ☐ |
| G04 | 123 | TiO2 | NiO | P0 | Blocked — not perovskite | Digits only | ☐ |
| G05 | Cu | TiO2 | NiO | P0 | Blocked — not perovskite | Random metal; not ABX₃ | ☐ |
| G06 | Au | TiO2 | NiO | P1 | Blocked — not perovskite | Electrode metal as absorber | ☐ |
| G07 | Al | TiO2 | NiO | P1 | Blocked — not perovskite | Same | ☐ |
| G08 | qwerty | TiO2 | NiO | P1 | Blocked — not perovskite | Keyboard mash | ☐ |

---

## D. Garbage ETL (valid perovskite absorber)

Valid absorber: **MAPbI3** (or **CsPbI3** / **FAPbI3** if preferred). Letter-soup strings that partially match element symbols (`ABCD`, `HELLO`) are blocked by full-token validation.

| ID | Absorber | ETL | HTL | Pri | Expected result | Notes | Pass? |
|----|----------|-----|-----|-----|-----------------|-------|-------|
| TE01 | MAPbI3 | xyz | NiO | P0 | Blocked — invalid contact | Message mentions unrecognized ETL; hint about garbage | ☐ |
| TE02 | MAPbI3 | asdf | NiO | P0 | Blocked — invalid contact | Same | ☐ |
| TE03 | MAPbI3 | !! | NiO | P0 | Blocked — invalid contact | Punctuation | ☐ |
| TE04 | MAPbI3 | 123 | NiO | P0 | Blocked — invalid contact | Digits | ☐ |
| TE05 | MAPbI3 | @@@ | NiO | P1 | Blocked — invalid contact | Special characters | ☐ |
| TE06 | MAPbI3 | ABCD | NiO | P1 | Blocked — invalid contact | Full-token check: A and D are not elements → reject (no silent drop) | ☐ |
| TE07 | MAPbI3 | HELLO | NiO | P1 | Blocked — invalid contact | Full-token check: E and L are not elements → reject | ☐ |
| TE08 | MAPbI3 | foobar | NiO | P1 | Blocked — invalid contact | No real-element parse | ☐ |

---

## E. Garbage HTL (valid perovskite absorber)

| ID | Absorber | ETL | HTL | Pri | Expected result | Notes | Pass? |
|----|----------|-----|-----|-----|-----------------|-------|-------|
| TH01 | MAPbI3 | TiO2 | xyz | P0 | Blocked — invalid contact | Unrecognized HTL | ☐ |
| TH02 | MAPbI3 | TiO2 | asdf | P0 | Blocked — invalid contact | Same | ☐ |
| TH03 | MAPbI3 | TiO2 | !! | P0 | Blocked — invalid contact | Punctuation | ☐ |
| TH04 | MAPbI3 | TiO2 | 123 | P0 | Blocked — invalid contact | Digits | ☐ |
| TH05 | MAPbI3 | TiO2 | $$$ | P1 | Blocked — invalid contact | Special characters | ☐ |
| TH06 | MAPbI3 | TiO2 | ABCD | P1 | Blocked — invalid contact | Full-token check (same as TE06) | ☐ |
| TH07 | MAPbI3 | TiO2 | HELLO | P1 | Blocked — invalid contact | Full-token check (same as TE07) | ☐ |
| TH08 | MAPbI3 | TiO2 | lorem | P1 | Blocked — invalid contact | Nonsense word | ☐ |

---

## F. Edge cases — whitespace & special characters

| ID | Absorber | ETL | HTL | Pri | Expected result | Notes | Pass? |
|----|----------|-----|-----|-----|-----------------|-------|-------|
| X01 | ` CdTe ` | TiO2 | NiO | P0 | Blocked — not perovskite | Leading/trailing spaces; still CdTe after normalize | ☐ |
| X02 | MAPbI3 | ` xyz ` | NiO | P0 | Blocked — invalid contact | Spaces around garbage ETL | ☐ |
| X03 | MAPbI3 | TiO2 | ` xyz ` | P0 | Blocked — invalid contact | Spaces around garbage HTL | ☐ |
| X04 | `\t` / NBSP-only absorber | TiO2 | NiO | P1 | Blocked — not perovskite | Odd whitespace; may need paste into field or API | ☐ |
| X05 | MAPbI3 | `...` | NiO | P1 | Blocked — invalid contact | Ellipsis / dots only | ☐ |

---

## G. Misplaced contact-as-absorber

These must fail as absorbers; message should identify ETL/HTL role and preferably hint to move the material.

| ID | Absorber | ETL | HTL | Pri | Expected result | Notes | Pass? |
|----|----------|-----|-----|-----|-----------------|-------|-------|
| M01 | TiO2 | SnO2 | NiO | P0 | Blocked — not perovskite | Pill: not perovskite; hint → ETL field | ☐ |
| M02 | ZnO | TiO2 | NiO | P0 | Blocked — not perovskite | ETL contact as absorber | ☐ |
| M03 | PCBM | TiO2 | NiO | P0 | Blocked — not perovskite | Organic ETL as absorber | ☐ |
| M04 | NiO | TiO2 | Spiro-OMeTAD | P0 | Blocked — not perovskite | Hint → HTL field | ☐ |
| M05 | MoO3 | TiO2 | NiO | P0 | Blocked — not perovskite | Deep-affinity HTL oxide | ☐ |
| M06 | Spiro-OMeTAD | TiO2 | NiO | P0 | Blocked — not perovskite | Organic HTL as absorber | ☐ |
| M07 | SnO2 | TiO2 | NiO | P1 | Blocked — not perovskite | Common ETL mis-entry | ☐ |
| M08 | PTAA | TiO2 | NiO | P1 | Blocked — not perovskite | Polymer HTL | ☐ |

---

## Smoke positive (sanity — should **not** block)

Run once so a broken deploy is not mistaken for “everything blocked.”

| ID | Absorber | ETL | HTL | Expected | Pass? |
|----|----------|-----|-----|----------|-------|
| P01 | MAPbI3 | TiO2 | NiO | Types + Eg shown; **not** blocked | ☐ |
| P02 | Cs2AgBiBr6 | SnO2 | Spiro-OMeTAD | Types + Eg (or estimates); **not** blocked | ☐ |

---

## Execution log (optional)

| Date | Tester | Branch / commit | Env | Notes |
|------|--------|-----------------|-----|-------|
| | | | e.g. Windows + Python 3.x + venv | |

---

## Implementation map (for reviewers)

| Gate | Function / module | Typical `method` |
|------|-------------------|------------------|
| Absorber taxonomy | `check_absorber_perovskite` ← `perovskite_rules.classify_family` | `blocked_non_perovskite` |
| Contact empty / garbage | `validate_contact_material` | `blocked_invalid_contact` |
| UI | `app.py` — `result.blocked` banner; Type/Eg grid only if `not result.blocked` | — |

Prediction order in `predict_stack`: perovskite check → contact check → literature → lookup / ML.
