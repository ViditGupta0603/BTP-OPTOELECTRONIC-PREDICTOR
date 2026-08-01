"""Verify unicode formula spellings resolve identically to their ASCII form.

Checks that the canonical fold in formula_parse.normalize_formula_text reaches the
eligibility gate, the formula parser, the alias table and the layer lookup, so
CH₃NH₃PbI₃ / ＣＨ３ＮＨ３ＰｂＩ３ / CH3NH3PbI3 / MAPbI3 cannot diverge.

Run:
  python scripts/verify_unicode_normalization.py
Writes data/unicode_normalization_verification.md
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formula_parse import parse_formula_counts  # noqa: E402
from perovskite_rules import classify_family, looks_like_perovskite_absorber  # noqa: E402
from predict_stack import normalize_material_name, predict_stack  # noqa: E402

ETL = "TiO2"
HTL = "MoO3"

# Equivalence groups: every spelling must give the same Eg / family / verdict.
EQUIVALENT_GROUPS: list[tuple[str, list[str], float | None]] = [
    ("MAPbI3", ["CH₃NH₃PbI₃", "CH3NH3PbI3", "MAPbI3", "MAPbI₃", "ＣＨ３ＮＨ３ＰｂＩ３"], 1.55),
    ("MAPbBr3", ["CH₃NH₃PbBr₃", "CH3NH3PbBr3", "MAPbBr3", "MAPbBr₃"], 2.32),
    ("FAPbI3", ["FAPbI₃", "FAPbI3", "HC(NH₂)₂PbI₃"], 1.48),
    ("FAPbBr3", ["FAPbBr₃", "FAPbBr3"], 2.23),
    ("Cs2TiI6", ["Cs₂TiI₆", "Cs2TiI6"], 1.58),
    ("Cs3Sb2I9", ["Cs₃Sb₂I₉", "Cs3Sb2I9"], None),
    ("Cs2AgBiBr6", ["Cs₂AgBiBr₆", "Cs2AgBiBr6"], None),
]

# Halide / A-site identity must not collapse across these pairs.
DISTINCT_PAIRS = [
    ("MAPbI3", "MAPbBr3"),
    ("FAPbI3", "FAPbBr3"),
    ("MAPbI3", "FAPbI3"),
]

# Must stay blocked in every spelling.
BLOCKED_GROUPS: list[list[str]] = [
    ["ZnO", "ＺｎＯ"],
    ["CZTS", "ＣＺＴＳ"],
    ["PbI2", "PbI₂", "ＰｂＩ２"],
    ["1T-PbI2", "1T‑PbI₂"],
]


def _run(name: str) -> dict:
    r = predict_stack(name, ETL, HTL, use_llm=False)
    return {
        "input": name,
        "normalized": normalize_material_name(name),
        "eg": r.get("absorber_band_gap_eV"),
        "family": classify_family(name).family_id,
        "perovskite": looks_like_perovskite_absorber(name),
        "counts": parse_formula_counts(name),
        "etl_type": r.get("absorber_etl_type"),
        "htl_type": r.get("absorber_htl_type"),
        "verdict": (r.get("optoelectronic") or {}).get("verdict"),
        "blocked": bool(r.get("blocked")),
    }


def main() -> int:
    lines = [
        "# Unicode formula normalization verification",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        "**Mode:** `predict_stack(..., use_llm=False)`  ",
        f"**Contacts:** ETL `{ETL}` / HTL `{HTL}`",
        "",
        "## Equivalent spellings",
        "",
        "| Group | Input | Normalized | Eg (eV) | Family | Types | Verdict | Result |",
        "|-------|-------|------------|---------|--------|-------|---------|--------|",
    ]
    failures: list[str] = []
    eg_by_group: dict[str, float | None] = {}

    for group, spellings, expected_eg in EQUIVALENT_GROUPS:
        runs = [_run(s) for s in spellings]
        ref = runs[0]
        eg_by_group[group] = ref["eg"]
        for run in runs:
            same = (
                run["eg"] == ref["eg"]
                and run["family"] == ref["family"]
                and run["verdict"] == ref["verdict"]
                and run["etl_type"] == ref["etl_type"]
                and run["htl_type"] == ref["htl_type"]
                and not run["blocked"]
            )
            if expected_eg is not None and (
                run["eg"] is None or abs(float(run["eg"]) - expected_eg) > 0.05
            ):
                same = False
                failures.append(
                    f"{group}: `{run['input']}` Eg={run['eg']} != expected {expected_eg}"
                )
            elif not same:
                failures.append(
                    f"{group}: `{run['input']}` diverges from `{ref['input']}` "
                    f"(Eg {run['eg']} vs {ref['eg']}, family {run['family']} vs {ref['family']})"
                )
            lines.append(
                f"| {group} | `{run['input']}` | `{run['normalized']}` | "
                f"{run['eg']} | {run['family']} | {run['etl_type']}/{run['htl_type']} | "
                f"{run['verdict']} | **{'PASS' if same else 'FAIL'}** |"
            )

    lines += ["", "## Halide / A-site identity kept distinct", "", "| A | B | Eg A | Eg B | Result |", "|---|---|------|------|--------|"]
    for a, b in DISTINCT_PAIRS:
        ega, egb = eg_by_group.get(a), eg_by_group.get(b)
        ok = ega is not None and egb is not None and abs(float(ega) - float(egb)) > 0.05
        if not ok:
            failures.append(f"{a} and {b} collapsed to the same Eg ({ega} / {egb})")
        lines.append(f"| {a} | {b} | {ega} | {egb} | **{'PASS' if ok else 'FAIL'}** |")

    lines += [
        "",
        "## Blocked cases (all spellings)",
        "",
        "| Input | Normalized | Blocked | Result |",
        "|-------|------------|---------|--------|",
    ]
    for spellings in BLOCKED_GROUPS:
        for s in spellings:
            run = _run(s)
            ok = run["blocked"]
            if not ok:
                failures.append(f"`{s}` should be blocked but returned Eg={run['eg']}")
            lines.append(
                f"| `{s}` | `{run['normalized']}` | {run['blocked']} | "
                f"**{'PASS' if ok else 'FAIL'}** |"
            )

    lines += ["", "## Verdict", ""]
    if failures:
        lines.append(f"**{len(failures)} failure(s):**")
        lines.append("")
        lines += [f"- {f}" for f in failures]
    else:
        lines.append("**All unicode spellings resolve identically to their ASCII form.**")
    lines.append("")

    out = ROOT / "data" / "unicode_normalization_verification.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"FAILURES={len(failures)}")
    for f in failures:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
