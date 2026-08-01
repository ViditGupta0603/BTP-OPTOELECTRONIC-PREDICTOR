"""Verify Anderson Type I / II / III classification against literature ground truth.

Guards the regression where broken-gap (Type III) alignment was unreachable: the
estimator clamped electron affinity to [1.3, 4.9] eV while a perovskite absorber
sits at chi + Eg = 5.2-5.6 eV, so the broken-gap condition chi_contact >=
chi_absorber + Eg_absorber could never be satisfied and every stack collapsed
onto Type I / Type II.

Two layers of checks:
  1. junction_type() against hand-computed band edges for textbook heterojunctions.
  2. predict_stack() end to end, so lookup and estimator must also supply band
     edges wide enough to express the answer.

Run:
  python scripts/verify_type_iii_alignment.py
Writes data/type_iii_alignment_verification.md
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formula_estimator import CHI_PHYSICAL_RANGE  # noqa: E402
from literature_bands import Layer, junction_type  # noqa: E402
from predict_stack import load_layer_lookup, predict_stack, resolve_layer  # noqa: E402

# (label, Layer A, Layer B, expected Type). Edges are vacuum referenced:
# CBM = -chi, VBM = -(chi + Eg).
EDGE_CASES: list[tuple[str, Layer, Layer, str]] = [
    (
        "InAs/GaSb — textbook broken gap (GaSb VBM above InAs CBM)",
        Layer("InAs", 0.35, 4.90),
        Layer("GaSb", 0.73, 4.06),
        "Type III",
    ),
    (
        "MAPbI3/MoO3 — deep-affinity oxide CBM below absorber VBM",
        Layer("MAPbI3", 1.55, 3.90),
        Layer("MoO3", 3.00, 6.70),
        "Type III",
    ),
    (
        "CsPbI3/V2O5 — deep-affinity oxide",
        Layer("CsPbI3", 1.73, 3.95),
        Layer("V2O5", 2.80, 6.60),
        "Type III",
    ),
    (
        "GaAs/AlGaAs — straddling",
        Layer("GaAs", 1.42, 4.07),
        Layer("AlGaAs", 1.80, 3.74),
        "Type I",
    ),
    (
        "MAPbI3/MgO — wide-gap insulator straddles absorber gap",
        Layer("MAPbI3", 1.55, 3.90),
        Layer("MgO", 7.80, 0.85),
        "Type I",
    ),
    (
        "MAPbI3/TiO2 — staggered",
        Layer("MAPbI3", 1.55, 3.90),
        Layer("TiO2", 3.20, 4.00),
        "Type II",
    ),
    (
        "Zero-overlap boundary (absorber VBM exactly at partner CBM)",
        Layer("A", 1.00, 4.00),
        Layer("B", 1.00, 5.00),
        "Type III",
    ),
]

# (absorber, ETL, HTL, expected ETL Type or None, expected HTL Type or None)
STACK_CASES: list[tuple[str, str, str, str | None, str | None]] = [
    ("MAPbI3", "TiO2", "MoO3", "Type II", "Type III"),
    ("CsPbI3", "SnO2", "V2O5", None, "Type III"),
    ("Cs2AgBiBr6", "ZnO", "MoO3", None, "Type III"),
    # User-reported stack, including the FA casing that used to parse as an
    # invented element "Fa".
    ("FaSnI3", "MoO3", "MgO", "Type III", "Type I"),
    ("FASnI3", "MoO3", "MgO", "Type III", "Type I"),
    # Must stay as they were — the fix may not turn ordinary stacks into Type III.
    ("MAPbI3", "TiO2", "Spiro-OMeTAD", "Type II", "Type II"),
    ("K2TiI6", "WS2", "NiO", "Type I", None),
    ("CsSnI3", "TiO2", "NiO", "Type II", "Type I"),
    ("FAPbI3", "TiO2", "CuSCN", "Type II", None),
]

# Contacts whose band edges must stay physical for broken gap to be expressible.
CONTACT_BANDS: dict[str, tuple[float, float]] = {
    "MoO3": (3.00, 6.70),
    "V2O5": (2.80, 6.60),
    "WO3": (3.10, 5.00),
    "MgO": (7.80, 0.85),
    "Al2O3": (8.80, 1.35),
}


def main() -> int:
    failures: list[str] = []
    lines = [
        "# Anderson Type I / II / III verification",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        "**Convention:** `CBM = -χ`, `VBM = -(χ + Eg)` (eV vs vacuum)  ",
        "**Type III test:** `VBM_a ≥ CBM_b or VBM_b ≥ CBM_a` (gaps do not overlap)",
        "",
        "## Direct `junction_type` on hand-computed edges",
        "",
        "| Case | CBM/VBM A | CBM/VBM B | Got | Expect | Result |",
        "|---|---|---|---|---|---|",
    ]

    for label, a, b, expect in EDGE_CASES:
        got = junction_type(a, b)
        ok = got == expect
        if not ok:
            failures.append(f"{label}: got {got}, expected {expect}")
        lines.append(
            f"| {label} | {a.cbm:+.2f}/{a.vbm:+.2f} | {b.cbm:+.2f}/{b.vbm:+.2f} | "
            f"`{got}` | `{expect}` | **{'PASS' if ok else 'FAIL'}** |"
        )

    lines += [
        "",
        "## Contact band edges available to the pipeline",
        "",
        "| Material | Eg (eV) | χ (eV) | Expected | Result |",
        "|---|---:|---:|---|---|",
    ]
    layers = load_layer_lookup()
    for mat, (eg_x, chi_x) in CONTACT_BANDS.items():
        entry = resolve_layer(layers, mat) or {}
        eg, chi = entry.get("Eg_eV"), entry.get("chi_eV")
        ok = (
            eg is not None
            and chi is not None
            and abs(float(eg) - eg_x) <= 0.05
            and abs(float(chi) - chi_x) <= 0.05
        )
        if not ok:
            failures.append(f"{mat}: Eg={eg} χ={chi}, expected Eg={eg_x} χ={chi_x}")
        lines.append(
            f"| {mat} | {eg} | {chi} | Eg={eg_x} χ={chi_x} | "
            f"**{'PASS' if ok else 'FAIL'}** |"
        )

    lines += [
        "",
        "## End-to-end `predict_stack`",
        "",
        "| Stack | ETL Type | HTL Type | Verdict | Result |",
        "|---|---|---|---|---|",
    ]
    types_seen: set[str] = set()
    for a, e, h, expect_etl, expect_htl in STACK_CASES:
        r = predict_stack(a, e, h)
        etl_t = r.get("absorber_etl_type") or r.get("predicted_absorber_etl_type")
        htl_t = r.get("absorber_htl_type") or r.get("predicted_absorber_htl_type")
        types_seen.update(str(t) for t in (etl_t, htl_t) if t)
        ok = (expect_etl is None or etl_t == expect_etl) and (
            expect_htl is None or htl_t == expect_htl
        )
        if not ok:
            failures.append(
                f"{a}/{e}/{h}: got {etl_t}/{htl_t}, "
                f"expected {expect_etl or 'any'}/{expect_htl or 'any'}"
            )
        lines.append(
            f"| {a}/{e}/{h} | {etl_t} | {htl_t} | "
            f"{(r.get('optoelectronic') or {}).get('verdict')} | "
            f"**{'PASS' if ok else 'FAIL'}** |"
        )

    all_three = {"Type I", "Type II", "Type III"} <= types_seen
    if not all_three:
        failures.append(f"stacks only produced {sorted(types_seen)}; need I, II and III")

    chi_span_ok = CHI_PHYSICAL_RANGE[1] >= 6.7
    if not chi_span_ok:
        failures.append(
            f"χ clamp upper bound {CHI_PHYSICAL_RANGE[1]} cannot express a "
            "deep-affinity oxide (MoO3 χ=6.7 eV)"
        )

    summary = {
        "types_seen": sorted(types_seen),
        "all_three_types_ok": all_three,
        "chi_clamp": list(CHI_PHYSICAL_RANGE),
        "chi_clamp_admits_deep_affinity": chi_span_ok,
        "n_failures": len(failures),
    }
    lines += ["", "## Summary", "", "```json", json.dumps(summary, indent=2), "```", ""]
    if failures:
        lines += [f"- FAIL {f}" for f in failures] + [""]

    out = ROOT / "data" / "type_iii_alignment_verification.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps(summary, indent=2))
    for f in failures:
        print(f"  FAIL {f}")
    print(f"FAILURES={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
