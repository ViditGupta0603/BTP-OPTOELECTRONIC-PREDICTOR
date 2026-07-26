"""Round-2 verification harness for Type / halide / PEDOT fixes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from literature_bands import Layer, junction_type
from predict_stack import load_layer_lookup, normalize_material_name, predict_stack, resolve_layer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "round2_fix_verification.md"


def main() -> None:
    layers = load_layer_lookup()
    lines: list[str] = []
    lines.append("# Round 2 fix verification\n")
    lines.append("Branch: `fix/round2-type-halide-pedot`\n")

    # --- Halide-sensitive Eg ---
    lines.append("## P0 — Halide-sensitive Eg lookup\n")
    halide_ok = True
    for label, expect_lo, expect_hi in (
        ("FAPbI3", 1.45, 1.55),
        ("FAPbBr3", 2.15, 2.35),
        ("FAPbBr\u2083", 2.15, 2.35),
        ("MAPbI3", 1.50, 1.60),
        ("MAPbBr3", 2.20, 2.40),
        ("MAPbBr\u2083", 2.20, 2.40),
    ):
        entry = resolve_layer(layers, label)
        eg = entry.get("Eg_eV") if entry else None
        ok = eg is not None and expect_lo <= float(eg) <= expect_hi
        halide_ok = halide_ok and ok
        lines.append(
            f"- `{label.encode('unicode_escape').decode()}` → "
            f"`{normalize_material_name(label)}` Eg={eg} "
            f"(expect {expect_lo}–{expect_hi}): **{'PASS' if ok else 'FAIL'}**"
        )
    lines.append("")

    # --- Anderson Type variety ---
    lines.append("## P0 — Junction Types vary (not all Type I)\n")
    stacks = [
        ("MAPbI3", "TiO2", "Spiro-OMeTAD"),
        ("Cs2SnI6", "TiO2", "P3HT"),
        ("K2TiI6", "PC60BM", "MoO3"),
        ("K2TiI6", "WS2", "NiO"),
        ("CsSnI3", "TiO2", "PEDOT:PSS"),
        ("CsPbBr3", "ZnO", "NiO"),
        ("FAPbBr3", "TiO2", "Spiro-OMeTAD"),
    ]
    types_seen: set[str] = set()
    lines.append("| Stack | Eg | ETL Type | HTL Type | Verdict | Notes |")
    lines.append("|---|---:|---|---|---|---|")
    for a, e, h in stacks:
        r = predict_stack(a, e, h)
        etl_t = r.get("absorber_etl_type") or r.get("predicted_absorber_etl_type")
        htl_t = r.get("absorber_htl_type") or r.get("predicted_absorber_htl_type")
        if etl_t:
            types_seen.add(str(etl_t))
        if htl_t:
            types_seen.add(str(htl_t))
        opto = r.get("optoelectronic") or {}
        note = "; ".join((r.get("notes") or [])[:2]) or "—"
        if opto.get("htl_caveat"):
            note = f"PEDOT caveat; {note}"
        lines.append(
            f"| {a}/{e}/{h} | {r.get('absorber_band_gap_eV')} | {etl_t} | {htl_t} | "
            f"{opto.get('verdict')} | {note} |"
        )
    vary_ok = len(types_seen) >= 2 and "Type I" in types_seen
    # Prefer seeing Type II as well after Anderson fix
    lines.append("")
    lines.append(f"Types observed: `{sorted(types_seen)}` — **{'PASS' if vary_ok else 'FAIL'}** (must not be only Type I).\n")

    # Manual Anderson sanity
    lines.append("## Anderson sanity (direct `junction_type`)\n")
    cases = [
        ("narrow-in-wide (Type I)", Layer("n", 1.5, 3.9), Layer("w", 3.2, 3.5), "Type I"),
        ("MAPbI3/TiO2 staggered (Type II)", Layer("MAPbI3", 1.55, 3.9), Layer("TiO2", 3.2, 4.0), "Type II"),
        ("broken gap (Type III)", Layer("A", 1.0, 5.5), Layer("B", 1.5, 3.5), "Type III"),
    ]
    for label, a, b, expect in cases:
        got = junction_type(a, b)
        lines.append(f"- {label}: got `{got}` expect `{expect}` — **{'PASS' if got == expect else 'FAIL'}**")
    lines.append("")

    # PEDOT caveat
    lines.append("## P1 — PEDOT:PSS degenerate HTL caveat\n")
    ped = predict_stack("CsSnI3", "TiO2", "PEDOT:PSS")
    opto = ped.get("optoelectronic") or {}
    notes = " ".join(ped.get("notes") or [])
    ped_ok = (
        opto.get("htl_caveat") == "degenerate_metallic"
        or "degenerate" in notes.lower()
        or "degenerate" in (opto.get("reason") or "").lower()
    )
    lines.append(
        f"- CsSnI3/TiO2/PEDOT:PSS caveat present: **{'PASS' if ped_ok else 'FAIL'}** "
        f"(verdict={opto.get('verdict')}, chi_CsSnI3={ped.get('absorber_chi_eV')})"
    )
    lines.append("")

    # χ distinctness sample
    lines.append("## χ sample (absorber vs contacts)\n")
    for m in ("MAPbI3", "FAPbBr3", "CsSnI3", "TiO2", "Spiro-OMeTAD", "PEDOT:PSS", "NiO", "ZnO"):
        e = resolve_layer(layers, m)
        lines.append(f"- `{m}`: {e}")
    lines.append("")

    summary = {
        "halide_ok": halide_ok,
        "types_vary": vary_ok,
        "types_seen": sorted(types_seen),
        "pedot_caveat": ped_ok,
    }
    lines.append("## Summary\n")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2))
    lines.append("```\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    print(json.dumps(summary, indent=2))
    if not (halide_ok and vary_ok and ped_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
