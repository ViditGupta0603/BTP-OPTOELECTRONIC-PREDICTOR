"""Round-3 verification: A2BX6 halide identity, Type I/III coverage, FAPbCl3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from literature_bands import Layer, junction_type
from predict_stack import load_layer_lookup, normalize_material_name, predict_stack, resolve_layer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "round3_fix_verification.md"


def _eg(layers: dict, name: str) -> float | None:
    e = resolve_layer(layers, name)
    return float(e["Eg_eV"]) if e and e.get("Eg_eV") is not None else None


def check_monotonic(layers: dict, series: list[tuple[str, str, str]]) -> list[dict]:
    """series: list of (I_formula, Br_formula, Cl_formula)."""
    rows = []
    for i_f, br_f, cl_f in series:
        eg_i, eg_br, eg_cl = _eg(layers, i_f), _eg(layers, br_f), _eg(layers, cl_f)
        ok = (
            eg_i is not None
            and eg_br is not None
            and eg_cl is not None
            and eg_i < eg_br < eg_cl
        )
        rows.append(
            {
                "series": f"{i_f} / {br_f} / {cl_f}",
                "Eg_I": eg_i,
                "Eg_Br": eg_br,
                "Eg_Cl": eg_cl,
                "ok": ok,
            }
        )
    return rows


def main() -> None:
    layers = load_layer_lookup()
    lines: list[str] = []
    lines.append("# Round 3 fix verification\n")
    lines.append("Branch: `fix/round3-a2bx6-halide-types`\n")

    # --- P1: Cs2TiI6 vs Br6 before/after narrative ---
    lines.append("## P1 — A2BX6 halide identity (Cs2TiX6)\n")
    eg_i = _eg(layers, "Cs2TiI6")
    eg_br = _eg(layers, "Cs2TiBr6")
    eg_cl = _eg(layers, "Cs2TiCl6")
    i_ok = eg_i is not None and 1.56 <= eg_i <= 1.65
    br_ok = eg_br is not None and abs(eg_br - 1.88) < 0.05 and (eg_i is None or eg_br > eg_i + 0.15)
    lines.append(
        f"- **Before (Round 2 / eval):** Cs2TiI6 Eg=**1.800** eV (collided with Br≈1.8)\n"
        f"- **After:** Cs2TiI6 Eg=**{eg_i}** eV (expect 1.56–1.65): "
        f"**{'PASS' if i_ok else 'FAIL'}**\n"
        f"- **After:** Cs2TiBr6 Eg=**{eg_br}** eV (expect ≈1.88, distinct): "
        f"**{'PASS' if br_ok else 'FAIL'}**\n"
        f"- **After:** Cs2TiCl6 Eg=**{eg_cl}** eV\n"
    )

    # Exact resolve — no Cs2Ti- prefix bleed
    lines.append("## P1 — Exact-match resolve (no prefix)\n")
    prefix_ok = True
    for label in ("Cs2TiI6", "Cs2TiBr6", "Cs2TiCl6"):
        got = resolve_layer(layers, label)
        # Must not return a sibling halide
        siblings = {k: _eg(layers, k) for k in ("Cs2TiI6", "Cs2TiBr6", "Cs2TiCl6") if k != label}
        eg = float(got["Eg_eV"]) if got else None
        collide = eg is not None and any(abs(eg - (s or -99)) < 1e-9 for s in siblings.values())
        # Distinct values required between I and Br
        ok = got is not None and not (
            label == "Cs2TiI6" and eg_br is not None and eg is not None and abs(eg - eg_br) < 0.05
        )
        prefix_ok = prefix_ok and ok and not collide
        lines.append(
            f"- `{label}` → Eg={eg} (norm=`{normalize_material_name(label)}`): "
            f"**{'PASS' if ok else 'FAIL'}**"
        )
    # Explicit: querying Cs2TiI6 must not equal Cs2TiBr6 entry
    distinct = eg_i is not None and eg_br is not None and abs(eg_i - eg_br) > 0.1
    lines.append(
        f"- Cs2TiI6 ≠ Cs2TiBr6 (|ΔEg|={abs((eg_i or 0)-(eg_br or 0)):.3f}): "
        f"**{'PASS' if distinct else 'FAIL'}**\n"
    )
    prefix_ok = prefix_ok and distinct

    # Monotonicity
    lines.append("## P1 — Halide monotonicity Eg(I) < Eg(Br) < Eg(Cl)\n")
    mono_rows = check_monotonic(
        layers,
        [
            ("Cs2TiI6", "Cs2TiBr6", "Cs2TiCl6"),
            ("K2TiI6", "K2TiBr6", "K2TiCl6"),
            ("Rb2TiI6", "Rb2TiBr6", "Rb2TiCl6"),
            ("Cs2SnI6", "Cs2SnBr6", "Cs2SnCl6"),
            ("Cs2PdI6", "Cs2PdBr6", "Cs2PdCl6"),
            ("Cs2PtI6", "Cs2PtBr6", "Cs2PtCl6"),
        ],
    )
    lines.append("| Series | Eg(I) | Eg(Br) | Eg(Cl) | Verdict |")
    lines.append("|---|---:|---:|---:|---|")
    mono_ok = True
    for r in mono_rows:
        mono_ok = mono_ok and r["ok"]
        lines.append(
            f"| {r['series']} | {r['Eg_I']} | {r['Eg_Br']} | {r['Eg_Cl']} | "
            f"**{'PASS' if r['ok'] else 'FAIL'}** |"
        )
    lines.append("")

    # --- P2: Type I / Type III ---
    lines.append("## P2 — Type I and Type III coverage (published CBM/VBM)\n")
    lines.append(
        "Anderson Types from vacuum edges (`CBM=-χ`, `VBM=-(χ+Eg)`). "
        "Typical PSC stacks are Type I/II; Type III (broken gap) appears when "
        "a deep-χ absorber meets a shallow-IP contact.\n"
    )
    stacks = [
        ("Cs2SnI6", "TiO2", "P3HT", "Type I", None),  # ETL straddling expected
        ("Cs2SnI6", "TiO2", "CFTS", None, "Type III"),  # HTL broken gap (IP_CFTS < χ_Cs2SnI6)
        ("MAPbI3", "TiO2", "Spiro-OMeTAD", "Type II", "Type II"),
        ("K2TiI6", "WS2", "NiO", "Type I", None),
        ("MAPbI3", "TiO2", "NiO", None, "Type I"),
    ]
    lines.append("| Stack | Eg | χ | ETL Type | HTL Type | Notes |")
    lines.append("|---|---:|---:|---|---|---|")
    types_seen: set[str] = set()
    type_checks_ok = True
    for a, e, h, expect_etl, expect_htl in stacks:
        r = predict_stack(a, e, h)
        etl_t = r.get("absorber_etl_type") or r.get("predicted_absorber_etl_type")
        htl_t = r.get("absorber_htl_type") or r.get("predicted_absorber_htl_type")
        if etl_t:
            types_seen.add(str(etl_t))
        if htl_t:
            types_seen.add(str(htl_t))
        ok_e = expect_etl is None or etl_t == expect_etl
        ok_h = expect_htl is None or htl_t == expect_htl
        type_checks_ok = type_checks_ok and ok_e and ok_h
        note = []
        if expect_etl:
            note.append(f"ETL expect {expect_etl}: {'OK' if ok_e else 'FAIL'}")
        if expect_htl:
            note.append(f"HTL expect {expect_htl}: {'OK' if ok_h else 'FAIL'}")
        lines.append(
            f"| {a}/{e}/{h} | {r.get('absorber_band_gap_eV')} | {r.get('absorber_chi_eV')} | "
            f"{etl_t} | {htl_t} | {'; '.join(note) or '—'} |"
        )

    # Direct Anderson sanity with published-style edges
    lines.append("\n### Direct `junction_type` (CBM/VBM)\n")
    # Cs2SnI6 / TiO2 Type I; Cs2SnI6 / CFTS Type III
    a_sn = Layer("Cs2SnI6", 1.35, 4.8)
    etl_tio2 = Layer("TiO2", 3.2, 4.0)
    htl_cfts = Layer("CFTS", 1.3, 3.3)
    jt_i = junction_type(a_sn, etl_tio2)
    jt_iii = junction_type(a_sn, htl_cfts)
    lines.append(
        f"- Cs2SnI6/TiO2: `{jt_i}` expect Type I — **{'PASS' if jt_i == 'Type I' else 'FAIL'}**\n"
        f"- Cs2SnI6/CFTS: `{jt_iii}` expect Type III — **{'PASS' if jt_iii == 'Type III' else 'FAIL'}**\n"
        f"- Synthetic broken-gap: `{junction_type(Layer('A', 1.0, 5.5), Layer('B', 1.5, 3.5))}` "
        f"expect Type III\n"
    )
    type_checks_ok = type_checks_ok and jt_i == "Type I" and jt_iii == "Type III"
    types_ok = (
        "Type I" in types_seen
        and "Type II" in types_seen
        and "Type III" in types_seen
        and type_checks_ok
    )
    lines.append(
        f"Types observed in stacks: `{sorted(types_seen)}` — "
        f"**{'PASS' if types_ok else 'FAIL'}** (need I, II, and III).\n"
    )

    # --- P3 FAPbCl3 ---
    lines.append("## P3 — FAPbCl3 toward ~2.9 eV\n")
    eg_cl3 = _eg(layers, "FAPbCl3")
    eg_cl3b = _eg(layers, "HC(NH2)2PbCl3")
    cl_ok = (eg_cl3 is not None and abs(eg_cl3 - 2.90) < 0.05) or (
        eg_cl3b is not None and abs(eg_cl3b - 2.90) < 0.05
    )
    lines.append(
        f"- FAPbCl3 Eg={eg_cl3} / HC(NH2)2PbCl3 Eg={eg_cl3b} (expect ≈2.90): "
        f"**{'PASS' if cl_ok else 'FAIL'}**\n"
    )

    summary = {
        "cs2tii6_eg": eg_i,
        "cs2tibr6_eg": eg_br,
        "cs2ticl6_eg": eg_cl,
        "cs2tii6_in_range": i_ok,
        "cs2tibr6_distinct": br_ok,
        "exact_resolve_ok": prefix_ok,
        "monotonicity_ok": mono_ok,
        "monotonicity": mono_rows,
        "types_seen": sorted(types_seen),
        "types_i_ii_iii_ok": types_ok,
        "fapbcl3_ok": cl_ok,
        "fapbcl3_eg": eg_cl3 or eg_cl3b,
    }
    lines.append("## Summary\n")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2, default=str))
    lines.append("```\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT)
    print(json.dumps(summary, indent=2, default=str))
    if not (i_ok and br_ok and prefix_ok and mono_ok and types_ok and cl_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
