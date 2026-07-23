"""Cross-check unique material Eg/χ against published literature ranges.

Tolerance policy (user): small disagreement OK; TiO2-scale errors (~0.8–1.0 eV) fail.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "material_literature_audit.json"

# Literature / web-verified plausible ranges (Eg_min, Eg_max, chi_min, chi_max, notes)
# Sources: experimental reviews, SCAPS consensus tables, PMC SCAPS papers, Anatase TiO2 ~3.2 eV
LITERATURE_RANGES: dict[str, tuple[float, float, float, float, str]] = {
    # Wide-gap oxides / nitrides ETLs
    "TiO2": (3.0, 3.5, 3.8, 4.3, "anatase/rutile expt ~3.0–3.2; SCAPS often 3.2–3.4; MP~2.3 REJECT"),
    "TiO2:N": (2.9, 3.4, 2.0, 2.6, "N-doped as HTL in Paper4; Eg~3.0 ok"),
    "SnO2": (3.4, 3.8, 3.8, 4.2, "literature ~3.5–3.6 / χ~4.0"),
    "Nb2O5": (3.2, 3.7, 4.1, 4.5, "SCAPS consensus ~3.46 / 4.33"),
    "ZnO": (3.2, 3.5, 4.0, 4.5, "expt ~3.3"),
    "WO3": (2.4, 2.8, 3.6, 4.0, "SCAPS common"),
    "V2O5": (2.0, 2.4, 3.2, 4.1, "SCAPS ~2.2; χ often 3.4–4.0"),
    "MoO3": (2.8, 3.2, 2.1, 2.7, "SCAPS ~3.0 / χ~2.3–2.5"),
    "NiO": (3.4, 4.0, 1.3, 2.3, "SCAPS ~3.6–3.8 / χ~1.46–2.1"),
    "CuAlO2": (3.2, 3.7, 2.2, 2.8, "SCAPS Paper4/CsPb"),
    # Chalcogenides
    "CdS": (2.3, 2.5, 4.0, 4.4, "expt ~2.4 / χ~4.18"),
    "CdZnS": (3.0, 3.4, 4.0, 4.4, "SCAPS Paper4"),
    "ZnSe": (2.6, 2.9, 3.9, 4.3, "SCAPS CsPb"),
    "ZnTe": (2.1, 2.4, 3.5, 4.0, "SCAPS CsPb"),
    "SnS2": (1.7, 2.2, 4.0, 4.5, "SCAPS CsPb ~1.85"),
    "WS2": (1.6, 2.1, 3.7, 4.2, "monolayer/SCAPS ~1.8 / χ~3.95"),
    "MoS2": (1.2, 2.3, 3.9, 4.5, "bulk~1.2–1.3; 2H mono HSE~1.8–2.2"),
    # Organics / polymers
    "PC60BM": (1.7, 2.1, 3.9, 4.4, "SCAPS ~1.8–2.0 / χ~3.9–4.2"),
    "PCBM": (1.7, 2.1, 3.8, 4.3, "same family as PC60BM"),
    "PTAA": (2.7, 3.2, 2.0, 2.6, "SCAPS ~2.96 / χ~2.3"),
    "MEH-PPV": (1.9, 2.3, 2.5, 3.1, "SCAPS CsPb"),
    "CuI": (2.9, 3.3, 1.9, 2.4, "SCAPS ~3.1 / χ~2.1"),
    "CuSCN": (3.4, 3.8, 1.5, 2.0, "SCAPS ~3.6 / χ~1.7"),
    # Absorbers
    "K2TiI6": (1.5, 1.7, 3.8, 4.2, "expt ~1.62; Paper4 1.61/χ4.0"),
    "CsPb0.625Zn0.375IBr2": (1.0, 1.2, 4.1, 4.4, "Paper CsPb Table1"),
    "BeSiP2": (1.2, 1.9, 3.8, 4.4, "SCAPS TierA 1.4/4.2; DFT often ~1.7–1.8"),
    # Paper4 HTLs (paper-internal + SCAPS consensus)
    "MWCNTs": (1.4, 1.7, 3.4, 3.9, "Paper4 Table2"),
    "NiCo2O4": (2.1, 2.5, 3.2, 3.7, "Paper4 Table2"),
    "C6TBTAPH2": (1.4, 1.8, 3.4, 3.8, "Paper4 Table2"),
    "nPB": (2.2, 2.6, 2.8, 3.3, "Paper4 Table2"),
    "C6PcH2": (1.4, 1.8, 3.5, 3.9, "Paper4 Table2"),
    "D-PBTTT-14": (2.0, 2.3, 3.0, 3.4, "Paper4 Table2"),
    "LBSO": (3.0, 3.3, 4.2, 4.6, "Paper4 Table1"),
    "GaAs": (1.3, 1.5, 3.9, 4.2, "bulk GaAs 1.42"),
    "CNTS": (1.5, 1.9, 3.6, 4.1, "CsPb Table2"),
    "Cu2Te": (1.0, 1.4, 4.0, 4.4, "CsPb Table2"),
    "Zn3P2": (1.3, 1.7, 4.0, 4.4, "CsPb Table2"),
}

# Tight fail if outside this absolute Eg window for TiO2 family (the supervisor rejection case)
TIO2_HARD_MIN = 2.9


def collect_unique_materials() -> list[dict]:
    mats: dict[tuple[str, str], dict] = {}
    scaps_files = [
        ("paper4_scaps_materials.csv", "10.1038/s41598-025-98351-y"),
        ("paper_cs_pb_scaps_materials.csv", "10.1038/s41598-024-81797-x"),
        ("paper_besip2_scaps_materials.csv", "10.1007/s11082-026-08738-y"),
    ]
    for fname, doi in scaps_files:
        path = RAW / fname
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["material"], doi)
                mats[key] = {
                    "material": row["material"],
                    "role": row["layer_role"],
                    "Eg_eV": float(row["Eg_eV"]),
                    "chi_eV": float(row["chi_eV"]),
                    "source_doi": doi,
                    "source_file": fname,
                }
    # Paper2 experimental
    with (RAW / "paper2_mps3_experimental.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ip = float(row["ionization_potential_eV"])
            eg = float(row["band_gap_eV"])
            mats[(row["material"], row["source_doi"])] = {
                "material": row["material"],
                "role": "2d_layer",
                "Eg_eV": eg,
                "chi_eV": round(ip - eg, 4),
                "ionization_potential_eV": ip,
                "source_doi": row["source_doi"],
                "source_file": "paper2_mps3_experimental.csv",
            }
    # Paper1 monolayers (Eg + VBM)
    with (RAW / "paper1_table2_monolayers.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eg = float(row["eg_hse06_eV"])
            vbm = float(row["ev_vbm_eV"])
            mats[(row["material"], "10.1038/s41699-021-00200-9")] = {
                "material": row["material"],
                "role": "2d_monolayer",
                "Eg_eV": eg,
                "chi_eV": round(-(vbm + eg), 4),
                "ev_vbm_eV": vbm,
                "source_doi": "10.1038/s41699-021-00200-9",
                "source_file": "paper1_table2_monolayers.csv",
            }
    # Ozcelik
    with (RAW / "ozcelik_prb2016_monolayers.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eg = float(row["eg_hse_eV"])
            vbm = float(row["ev_vbm_hse_eV"])
            mats[(row["material"], row["source_doi"])] = {
                "material": row["material"],
                "role": "2d_monolayer",
                "Eg_eV": eg,
                "chi_eV": round(-float(row["ec_cbm_hse_eV"]), 4),
                "ev_vbm_eV": vbm,
                "source_doi": row["source_doi"],
                "source_file": "ozcelik_prb2016_monolayers.csv",
            }
    return list(mats.values())


def audit_material(m: dict) -> dict:
    name = m["material"]
    eg = m["Eg_eV"]
    chi = m["chi_eV"]
    base = name.split(":")[0]  # TiO2:N → TiO2 for family check
    result = {
        **m,
        "status": "pass",
        "flags": [],
        "literature_check": "no_independent_range",
    }

    # Hard fail: TiO2 family with MP-like gap
    if base.lower().startswith("tio2") and eg < TIO2_HARD_MIN:
        result["status"] = "fail"
        result["flags"].append(f"TiO2-scale error: Eg={eg} < {TIO2_HARD_MIN} (MP~2.3 trap)")
        result["literature_check"] = LITERATURE_RANGES.get("TiO2", ("?", "?", "?", "?", ""))[4]
        return result

    key = name if name in LITERATURE_RANGES else (base if base in LITERATURE_RANGES else None)
    if key is None:
        # Generic sanity: Eg in (0.3, 8), chi in (1.0, 6.5)
        if not (0.3 <= eg <= 8.0):
            result["status"] = "fail"
            result["flags"].append(f"Eg={eg} outside global sanity [0.3, 8]")
        elif not (1.0 <= chi <= 6.5):
            result["status"] = "warn"
            result["flags"].append(f"chi={chi} unusual vs typical SCAPS [1, 6.5]")
        else:
            result["status"] = "pass_paper_only"
            result["literature_check"] = "traceable to cited paper; no separate web range keyed"
        return result

    eg_lo, eg_hi, chi_lo, chi_hi, note = LITERATURE_RANGES[key]
    result["literature_check"] = note
    eg_ok = eg_lo <= eg <= eg_hi
    chi_ok = chi_lo <= chi <= chi_hi
    if not eg_ok:
        # allow mild (±0.25) overflow as warn; >0.5 as fail
        delta = min(abs(eg - eg_lo), abs(eg - eg_hi)) if eg < eg_lo or eg > eg_hi else 0
        if eg < eg_lo:
            delta = eg_lo - eg
        else:
            delta = eg - eg_hi
        if delta > 0.5:
            result["status"] = "fail"
            result["flags"].append(f"Eg={eg} outside literature [{eg_lo},{eg_hi}] by {delta:.2f} eV")
        else:
            result["status"] = "warn"
            result["flags"].append(f"Eg={eg} mildly outside [{eg_lo},{eg_hi}] (Δ={delta:.2f})")
    if not chi_ok:
        if chi < chi_lo:
            delta = chi_lo - chi
        else:
            delta = chi - chi_hi
        if delta > 0.5:
            result["status"] = "fail" if result["status"] != "warn" else "fail"
            result["flags"].append(f"chi={chi} outside literature [{chi_lo},{chi_hi}] by {delta:.2f} eV")
        else:
            if result["status"] == "pass":
                result["status"] = "warn"
            result["flags"].append(f"chi={chi} mildly outside [{chi_lo},{chi_hi}] (Δ={delta:.2f})")
    if eg_ok and chi_ok:
        result["status"] = "pass"
    return result


def main() -> None:
    materials = collect_unique_materials()
    audits = [audit_material(m) for m in materials]
    fails = [a for a in audits if a["status"] == "fail"]
    warns = [a for a in audits if a["status"] == "warn"]
    report = {
        "n_unique_materials": len(audits),
        "n_pass": sum(1 for a in audits if a["status"] in ("pass", "pass_paper_only")),
        "n_warn": len(warns),
        "n_fail": len(fails),
        "failures": fails,
        "warnings": warns,
        "audits": audits,
        "policy": {
            "tio2_hard_min_eV": TIO2_HARD_MIN,
            "mild_tolerance_eV": 0.25,
            "fail_tolerance_eV": 0.5,
            "note": "Mild deviations allowed; TiO2-scale (~1 eV) errors fail",
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_unique_materials": report["n_unique_materials"],
                "n_pass": report["n_pass"],
                "n_warn": report["n_warn"],
                "n_fail": report["n_fail"],
                "failures": [
                    {"material": f["material"], "Eg": f["Eg_eV"], "chi": f["chi_eV"], "flags": f["flags"]}
                    for f in fails
                ],
                "warnings_sample": [
                    {"material": w["material"], "Eg": w["Eg_eV"], "flags": w["flags"]} for w in warns[:10]
                ],
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
