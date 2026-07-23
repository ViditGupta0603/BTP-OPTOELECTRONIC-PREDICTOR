"""Verify curated perovskite extracts before merge (no non-perovskite scoring)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from literature_bands import Layer, cbo_absorber_etl, junction_type, stack_row, vbo_absorber_htl  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "research_paper_verification_report.json"

FAIL_DELTA = 0.55
WARN_DELTA = 0.30
TIO2_HARD = 2.9

WEB: dict[str, tuple[float, float, float | None, float | None, str, str]] = {
    "Cs3Sb2Br9": (1.70, 2.60, 3.7, 4.2, "HSE/SCAPS ~1.95–2.0 eV; chi~3.98 common", "high"),
    "TiO2": (2.9, 3.5, 3.8, 4.2, "bulk device ETL", "high"),
    "CFTS": (1.2, 1.5, 3.1, 3.5, "Cu2FeSnS4 SCAPS ~1.3 eV", "medium"),
    "K2GeI6": (1.15, 1.40, None, None, "Noor DFT ~1.27; Raj DFT 1.28", "high"),
}


def score_eg(eg: float, lo: float, hi: float) -> tuple[str, float]:
    if eg < lo:
        d = lo - eg
    elif eg > hi:
        d = eg - hi
    else:
        d = 0.0
    if d > FAIL_DELTA:
        return "FAIL", d
    if d > WARN_DELTA:
        return "WARN", d
    return "PASS", d


def score_chi(chi: float | None, lo: float | None, hi: float | None) -> tuple[str, float]:
    if chi is None or lo is None or hi is None:
        return "SKIP", 0.0
    if chi < lo:
        d = lo - chi
    elif chi > hi:
        d = chi - hi
    else:
        d = 0.0
    if d > FAIL_DELTA:
        return "FAIL", d
    if d > WARN_DELTA:
        return "WARN", d
    return "PASS", d


def check_material(name: str, eg: float | None, chi: float | None) -> dict:
    if name not in WEB:
        return {"material": name, "status": "NO_RANGE", "eg": eg, "chi": chi}
    lo, hi, clo, chi_hi, note, conf = WEB[name]
    st_eg, d_eg = ("SKIP", 0.0) if eg is None else score_eg(eg, lo, hi)
    st_chi, d_chi = score_chi(chi, clo, chi_hi)
    if name.startswith("TiO2") and eg is not None and eg < TIO2_HARD:
        return {"material": name, "status": "FAIL", "eg": eg, "note": "TiO2 hard floor"}
    ranks = {"FAIL": 2, "WARN": 1, "PASS": 0, "SKIP": 0}
    status = max([st_eg, st_chi], key=lambda s: ranks[s])
    return {
        "material": name,
        "status": status,
        "eg": eg,
        "chi": chi,
        "eg_status": st_eg,
        "eg_delta": d_eg,
        "chi_status": st_chi,
        "note": note,
        "confidence": conf,
    }


def verify_cs3sb2br9() -> dict:
    path = RAW / "paper_cs3sb2br9_scaps_materials.csv"
    layers: dict[str, Layer] = {}
    checks = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eg, chi = float(row["Eg_eV"]), float(row["chi_eV"])
            layers[row["layer_role"]] = Layer(row["material"], eg, chi)
            checks.append(check_material(row["material"], eg, chi))
    a, e, h = layers["absorber"], layers["etl"], layers["htl"]
    row = stack_row(a, e, h, "10.1016/j.nxmate.2026.102491", "Dubey Cs3Sb2Br9 Table 1")
    any_fail = any(c["status"] == "FAIL" for c in checks)
    return {
        "paper": "next aditya dubey.pdf",
        "action": "MERGE" if not any_fail else "HOLD",
        "material_checks": checks,
        "stack_row": row,
    }


def verify_k2gei6() -> dict:
    path = RAW / "paper_k2gei6_dft_absorber.csv"
    with path.open(encoding="utf-8") as f:
        r = next(csv.DictReader(f))
    check = check_material("K2GeI6", float(r["Eg_eV"]), None)
    return {
        "paper": "kaushiki.pdf",
        "action": "MERGE_ABSORBER_ONLY" if check["status"] != "FAIL" else "HOLD",
        "material_checks": [check],
    }


def main() -> None:
    report = {
        "perovskite_candidates": [verify_cs3sb2br9(), verify_k2gei6()],
        "excluded": [
            {
                "paper": "tilt ppr.pdf",
                "action": "EXCLUDE",
                "reason": "Ca3NI3 extract failed lit check (CdS Eg=1.64; absorber Eg contested)",
            }
        ],
        "skipped_non_perovskite": ["DSSC", "GaInP", "CdTe"],
    }
    mergeable = [p for p in report["perovskite_candidates"] if str(p["action"]).startswith("MERGE")]
    report["summary"] = {
        "merge_papers": [p["paper"] for p in mergeable],
        "verdict": "VERIFIED_PARTIAL_MERGE_OK" if mergeable else "NOTHING_TO_MERGE",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
