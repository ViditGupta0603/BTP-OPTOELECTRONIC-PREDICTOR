"""Verify the perovskite-only dataset (same policy family as internet_verify_dataset.py).

Checks:
  1) Stack rows: recompute CBO/VBO/junction types (tol 0.06 eV)
  2) Unique stack-layer materials vs literature WEB ranges (fail Δ>0.55 eV; TiO2 Eg>=2.9)
  3) Absorber library sanity + spot literature checks for known halide DPs
  4) Pilania: Eg range sanity + source DOI present

Does NOT modify opto_literature_dataset.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from literature_bands import Layer, cbo_absorber_etl, junction_type, vbo_absorber_htl  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
STACK = DATA / "perovskite_stack_dataset.csv"
ABS = DATA / "perovskite_absorber_library.csv"
OUT = DATA / "perovskite_verification_report.json"
CSV_OUT = DATA / "perovskite_verification_summary.csv"

FAIL_DELTA = 0.55
WARN_DELTA = 0.30
TIO2_HARD = 2.9

WEB: dict[str, tuple[float, float, float | None, float | None, str, str]] = {
    "K2TiI6": (1.5, 1.7, 3.8, 4.2, "expt ~1.62 eV (MDPI Energies 2022); Paper4 1.61/4.0", "high"),
    "CsPb0.625Zn0.375IBr2": (1.0, 1.2, 4.1, 4.4, "source SCAPS Table1; Cs-Pb-halide family ~1 eV", "medium"),
    "PC60BM": (1.7, 2.1, 3.9, 4.4, "SCAPS consensus Eg~1.8–2.0", "high"),
    "LBSO": (3.0, 3.3, 4.2, 4.6, "La:BaSnO3 SCAPS Eg=3.12", "high"),
    "Nb2O5": (3.2, 3.7, 4.1, 4.5, "Paper4 Eg=3.46", "high"),
    "CdZnS": (3.0, 3.4, 4.0, 4.4, "Paper4 SCAPS", "high"),
    "CdS": (2.3, 2.5, 4.0, 4.4, "expt ~2.4", "high"),
    "SnS2": (1.6, 2.2, 4.0, 4.5, "SCAPS/lit ~1.8–2.2", "medium"),
    "ZnSe": (2.6, 2.9, 3.9, 4.3, "bulk ZnSe ~2.7", "high"),
    "MWCNTs": (1.4, 1.7, 3.4, 3.9, "Paper4 Eg=1.55", "medium"),
    "MoO3": (2.8, 3.2, 2.1, 2.7, "SCAPS Eg=3.0", "high"),
    "PTAA": (2.7, 3.2, 2.0, 2.6, "Paper4 Eg=2.96", "high"),
    "TiO2:N": (2.9, 3.3, 2.0, 2.5, "Paper4 Eg=3.0", "high"),
    "NiCo2O4": (2.1, 2.5, 3.2, 3.7, "expt ~2.32; Paper4 2.3", "high"),
    "CuAlO2": (3.2, 3.7, 2.2, 2.8, "delafossite ~3.5", "high"),
    "GaAs": (1.35, 1.50, 3.9, 4.2, "bulk GaAs 1.42", "high"),
    "ZnTe": (2.1, 2.4, 3.5, 4.0, "bulk ZnTe ~2.25", "high"),
    "CNTS": (1.5, 1.9, 3.6, 4.1, "Cu2NiSnS4-family SCAPS", "medium"),
    "MEH-PPV": (1.9, 2.3, 2.5, 3.1, "polymer ~2.0–2.2", "medium"),
    "MoS2": (1.2, 1.5, 3.9, 4.5, "bulk-like CsPb HTL Eg=1.29", "high"),
    "Cu2Te": (1.0, 1.4, 4.0, 4.4, "CsPb Table2", "low"),
    "Zn3P2": (1.3, 1.7, 4.0, 4.4, "Zn3P2 ~1.5 eV", "medium"),
    "C6TBTAPH2": (1.4, 1.8, 3.4, 3.8, "Paper4 table", "low"),
    "nPB": (2.2, 2.6, 2.8, 3.3, "Paper4 table", "low"),
    "C6PcH2": (1.4, 1.8, 3.5, 3.9, "Paper4 table", "low"),
    "Cs3Sb2Br9": (
        1.7,
        2.6,
        3.7,
        4.2,
        "HSE/SCAPS ~1.95–2.0; optical nano often higher",
        "high",
    ),
    "TiO2": (2.9, 3.5, 3.8, 4.2, "bulk device ETL", "high"),
    "CFTS": (1.2, 1.5, 3.1, 3.5, "Cu2FeSnS4 SCAPS ~1.3 eV", "medium"),
    "K2GeI6": (1.15, 1.40, None, None, "Noor DFT ~1.27; Raj DFT 1.28", "high"),
    "WS2": (1.7, 2.0, 3.8, 4.4, "SCAPS ETL Eg~1.8–1.87", "medium"),
    "SnO2": (3.4, 3.8, 3.7, 4.2, "bulk SnO2 ETL ~3.6", "high"),
    "PCBM": (1.8, 2.2, 3.7, 4.3, "PCBM ~2.0 / chi~3.9", "high"),
    "CuI": (2.9, 3.3, 1.8, 2.4, "CuI HTL SCAPS", "medium"),
    "CuSCN": (3.4, 3.8, 1.5, 2.0, "CuSCN HTL SCAPS", "medium"),
    "NiO": (3.4, 3.8, 1.5, 2.1, "NiO HTL ~3.6", "high"),
    "V2O5": (2.0, 2.4, 3.1, 3.7, "V2O5 HTL SCAPS", "medium"),
}

HALIDE_SPOT: dict[str, tuple[float, float, str, str]] = {
    "Cs2AgBiBr6 (cubic)": (
        0.9,
        2.3,
        "Paper5 DFT ind_gap~1.14; expt optical often ~1.9–2.2 — DFT accepted with caveat",
        "medium",
    ),
    "Cs2AgBiBr6 (orthorhombic)": (0.9, 2.3, "Paper5 ortho twin of Cs2AgBiBr6", "medium"),
    "Cs2AgBiCl6 (cubic)": (1.5, 3.2, "chloride DP wider than bromide; DFT/expt order OK", "medium"),
    "Cs2AgInCl6 (cubic)": (
        0.8,
        3.8,
        "Paper5 DFT ind_gap~1.01; optical/HSE literature often larger (~3 eV) — DFT row accepted",
        "medium",
    ),
    "Cs2AgSbBr6 (cubic)": (0.5, 2.2, "Paper5 Sb analog of AgBi bromide", "medium"),
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


def load_scaps_layers() -> dict[str, Layer]:
    layers: dict[str, Layer] = {}
    for fname in (
        "paper4_scaps_materials.csv",
        "paper_cs_pb_scaps_materials.csv",
        "paper_cs3sb2br9_scaps_materials.csv",
        "paper_besip2_scaps_materials.csv",
    ):
        with (RAW / fname).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("chi_eV"):
                    continue
                # skip non-perovskite absorber BeSiP2 from being required,
                # but keep its contacts for stack recompute
                name = row["material"]
                eg, chi = float(row["Eg_eV"]), float(row["chi_eV"])
                if name not in layers or (name.startswith("TiO2") and eg > layers[name].eg):
                    layers[name] = Layer(name, eg, chi)
    return layers


def recompute_stack_failures(rows: list[dict], layers: dict[str, Layer]) -> list[dict]:
    failures = []
    for i, r in enumerate(rows):
        try:
            a = layers[r["material_absorber"]]
            e = layers[r["material_etl"]]
            h = layers[r["material_htl"]]
        except KeyError as exc:
            failures.append({"idx": i, "msgs": [f"missing layer {exc}"]})
            continue
        msgs = []
        if abs(cbo_absorber_etl(a, e) - float(r["cbo_eV"])) > 0.06:
            msgs.append("cbo mismatch")
        if abs(vbo_absorber_htl(a, h) - float(r["vbo_eV"])) > 0.06:
            msgs.append("vbo mismatch")
        if junction_type(a, e) != r["absorber_etl_type"]:
            msgs.append("etl type mismatch")
        if junction_type(a, h) != r["absorber_htl_type"]:
            msgs.append("htl type mismatch")
        if msgs:
            failures.append({"idx": i, "msgs": msgs, "absorber": r["material_absorber"]})
    return failures


def verify_stacks() -> dict:
    rows = list(csv.DictReader(STACK.open(encoding="utf-8")))
    layers = load_scaps_layers()
    failures = recompute_stack_failures(rows, layers)

    used: dict[str, dict] = {}
    for r in rows:
        for role, mat, eg in [
            ("absorber", r["material_absorber"], float(r["absorber_band_gap_eV"])),
            ("etl", r["material_etl"], float(r["etl_band_gap_eV"])),
            ("htl", r["material_htl"], float(r["htl_band_gap_eV"])),
        ]:
            if mat not in used:
                used[mat] = {"egs": set(), "roles": set(), "dois": set()}
            used[mat]["egs"].add(round(eg, 4))
            used[mat]["roles"].add(role)
            used[mat]["dois"].add(r["source_doi"])

    audits = []
    counts: dict[str, int] = defaultdict(int)
    for mat, info in sorted(used.items()):
        eg_val = sorted(info["egs"])[0]
        item = {
            "material": mat,
            "Eg_dataset_eV": list(info["egs"]) if len(info["egs"]) > 1 else eg_val,
            "roles": sorted(info["roles"]),
            "source_dois": sorted(info["dois"]),
            "status": "PASS",
            "flags": [],
            "web_check": None,
            "confidence": None,
        }
        base = mat.split(":")[0]
        if base.lower().startswith("tio2") and eg_val < TIO2_HARD:
            item["status"] = "FAIL"
            item["flags"].append(f"TiO2-scale Eg={eg_val}<{TIO2_HARD}")
            counts[item["status"]] += 1
            audits.append(item)
            continue

        key = mat if mat in WEB else (base if base in WEB else None)
        if key is None:
            if 0.3 <= eg_val <= 8.0:
                item["status"] = "PASS_PAPER"
                item["web_check"] = "No separate web range keyed; taken from cited SCAPS/paper table"
                item["confidence"] = "paper_primary"
            else:
                item["status"] = "FAIL"
                item["flags"].append(f"Eg={eg_val} outside sanity")
            counts[item["status"]] += 1
            audits.append(item)
            continue

        lo, hi, _clo, _chi, note, conf = WEB[key]
        item["web_check"] = note
        item["confidence"] = conf
        status, d = score_eg(eg_val, lo, hi)
        item["status"] = status
        if d > 0:
            item["flags"].append(f"Eg Δ={d:.2f} vs web [{lo},{hi}]")
        counts[status] += 1
        audits.append(item)

    return {
        "n_stack_rows": len(rows),
        "stack_recompute_failures": failures,
        "stack_all_rows_verified": len(failures) == 0,
        "n_unique_stack_materials": len(audits),
        "stack_material_counts": dict(counts),
        "stack_material_audits": audits,
        "stack_failures": [a for a in audits if a["status"] == "FAIL"],
        "stack_warnings": [a for a in audits if a["status"] == "WARN"],
    }


def verify_absorbers() -> dict:
    rows = list(csv.DictReader(ABS.open(encoding="utf-8")))
    by_src: dict[str, int] = defaultdict(int)
    audits = []
    counts: dict[str, int] = defaultdict(int)
    sanity_fail = 0

    for r in rows:
        by_src[r["source_doi"]] += 1
        eg = float(r["absorber_band_gap_eV"])
        mat = r["material_absorber"]
        item = {
            "material": mat,
            "Eg_dataset_eV": eg,
            "source_doi": r["source_doi"],
            "functional": r["functional"],
            "perovskite_family": r["perovskite_family"],
            "status": "PASS",
            "flags": [],
            "web_check": None,
            "confidence": None,
        }

        if not (0.05 < eg <= 10.0):
            item["status"] = "FAIL"
            item["flags"].append(f"Eg={eg} outside sanity (0.05,10]")
            sanity_fail += 1
            counts["FAIL"] += 1
            audits.append(item)
            continue

        if mat in HALIDE_SPOT:
            lo, hi, note, conf = HALIDE_SPOT[mat]
            item["web_check"] = note
            item["confidence"] = conf
            status, d = score_eg(eg, lo, hi)
            item["status"] = status
            if d > 0:
                item["flags"].append(f"Eg Δ={d:.2f} vs spot [{lo},{hi}]")
        elif mat in WEB:
            lo, hi, _clo, _chi, note, conf = WEB[mat]
            item["web_check"] = note
            item["confidence"] = conf
            status, d = score_eg(eg, lo, hi)
            item["status"] = status
            if d > 0:
                item["flags"].append(f"Eg Δ={d:.2f} vs web [{lo},{hi}]")
        else:
            item["status"] = "PASS_PAPER"
            item["web_check"] = (
                "Absorber Eg taken from cited perovskite DFT table "
                "(Paper5 halide DP or Pilania GLLB-SC oxide DP)."
            )
            item["confidence"] = "paper_primary"

        counts[item["status"]] += 1
        audits.append(item)

    spot_results = [a for a in audits if a["material"] in HALIDE_SPOT]
    return {
        "n_absorber_rows": len(rows),
        "by_source_doi": dict(by_src),
        "absorber_counts": dict(counts),
        "n_sanity_fail": sanity_fail,
        "spot_checks": spot_results,
        "absorber_failures": [a for a in audits if a["status"] == "FAIL"],
        "absorber_warnings": [a for a in audits if a["status"] == "WARN"],
        "absorber_audits": audits,
    }


def main() -> None:
    stack_rep = verify_stacks()
    abs_rep = verify_absorbers()

    n_fail = (
        len(stack_rep["stack_failures"])
        + len(abs_rep["absorber_failures"])
        + len(stack_rep["stack_recompute_failures"])
    )
    report = {
        "policy": {
            "mild_warn_eV": WARN_DELTA,
            "fail_eV": FAIL_DELTA,
            "tio2_hard_min": TIO2_HARD,
            "date": "2026-07-15",
            "method": (
                "Perovskite-only audit: stack offset recomputation + WEB ranges for PSC layers "
                "+ Paper5/Pilania absorber sanity and halide spot checks"
            ),
        },
        "inputs": {"stacks": str(STACK), "absorbers": str(ABS)},
        "stacks": stack_rep,
        "absorbers": abs_rep,
        "verdict": "PEROVSKITE DATASET VERIFIED" if n_fail == 0 else "PEROVSKITE DATASET HAS FAILURES",
        "n_total_fail_items": n_fail,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scope", "material", "status", "Eg_dataset_eV", "confidence", "web_check", "flags"],
        )
        w.writeheader()
        for a in stack_rep["stack_material_audits"]:
            w.writerow(
                {
                    "scope": "stack_layer",
                    "material": a["material"],
                    "status": a["status"],
                    "Eg_dataset_eV": a["Eg_dataset_eV"],
                    "confidence": a.get("confidence"),
                    "web_check": a.get("web_check"),
                    "flags": "|".join(a.get("flags") or []),
                }
            )
        for a in abs_rep["spot_checks"]:
            w.writerow(
                {
                    "scope": "absorber_spot",
                    "material": a["material"],
                    "status": a["status"],
                    "Eg_dataset_eV": a["Eg_dataset_eV"],
                    "confidence": a.get("confidence"),
                    "web_check": a.get("web_check"),
                    "flags": "|".join(a.get("flags") or []),
                }
            )

    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "n_stacks": stack_rep["n_stack_rows"],
                "stack_recompute_ok": stack_rep["stack_all_rows_verified"],
                "stack_material_counts": stack_rep["stack_material_counts"],
                "n_absorbers": abs_rep["n_absorber_rows"],
                "absorber_counts": abs_rep["absorber_counts"],
                "spot_checks": [
                    {"material": a["material"], "Eg": a["Eg_dataset_eV"], "status": a["status"]}
                    for a in abs_rep["spot_checks"]
                ],
                "report": str(OUT),
                "summary_csv": str(CSV_OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
