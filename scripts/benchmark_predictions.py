"""Benchmark OptoStack ML predictions vs literature/DFT ground truth.

Compares predict_eg / formula_estimator outputs (no lookup, no LLM) against
verified Eg and χ from project datasets and a small external literature set.

  python scripts/benchmark_predictions.py

Writes:
  data/perovskite_prediction_benchmark.json
  data/perovskite_prediction_benchmark.csv
  data/perovskite_prediction_benchmark.md
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formula_estimator import base_name, estimate_eg_chi  # noqa: E402
from predict_stack import (  # noqa: E402
    ABS_PATH,
    EG_MODEL,
    RAW,
    _feature_frame,
    load_layer_lookup,
    predict_eg,
)

DATA = ROOT / "data"
OUT_JSON = DATA / "perovskite_prediction_benchmark.json"
OUT_CSV = DATA / "perovskite_prediction_benchmark.csv"
OUT_MD = DATA / "perovskite_prediction_benchmark.md"

RANDOM_SEED = 42
N_SAMPLE = 40

# Small external holdout set — literature/DFT values with DOI (not in absorber library)
EXTERNAL_HOLDOUT = [
    {
        "material": "CsPbI3",
        "actual_Eg": 1.73,
        "actual_chi": None,
        "source": "literature_experimental",
        "source_doi": "10.3390/physchem5010003",
        "notes": "α-CsPbI3 photoactive phase, experimental UV-Vis",
    },
    {
        "material": "CsPbBr3",
        "actual_Eg": 2.36,
        "actual_chi": None,
        "source": "literature_experimental",
        "source_doi": "10.1039/D3CP05956A",
        "notes": "CsPbBr3 nanocrystals, optical band gap",
    },
    {
        "material": "CsPbCl3",
        "actual_Eg": 2.98,
        "actual_chi": None,
        "source": "literature_experimental",
        "source_doi": "10.1039/D3CP05956A",
        "notes": "CsPbCl3 nanocrystals, optical band gap",
    },
    {
        "material": "CH3NH3PbI3",
        "actual_Eg": 1.55,
        "actual_chi": None,
        "source": "literature_experimental",
        "source_doi": "10.1038/nature12340",
        "notes": "MAPbI3 thin film, widely cited experimental Eg",
    },
    {
        "material": "HC(NH2)2PbI3",
        "actual_Eg": 1.48,
        "actual_chi": None,
        "source": "literature_experimental",
        "source_doi": "10.1038/ncomms7382",
        "notes": "FAPbI3 α-phase experimental band gap",
    },
]

SCAPS_FILES = (
    "paper4_scaps_materials.csv",
    "paper_cs_pb_scaps_materials.csv",
    "paper_cs3sb2br9_scaps_materials.csv",
    "paper_besip2_scaps_materials.csv",
    "paper_k2gei6_dft_absorber.csv",
)


def _fmt(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def render_benchmark_markdown(report: dict) -> str:
    """Human-readable summary from the benchmark report dict."""
    m = report["metrics"]
    lines = [
        "# Perovskite Prediction Benchmark",
        "",
        f"**Date:** {report['benchmark_date']}  ",
        f"**Method:** {report['method']}  ",
        f"**Random seed:** {report['random_seed']}",
        "",
        "## Summary metrics",
        "",
        "Eg from `predict_eg` (RandomForest regressor, lookup bypassed). "
        "χ from `formula_estimator` on SCAPS literature tables.",
        "",
        "| Test set | *n* | Eg MAE (eV) | Eg RMSE (eV) | Eg R² | χ MAE (eV) |",
        "|----------|----:|------------:|-------------:|------:|-----------:|",
    ]

    eg_rows = [
        ("Random library sample", "random_sample_40", None),
        ("External literature holdout", "external_literature_holdout", None),
        ("Combined Eg sample", "combined_Eg_sample", None),
        ("SCAPS literature", "scaps_literature_Eg", "scaps_literature_chi"),
        ("Holdout retrain (20% library)", "holdout_retrain_20pct_full_library", None),
    ]
    for label, eg_key, chi_key in eg_rows:
        eg = m[eg_key]
        if "predict_eg_regressor" in eg:
            eg = eg["predict_eg_regressor"]
        chi_mae = m[chi_key]["MAE"] if chi_key else None
        lines.append(
            f"| {label} | {eg['n']} | {_fmt(eg['MAE'])} | {_fmt(eg['RMSE'])} "
            f"| {_fmt(eg['R2'])} | {_fmt(chi_mae)} |"
        )

    lines.extend(
        [
            "",
            "## Best & worst Eg predictions",
            "",
            "### Worst 5 (largest |error|)",
            "",
            "| Material | Actual Eg | Predicted Eg | Error | Group |",
            "|----------|----------:|-------------:|------:|-------|",
        ]
    )
    for r in report["worst_5_Eg_predictions"]:
        lines.append(
            f"| {r['material']} | {_fmt(r['actual_Eg'], 2)} | {_fmt(r['predicted_Eg'], 2)} "
            f"| {_fmt(r['error_Eg'], 2)} | {r['sample_group']} |"
        )

    lines.extend(
        [
            "",
            "### Best 5 (smallest |error|)",
            "",
            "| Material | Actual Eg | Predicted Eg | Error | Group |",
            "|----------|----------:|-------------:|------:|-------|",
        ]
    )
    for r in report["best_5_Eg_predictions"]:
        lines.append(
            f"| {r['material']} | {_fmt(r['actual_Eg'], 2)} | {_fmt(r['predicted_Eg'], 2)} "
            f"| {_fmt(r['error_Eg'], 2)} | {r['sample_group']} |"
        )

    lines.extend(["", "## Key takeaways", ""])
    rand = m["random_sample_40"]["predict_eg_regressor"]
    ext = m["external_literature_holdout"]["predict_eg_regressor"]
    chi = m["scaps_literature_chi"]
    hold = m["holdout_retrain_20pct_full_library"]
    ext_note = (
        f"lead-halide ABX₃ perovskites (CsPbX₃, MAPbI₃, FAPbI₃) Eg MAE {_fmt(ext['MAE'])} eV"
        if ext["MAE"] < 0.5
        else "lead-halide perovskites (CsPbX₃, MAPbI₃, FAPbI₃) remain poorly predicted"
    )
    lines.extend(
        [
            f"- **In-distribution (random library, *n*={rand['n']}):** Eg MAE {_fmt(rand['MAE'])} eV, R² {_fmt(rand['R2'])} — double-perovskite formulas from the training library.",
            f"- **External literature holdout (*n*={ext['n']}):** Eg MAE {_fmt(ext['MAE'])} eV, R² {_fmt(ext['R2'])} — {ext_note}.",
            f"- **χ on SCAPS literature (*n*={chi['n']}):** MAE {_fmt(chi['MAE'])} eV (median AE {_fmt(chi['median_AE'])} eV); includes ETL/HTL contact layers with occasional large outliers (e.g. CuSCN, PTAA).",
            f"- **Full-library holdout retrain (*n*={hold['n']}):** Eg MAE {_fmt(hold['MAE'])} eV, R² {_fmt(hold['R2'])} — cross-validated generalization on unseen formulas from the absorber library.",
        ]
    )

    lines.extend(["", "## Caveats", ""])
    for c in report["caveats"]:
        lines.append(f"- {c}")

    lines.extend(
        [
            "",
            "## Regenerate",
            "",
            "```bash",
            "python scripts/benchmark_predictions.py",
            "```",
            "",
            "Outputs: `data/perovskite_prediction_benchmark.json`, `.csv`, and this `.md` file.",
            "",
            "Per-row predictions: see `data/perovskite_prediction_benchmark.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"n": 0}
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "median_AE": float(np.median(np.abs(err))),
        "max_AE": float(np.max(np.abs(err))),
    }


def load_absorber_ground_truth() -> list[dict]:
    rows: list[dict] = []
    with ABS_PATH.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("material_absorber") or "").strip()
            eg = r.get("absorber_band_gap_eV")
            if not name or not eg:
                continue
            chi = float(r["chi_eV"]) if r.get("chi_eV") else None
            rows.append(
                {
                    "material": name,
                    "actual_Eg": float(eg),
                    "actual_chi": chi,
                    "source": "perovskite_absorber_library",
                    "source_doi": r.get("source_doi") or "",
                    "gap_method": r.get("gap_method") or "",
                    "material_class": r.get("material_class") or "",
                    "chi_is_literature": False,
                }
            )
    return rows


def load_scaps_ground_truth() -> list[dict]:
    rows: list[dict] = []
    for fname in SCAPS_FILES:
        path = RAW / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("material") or not r.get("Eg_eV"):
                    continue
                role = (r.get("layer_role") or "absorber").strip().lower()
                chi = float(r["chi_eV"]) if r.get("chi_eV") else None
                rows.append(
                    {
                        "material": r["material"].strip(),
                        "role": role,
                        "actual_Eg": float(r["Eg_eV"]),
                        "actual_chi": chi,
                        "source": "literature_SCAPS",
                        "source_doi": r.get("source_doi") or "",
                        "file": fname,
                    }
                )
    return rows


def in_layer_lookup(material: str, layers: dict) -> bool:
    bn = base_name(material)
    low = material.lower()
    for k in layers:
        if k == material or k == bn or k.lower() == low or base_name(k).lower() == bn.lower():
            return True
    return False


def predict_ml_eg(material: str) -> float:
    """Eg via perovskite_eg_regressor (no lookup)."""
    return float(predict_eg(material))


def predict_ml_eg_chi(material: str, role: str = "absorber") -> tuple[float, float]:
    """Eg+χ via formula_estimator (no lookup)."""
    est = estimate_eg_chi(material, role)
    return float(est["Eg_eV"]), float(est["chi_eV"])


def build_random_sample(absorber_rows: list[dict], layers: dict, n: int) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    pool = list(absorber_rows)
    rng.shuffle(pool)
    chosen = pool[:n]
    for row in chosen:
        row = dict(row)
        row["in_layer_lookup"] = in_layer_lookup(row["material"], layers)
        row["sample_group"] = "random_library"
    return [dict(r, in_layer_lookup=in_layer_lookup(r["material"], layers), sample_group="random_library") for r in chosen]


def run_holdout_retrain(absorber_rows: list[dict]) -> dict:
    """Leave-20%-out: retrain Eg regressor on subset, predict held-out formulas."""
    formulas = [r["material"] for r in absorber_rows]
    y = np.array([r["actual_Eg"] for r in absorber_rows], dtype=float)
    keys, X = _feature_frame(formulas)
    Xtr, Xte, ytr, yte, ftr, fte = train_test_split(
        X, y, formulas, test_size=0.2, random_state=RANDOM_SEED
    )
    model = RandomForestRegressor(
        n_estimators=300, max_depth=24, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    per_row = []
    for mat, actual, predicted in zip(fte, yte, pred):
        per_row.append(
            {
                "material": mat,
                "actual_Eg": float(actual),
                "predicted_Eg": float(predicted),
                "error_Eg": float(predicted - actual),
                "sample_group": "holdout_retrain_20pct",
            }
        )
    return {
        "description": "Retrained RandomForest on 80% of absorber library, tested on 20% holdout",
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "metrics": _metrics(yte, pred),
        "rows": per_row,
    }


def main() -> None:
    if not EG_MODEL.exists():
        raise FileNotFoundError("Train models first: python scripts/predict_stack.py --train")

    layers = load_layer_lookup()
    absorber_gt = load_absorber_ground_truth()
    scaps_gt = load_scaps_ground_truth()

    # --- Random sample from library (ML path, bypass lookup) ---
    random_sample = build_random_sample(absorber_gt, layers, N_SAMPLE)

    # --- External literature holdout (not in training library) ---
    external = []
    lib_names = {base_name(r["material"]).lower() for r in absorber_gt}
    for row in EXTERNAL_HOLDOUT:
        ext = dict(row)
        ext["in_layer_lookup"] = in_layer_lookup(row["material"], layers)
        ext["sample_group"] = "external_literature_holdout"
        ext["not_in_library"] = base_name(row["material"]).lower() not in lib_names
        external.append(ext)

    benchmark_rows: list[dict] = []

    def add_row(
        material: str,
        actual_eg: float,
        actual_chi: float | None,
        source: str,
        sample_group: str,
        **extra,
    ) -> None:
        pred_eg_rf = predict_ml_eg(material)
        pred_eg_fe, pred_chi_fe = predict_ml_eg_chi(
            material, extra.get("role", "absorber")
        )
        row = {
            "material": material,
            "actual_Eg": actual_eg,
            "predicted_Eg": pred_eg_rf,
            "error_Eg": pred_eg_rf - actual_eg,
            "predicted_Eg_formula": pred_eg_fe,
            "error_Eg_formula": pred_eg_fe - actual_eg,
            "actual_chi": actual_chi,
            "predicted_chi": pred_chi_fe if actual_chi is not None else None,
            "error_chi": (pred_chi_fe - actual_chi) if actual_chi is not None else None,
            "source": source,
            "sample_group": sample_group,
            "prediction_method": "ml_no_lookup",
            "use_llm": False,
            **{k: v for k, v in extra.items() if k != "role"},
        }
        benchmark_rows.append(row)

    for r in random_sample:
        add_row(
            r["material"],
            r["actual_Eg"],
            r["actual_chi"],
            r["source"],
            r["sample_group"],
            in_layer_lookup=r["in_layer_lookup"],
            source_doi=r.get("source_doi", ""),
            gap_method=r.get("gap_method", ""),
        )

    for r in external:
        add_row(
            r["material"],
            r["actual_Eg"],
            r["actual_chi"],
            r["source"],
            r["sample_group"],
            in_layer_lookup=r["in_layer_lookup"],
            not_in_library=r.get("not_in_library"),
            source_doi=r.get("source_doi", ""),
            notes=r.get("notes", ""),
        )

    # --- SCAPS χ benchmark (all roles with literature χ) ---
    scaps_chi_rows: list[dict] = []
    for r in scaps_gt:
        if r["actual_chi"] is None:
            continue
        role = r.get("role", "absorber")
        pred_eg, pred_chi = predict_ml_eg_chi(r["material"], role)
        scaps_chi_rows.append(
            {
                "material": r["material"],
                "role": role,
                "actual_Eg": r["actual_Eg"],
                "predicted_Eg": pred_eg,
                "error_Eg": pred_eg - r["actual_Eg"],
                "actual_chi": r["actual_chi"],
                "predicted_chi": pred_chi,
                "error_chi": pred_chi - r["actual_chi"],
                "source": r["source"],
                "source_doi": r.get("source_doi", ""),
                "sample_group": "scaps_literature_chi",
            }
        )

    # --- Metrics ---
    rand_rows = [r for r in benchmark_rows if r["sample_group"] == "random_library"]
    ext_rows = [r for r in benchmark_rows if r["sample_group"] == "external_literature_holdout"]
    all_eg_rows = rand_rows + ext_rows

    y_eg = np.array([r["actual_Eg"] for r in all_eg_rows])
    p_eg_rf = np.array([r["predicted_Eg"] for r in all_eg_rows])
    p_eg_fe = np.array([r["predicted_Eg_formula"] for r in all_eg_rows])

    holdout = run_holdout_retrain(absorber_gt)

    chi_with_gt = [r for r in scaps_chi_rows if r["actual_chi"] is not None]
    y_chi = np.array([r["actual_chi"] for r in chi_with_gt])
    p_chi = np.array([r["predicted_chi"] for r in chi_with_gt])

    # Sort for worst/best
    by_abs_err = sorted(all_eg_rows, key=lambda r: abs(r["error_Eg"]), reverse=True)

    report = {
        "benchmark_date": "2026-07-16",
        "method": "ML only (predict_eg + formula_estimator), use_llm=False, lookup bypassed",
        "random_seed": RANDOM_SEED,
        "n_random_sample": len(rand_rows),
        "n_external_holdout": len(ext_rows),
        "n_scaps_chi": len(chi_with_gt),
        "caveats": [
            "Eg ground truth is DFT/literature from absorber library (Paper5 double perovskites, verified ABX3 lead halides) plus 5 external refs.",
            "χ in absorber library is mostly ML-estimated — χ metrics use SCAPS literature tables only (~42 materials).",
            "Predictions bypass layer_lookup; this tests ML generalization, not lookup accuracy.",
            "Perovskite absorbers only; SCAPS χ set includes ETL/HTL contact layers.",
        ],
        "metrics": {
            "random_sample_40": {
                "predict_eg_regressor": _metrics(
                    np.array([r["actual_Eg"] for r in rand_rows]),
                    np.array([r["predicted_Eg"] for r in rand_rows]),
                ),
                "formula_estimator_Eg": _metrics(
                    np.array([r["actual_Eg"] for r in rand_rows]),
                    np.array([r["predicted_Eg_formula"] for r in rand_rows]),
                ),
            },
            "external_literature_holdout": {
                "predict_eg_regressor": _metrics(
                    np.array([r["actual_Eg"] for r in ext_rows]),
                    np.array([r["predicted_Eg"] for r in ext_rows]),
                ),
                "formula_estimator_Eg": _metrics(
                    np.array([r["actual_Eg"] for r in ext_rows]),
                    np.array([r["predicted_Eg_formula"] for r in ext_rows]),
                ),
            },
            "combined_Eg_sample": {
                "predict_eg_regressor": _metrics(y_eg, p_eg_rf),
                "formula_estimator_Eg": _metrics(y_eg, p_eg_fe),
            },
            "scaps_literature_chi": _metrics(y_chi, p_chi),
            "scaps_literature_Eg": _metrics(
                np.array([r["actual_Eg"] for r in chi_with_gt]),
                np.array([r["predicted_Eg"] for r in chi_with_gt]),
            ),
            "holdout_retrain_20pct_full_library": holdout["metrics"],
        },
        "worst_5_Eg_predictions": [
            {
                "material": r["material"],
                "actual_Eg": r["actual_Eg"],
                "predicted_Eg": r["predicted_Eg"],
                "error_Eg": r["error_Eg"],
                "sample_group": r["sample_group"],
            }
            for r in by_abs_err[:5]
        ],
        "best_5_Eg_predictions": [
            {
                "material": r["material"],
                "actual_Eg": r["actual_Eg"],
                "predicted_Eg": r["predicted_Eg"],
                "error_Eg": r["error_Eg"],
                "sample_group": r["sample_group"],
            }
            for r in sorted(all_eg_rows, key=lambda r: abs(r["error_Eg"]))[:5]
        ],
        "holdout_retrain": holdout,
        "scaps_chi_rows": scaps_chi_rows,
        "rows": benchmark_rows,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_rows = []
    for r in benchmark_rows:
        csv_rows.append(
            {
                "material": r["material"],
                "actual_Eg": r["actual_Eg"],
                "predicted_Eg": r["predicted_Eg"],
                "error_Eg": r["error_Eg"],
                "actual_chi": r["actual_chi"],
                "predicted_chi": r["predicted_chi"],
                "error_chi": r["error_chi"],
                "source": r["source"],
                "sample_group": r["sample_group"],
            }
        )
    for r in scaps_chi_rows:
        if r["material"] not in {x["material"] for x in csv_rows}:
            csv_rows.append(
                {
                    "material": r["material"],
                    "actual_Eg": r["actual_Eg"],
                    "predicted_Eg": r["predicted_Eg"],
                    "error_Eg": r["error_Eg"],
                    "actual_chi": r["actual_chi"],
                    "predicted_chi": r["predicted_chi"],
                    "error_chi": r["error_chi"],
                    "source": r["source"],
                    "sample_group": r["sample_group"],
                }
            )

    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(render_benchmark_markdown(report), encoding="utf-8")

    m = report["metrics"]
    print("=== Perovskite Prediction Benchmark (ML, no lookup, no LLM) ===")
    print(f"Random sample (n={len(rand_rows)}) — predict_eg:")
    print(f"  MAE={m['random_sample_40']['predict_eg_regressor']['MAE']:.4f} eV, "
          f"RMSE={m['random_sample_40']['predict_eg_regressor']['RMSE']:.4f} eV, "
          f"R2={m['random_sample_40']['predict_eg_regressor']['R2']:.4f}")
    print(f"External holdout (n={len(ext_rows)}) — predict_eg:")
    print(f"  MAE={m['external_literature_holdout']['predict_eg_regressor']['MAE']:.4f} eV, "
          f"RMSE={m['external_literature_holdout']['predict_eg_regressor']['RMSE']:.4f} eV, "
          f"R2={m['external_literature_holdout']['predict_eg_regressor']['R2']:.4f}")
    print(f"SCAPS literature chi (n={len(chi_with_gt)}):")
    print(f"  MAE={m['scaps_literature_chi']['MAE']:.4f} eV")
    print(f"Holdout retrain 20% (full library n={holdout['n_test']}):")
    print(f"  MAE={holdout['metrics']['MAE']:.4f} eV, R2={holdout['metrics']['R2']:.4f}")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
