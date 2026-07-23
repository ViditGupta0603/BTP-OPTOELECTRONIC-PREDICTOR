"""Evaluate OptoStack on perovskite_test_set_literature (use_llm=False)."""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_stack import (  # noqa: E402
    check_absorber_perovskite,
    load_layer_lookup,
    ml_estimate_eg_chi,
    normalize_material_name,
    predict_eg,
    predict_stack,
    resolve_layer,
)

DATA = ROOT / "data"
TEST_CSV = DATA / "perovskite_test_set_literature.csv"
OUT_CSV = DATA / "perovskite_test_set_literature_predictions.csv"
OUT_MD = DATA / "perovskite_test_set_literature_accuracy_report.md"
OUT_JSON = DATA / "perovskite_test_set_literature_accuracy_report.json"

DEFAULT_ETL = "TiO2"
DEFAULT_HTL = "Spiro-MeOTAD"

_SKIP_ABS = {
    "NOT_MATERIAL_SPECIFIC",
    "NOT_APPLICABLE",
    "NOT_SPECIFIED",
    "",
}

_BAD_LAYER = {
    "NOT_SPECIFIED",
    "NOT_APPLICABLE",
    "NOT_STATED",
    "",
}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = int(len(yt))
    if n == 0:
        return {"n": 0, "MAE": None, "RMSE": None, "R2": None}
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return {
        "n": n,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": None if r2 is None else round(r2, 4),
    }


def _clean_absorber(raw: str) -> str | None:
    s = (raw or "").strip()
    if not s or s.upper() in _SKIP_ABS:
        return None
    # Class / multi-material rows — not a single formula
    if " / " in s and "class" in s.lower():
        return None
    if s.lower().endswith("class (sn-based)") or "class (" in s.lower():
        return None
    # Prefer parenthetical alias when present: "CH3NH3PbI3 (MAPbI3)" → MAPbI3
    # Skip phase/composition notes: (B-gamma), (I,Br), nested stoichiometry, etc.
    m = re.search(r"\(([^)]+)\)\s*$", s)
    if m:
        alias = m.group(1).strip()
        looks_like_formula = bool(
            re.fullmatch(r"[A-Za-z0-9]+", alias)
            and re.search(r"[A-Za-z]", alias)
            and not re.search(r"\d\.\d", alias)
        )
        if looks_like_formula and len(alias) < 40 and "/" not in alias:
            # Common aliases like MAPbI3 / CBTS — not phase labels
            if alias.lower() not in {"b-gamma", "experimental"}:
                return alias
        # Keep formula before trailing note parentheses when alias is not usable
        if " (" in s:
            s = s.split(" (", 1)[0].strip()
    # Drop trailing notes after comma if formula-like prefix
    if "," in s and not s.startswith("("):
        head = s.split(",", 1)[0].strip()
        if re.search(r"[A-Z][a-z]?\d", head) or "Pb" in head or "Sn" in head:
            s = head
    return s


def _parse_eg(raw: str) -> tuple[float | None, str]:
    """Return (eg_eV, note). Prefer experimental / primary numeric when dual values."""
    s = (raw or "").strip()
    if not s:
        return None, "empty"
    up = s.upper()
    if any(
        tok in up
        for tok in (
            "NOT_STATED",
            "NOT_APPLICABLE",
            "NOT_MATERIAL",
        )
    ):
        return None, "not_stated"

    # Prefer experimental when dual: "2.30 (experimental, ...)"
    m_exp = re.search(
        r"(\d+(?:\.\d+)?)\s*\(\s*experimental", s, flags=re.IGNORECASE
    )
    if m_exp:
        return float(m_exp.group(1)), "experimental_preferred"

    # Prefer HSE over PBE: "1.40 (HSE) / 2.35 (PBE)"
    m_hse = re.search(
        r"(\d+(?:\.\d+)?)\s*\(\s*HSE", s, flags=re.IGNORECASE
    )
    if m_hse:
        return float(m_hse.group(1)), "hse_preferred"

    # Prefer single crystal over thin film when both given
    m_sc = re.search(
        r"(\d+(?:\.\d+)?)\s*\(\s*single crystal", s, flags=re.IGNORECASE
    )
    if m_sc:
        return float(m_sc.group(1)), "single_crystal_preferred"

    # Range midpoint: "1.2-1.4 (range)"
    m_rng = re.search(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)", s)
    if m_rng and "range" in s.lower():
        a, b = float(m_rng.group(1)), float(m_rng.group(2))
        return (a + b) / 2.0, "range_midpoint"

    # First plain float
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), "first_numeric"
    return None, "unparsed"


def _pick_layer(raw: str, default: str) -> tuple[str, str]:
    """Pick a concrete ETL/HTL name; return (name, note)."""
    s = (raw or "").strip()
    if not s or s.upper() in _BAD_LAYER or s.upper().startswith("NOT_"):
        return default, "defaulted"
    # Class lists: take first concrete token before /
    if " / " in s or "(class)" in s.lower():
        first = s.split("/")[0].strip()
        first = re.sub(r"\(.*?\)", "", first).strip()
        if first:
            return first, "first_from_class"
    # Parenthetical expansion: "CBTS (Cu2BaSnS4)" → CBTS (lookup may know either)
    m = re.match(r"^([A-Za-z0-9.]+)(?:\s*\(([^)]+)\))?$", s)
    if m:
        return m.group(1), "as_stated"
    # Strip trailing notes
    head = s.split(",")[0].strip()
    head = re.sub(r"\(.*?\)", "", head).strip()
    return head or default, "cleaned" if head else "defaulted"


def _get_absorber_eg_chi(pred: dict) -> tuple[float | None, float | None]:
    if pred.get("absorber_band_gap_eV") is not None:
        chi = pred.get("absorber_chi_eV")
        return float(pred["absorber_band_gap_eV"]), (
            None if chi is None else float(chi)
        )
    abs_block = pred.get("absorber")
    if isinstance(abs_block, dict) and abs_block.get("Eg_eV") is not None:
        chi = abs_block.get("chi_eV")
        return float(abs_block["Eg_eV"]), (None if chi is None else float(chi))
    if pred.get("absorber_Eg_eV") is not None:
        chi = pred.get("absorber_chi_eV")
        return float(pred["absorber_Eg_eV"]), (
            None if chi is None else float(chi)
        )
    return None, None


def _extract_types(pred: dict) -> tuple[str | None, str | None]:
    etl = pred.get("absorber_etl_type") or pred.get("Type_absorber_etl")
    htl = pred.get("absorber_htl_type") or pred.get("Type_absorber_htl")
    for side, bucket in (("etl", "absorber_etl"), ("htl", "absorber_htl")):
        block = pred.get(bucket)
        if isinstance(block, dict) and block.get("type"):
            if side == "etl":
                etl = etl or str(block["type"])
            else:
                htl = htl or str(block["type"])
    opto = pred.get("optoelectronic") or {}
    if isinstance(opto, dict):
        etl = etl or (str(opto["etl_type"]) if opto.get("etl_type") else None)
        htl = htl or (str(opto["htl_type"]) if opto.get("htl_type") else None)
    return etl, htl


def _load_rows() -> list[dict]:
    with TEST_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    raw_rows = _load_rows()
    layers = load_layer_lookup()
    rows: list[dict] = []
    skipped: list[dict] = []

    for i, r in enumerate(raw_rows):
        abs_raw = str(r.get("material_absorber") or "").strip()
        eg_raw = str(r.get("absorber_band_gap_eV") or "").strip()
        etl_raw = str(r.get("material_etl") or "").strip()
        htl_raw = str(r.get("material_htl") or "").strip()

        absorber = _clean_absorber(abs_raw)
        actual_eg, eg_note = _parse_eg(eg_raw)
        etl, etl_note = _pick_layer(etl_raw, DEFAULT_ETL)
        htl, htl_note = _pick_layer(htl_raw, DEFAULT_HTL)

        if absorber is None:
            skipped.append(
                {
                    "row": i,
                    "reason": "no_single_absorber",
                    "material_absorber": abs_raw,
                    "absorber_band_gap_eV": eg_raw,
                }
            )
            continue
        if actual_eg is None:
            skipped.append(
                {
                    "row": i,
                    "reason": f"no_eg:{eg_note}",
                    "material_absorber": abs_raw,
                    "absorber_band_gap_eV": eg_raw,
                }
            )
            continue

        mat = normalize_material_name(absorber)
        entry = resolve_layer(layers, mat)
        in_lookup = bool(entry and "Eg_eV" in entry)
        lookup_eg = float(entry["Eg_eV"]) if in_lookup else None

        eligible = bool(check_absorber_perovskite(mat).get("eligible", False))
        ml_eg = float(predict_eg(mat))
        est = ml_estimate_eg_chi(mat, "absorber")
        est_eg = float(est["Eg_eV"])

        pred = predict_stack(mat, etl, htl, use_llm=False)
        blocked = bool(pred.get("blocked") or pred.get("screening_blocked"))
        pipe_eg, _pipe_chi = _get_absorber_eg_chi(pred)
        sources = pred.get("sources") or {}
        eg_src = sources.get("absorber_Eg", "")
        etl_type, htl_type = _extract_types(pred)
        suitability = pred.get("suitability") or pred.get("stack_suitability")
        if isinstance(suitability, dict):
            suitability = suitability.get("label") or suitability.get("status")

        if pipe_eg is not None and not blocked:
            tool_eg = pipe_eg
            tool_eg_source = eg_src or "pipeline"
        elif in_lookup:
            tool_eg = lookup_eg
            tool_eg_source = "lookup"
        else:
            tool_eg = ml_eg
            tool_eg_source = "predict_eg_ml"

        rows.append(
            {
                "row_idx": i,
                "material_raw": abs_raw,
                "material": mat,
                "actual_Eg_eV": actual_eg,
                "eg_parse_note": eg_note,
                "eligible": eligible,
                "blocked": blocked,
                "in_lookup": in_lookup,
                "lookup_Eg_eV": lookup_eg,
                "tool_Eg_eV": tool_eg,
                "tool_Eg_error": float(tool_eg) - actual_eg,
                "tool_Eg_source": tool_eg_source,
                "pipeline_Eg_eV": pipe_eg,
                "pipeline_Eg_source": eg_src,
                "ml_predict_eg_eV": ml_eg,
                "ml_predict_eg_error": ml_eg - actual_eg,
                "formula_est_Eg_eV": est_eg,
                "etl_raw": etl_raw,
                "htl_raw": htl_raw,
                "etl": etl,
                "htl": htl,
                "etl_note": etl_note,
                "htl_note": htl_note,
                "pred_etl_type": etl_type,
                "pred_htl_type": htl_type,
                "pred_suitability": suitability,
                "source_doi": r.get("source_doi"),
                "source_paper": r.get("source_paper"),
                "value_type": r.get("value_type"),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # Deduplicate by material for alternate "unique absorbers" view (mean actual)
    # Primary metrics: all scored rows (including duplicate stacks with same Eg)
    y_act = out["actual_Eg_eV"].to_numpy(dtype=float)
    y_tool = out["tool_Eg_eV"].to_numpy(dtype=float)
    y_ml = out["ml_predict_eg_eV"].to_numpy(dtype=float)
    lookup_mask = out["in_lookup"].to_numpy(dtype=bool)
    unseen_mask = ~lookup_mask

    # Unique-material metrics (first occurrence per normalized material)
    uniq = out.drop_duplicates(subset=["material"], keep="first")
    y_act_u = uniq["actual_Eg_eV"].to_numpy(dtype=float)
    y_tool_u = uniq["tool_Eg_eV"].to_numpy(dtype=float)
    y_ml_u = uniq["ml_predict_eg_eV"].to_numpy(dtype=float)
    lookup_u = uniq["in_lookup"].to_numpy(dtype=bool)

    abs_err = np.abs(y_tool - y_act)
    abs_err_u = np.abs(y_tool_u - y_act_u)

    metrics = {
        "tool_vs_literature_all_rows": _metrics(y_act, y_tool),
        "tool_lookup_subset": _metrics(y_act[lookup_mask], y_tool[lookup_mask]),
        "tool_unseen_subset": _metrics(y_act[unseen_mask], y_tool[unseen_mask]),
        "ml_predict_eg_all_rows": _metrics(y_act, y_ml),
        "ml_predict_eg_unseen": _metrics(y_act[unseen_mask], y_ml[unseen_mask]),
        "ml_predict_eg_lookup_materials": _metrics(
            y_act[lookup_mask], y_ml[lookup_mask]
        ),
        "tool_unique_materials": _metrics(y_act_u, y_tool_u),
        "ml_unique_materials": _metrics(y_act_u, y_ml_u),
        "tool_unique_lookup": _metrics(y_act_u[lookup_u], y_tool_u[lookup_u]),
        "tool_unique_unseen": _metrics(y_act_u[~lookup_u], y_tool_u[~lookup_u]),
        "tool_frac_within_0.2eV_rows": round(float(np.mean(abs_err <= 0.2)), 4)
        if len(abs_err)
        else None,
        "tool_frac_within_0.3eV_rows": round(float(np.mean(abs_err <= 0.3)), 4)
        if len(abs_err)
        else None,
        "tool_frac_within_0.5eV_rows": round(float(np.mean(abs_err <= 0.5)), 4)
        if len(abs_err)
        else None,
        "tool_frac_within_0.2eV_unique": round(float(np.mean(abs_err_u <= 0.2)), 4)
        if len(abs_err_u)
        else None,
        "tool_frac_within_0.3eV_unique": round(float(np.mean(abs_err_u <= 0.3)), 4)
        if len(abs_err_u)
        else None,
    }

    if lookup_mask.any():
        look_eg = out.loc[lookup_mask, "lookup_Eg_eV"].to_numpy(dtype=float)
        tool_look = out.loc[lookup_mask, "tool_Eg_eV"].to_numpy(dtype=float)
        metrics["lookup_self_consistency_MAE"] = round(
            float(np.mean(np.abs(tool_look - look_eg))), 6
        )
        metrics["lookup_vs_literature_MAE"] = round(
            float(np.mean(np.abs(look_eg - y_act[lookup_mask]))), 4
        )

    train_meta = {}
    eval_report = {}
    tm_path = DATA / "models" / "train_meta.json"
    er_path = DATA / "model_eval_report.json"
    if tm_path.exists():
        train_meta = json.loads(tm_path.read_text(encoding="utf-8"))
    if er_path.exists():
        eval_report = json.loads(er_path.read_text(encoding="utf-8"))

    worst = out.reindex(
        np.abs(out["tool_Eg_error"]).sort_values(ascending=False).index
    ).head(5)
    best = out.reindex(
        np.abs(out["tool_Eg_error"]).sort_values(ascending=True).index
    ).head(5)

    n_raw = len(raw_rows)
    n_scored = int(len(out))
    n_pred = int(out["tool_Eg_eV"].notna().sum()) if len(out) else 0
    n_known_eg = n_scored  # scored rows all have known Eg
    report = {
        "date": str(date.today()),
        "dataset": str(TEST_CSV.relative_to(ROOT)),
        "method": "predict_stack(..., use_llm=False); formula/ML fallback",
        "n_raw_rows": n_raw,
        "n_skipped": len(skipped),
        "n_scored_rows": n_scored,
        "n_unique_materials": int(len(uniq)),
        "n_predicted": n_pred,
        "pct_predicted_of_scored": round(100.0 * n_pred / n_scored, 1)
        if n_scored
        else 0.0,
        "pct_scored_of_raw": round(100.0 * n_scored / n_raw, 1) if n_raw else 0.0,
        "n_in_lookup": int(lookup_mask.sum()) if len(out) else 0,
        "n_unseen": int(unseen_mask.sum()) if len(out) else 0,
        "n_blocked": int(out["blocked"].sum()) if len(out) else 0,
        "type_accuracy": None,
        "type_accuracy_note": (
            "No ground-truth Type-I/II/III labels in perovskite_test_set_literature.csv"
        ),
        "suitability_accuracy": None,
        "suitability_accuracy_note": (
            "No ground-truth suitability labels in this dataset"
        ),
        "skipped": skipped,
        "metrics": metrics,
        "train_meta_summary": {
            "layers": train_meta.get("layers"),
            "eg_holdout_mae_eV": (train_meta.get("eg") or {}).get("holdout_mae_eV"),
            "eg_cv_mae_eV": (train_meta.get("eg") or {}).get("cv_mae_eV"),
            "eg_n": (train_meta.get("eg") or {}).get("n"),
            "type_etl_holdout_acc": (train_meta.get("type") or {}).get(
                "etl_holdout_acc"
            ),
            "type_htl_holdout_acc": (train_meta.get("type") or {}).get(
                "htl_holdout_acc"
            ),
            "formula_eg_holdout_mae_eV": (train_meta.get("formula") or {}).get(
                "eg_holdout_mae_eV"
            ),
        },
        "model_eval_summary": {
            "eg_holdout_MAE_eV": (
                (eval_report.get("eg_regressor") or {})
                .get("holdout_20pct", {})
                .get("MAE_eV")
            ),
            "eg_holdout_RMSE_eV": (
                (eval_report.get("eg_regressor") or {})
                .get("holdout_20pct", {})
                .get("RMSE_eV")
            ),
            "eg_holdout_R2": (
                (eval_report.get("eg_regressor") or {})
                .get("holdout_20pct", {})
                .get("R2")
            ),
            "type_etl_holdout_acc": (
                (eval_report.get("type_etl_classifier") or {})
                .get("holdout_20pct", {})
                .get("accuracy")
            ),
            "type_htl_holdout_acc": (
                (eval_report.get("type_htl_classifier") or {})
                .get("holdout_20pct", {})
                .get("accuracy")
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _fmt_m(m: dict) -> tuple[str, str, str]:
        mae = "—" if m["MAE"] is None else f"{m['MAE']:.4f}"
        rmse = "—" if m["RMSE"] is None else f"{m['RMSE']:.4f}"
        r2 = "—" if m["R2"] is None else f"{m['R2']:.4f}"
        return mae, rmse, r2

    md: list[str] = []
    md.append("# Literature perovskite test set — OptoStack accuracy")
    md.append("")
    md.append(f"**Date:** {date.today()}  ")
    md.append(f"**Dataset:** `{TEST_CSV.relative_to(ROOT).as_posix()}`  ")
    md.append(
        "**Method:** `predict_stack(..., use_llm=False)` + family/Vegard+ML "
        "formula estimator  "
    )
    md.append(
        "**Ground truth:** literature-extracted absorber Eg from "
        "`perovskite_test_set_literature.csv` (DOI / paper quotes).  "
    )
    md.append("")
    md.append("## Coverage")
    md.append("")
    md.append(f"| | n |")
    md.append("|---|---:|")
    md.append(f"| Raw CSV rows | {n_raw} |")
    md.append(f"| Skipped (no absorber / no Eg) | {len(skipped)} |")
    md.append(f"| Scored rows (known Eg + predicted) | {n_scored} |")
    md.append(f"| Unique absorbers (normalized) | {len(uniq)} |")
    md.append(f"| In `layer_lookup` | {int(lookup_mask.sum()) if len(out) else 0} |")
    md.append(f"| Unseen (not in lookup) | {int(unseen_mask.sum()) if len(out) else 0} |")
    md.append(f"| Blocked by perovskite screen | {int(out['blocked'].sum()) if len(out) else 0} |")
    md.append(
        f"| % predicted of scored | "
        f"{report['pct_predicted_of_scored']}% |"
    )
    md.append(
        f"| % scored of raw | {report['pct_scored_of_raw']}% |"
    )
    md.append("")
    md.append("### Skipped rows")
    md.append("")
    if skipped:
        md.append("| Row | Reason | Absorber | Eg raw |")
        md.append("|---:|---|---|---|")
        for s in skipped:
            md.append(
                f"| {s['row']} | {s['reason']} | "
                f"{s['material_absorber'][:50]} | "
                f"{str(s['absorber_band_gap_eV'])[:40]} |"
            )
    else:
        md.append("None.")
    md.append("")

    md.append("## Summary metrics (Eg)")
    md.append("")
    md.append("| Evaluation | n | MAE (eV) | RMSE (eV) | R² |")
    md.append("|---|---:|---:|---:|---:|")
    for label, key in [
        ("Tool vs literature — all scored rows", "tool_vs_literature_all_rows"),
        ("Tool — in layer_lookup", "tool_lookup_subset"),
        ("Tool — unseen (not in lookup)", "tool_unseen_subset"),
        ("Tool — unique absorbers", "tool_unique_materials"),
        ("Tool unique — lookup", "tool_unique_lookup"),
        ("Tool unique — unseen", "tool_unique_unseen"),
        ("ML `predict_eg` — all scored rows", "ml_predict_eg_all_rows"),
        ("ML `predict_eg` — unseen", "ml_predict_eg_unseen"),
        ("ML `predict_eg` — lookup materials", "ml_predict_eg_lookup_materials"),
        ("ML — unique absorbers", "ml_unique_materials"),
    ]:
        m = metrics[key]
        mae, rmse, r2 = _fmt_m(m)
        md.append(f"| {label} | {m['n']} | {mae} | {rmse} | {r2} |")
    md.append("")
    md.append(
        f"**Hit-rate tool |error| ≤ 0.2 / 0.3 / 0.5 eV (scored rows):** "
        f"{100 * (metrics['tool_frac_within_0.2eV_rows'] or 0):.1f}% / "
        f"{100 * (metrics['tool_frac_within_0.3eV_rows'] or 0):.1f}% / "
        f"{100 * (metrics['tool_frac_within_0.5eV_rows'] or 0):.1f}%"
    )
    md.append("")
    md.append(
        f"**Hit-rate (unique absorbers) |error| ≤ 0.2 / 0.3 eV:** "
        f"{100 * (metrics['tool_frac_within_0.2eV_unique'] or 0):.1f}% / "
        f"{100 * (metrics['tool_frac_within_0.3eV_unique'] or 0):.1f}%"
    )
    md.append("")
    if metrics.get("lookup_self_consistency_MAE") is not None:
        md.append(
            f"**Lookup self-consistency** (tool vs stored lookup): "
            f"MAE = {metrics['lookup_self_consistency_MAE']} eV.  "
        )
        md.append(
            f"**Lookup vs literature** (same subset): MAE = "
            f"{metrics.get('lookup_vs_literature_MAE')} eV."
        )
        md.append("")

    md.append("## Type / suitability")
    md.append("")
    md.append(
        "- **Type accuracy:** not scored — dataset has no Type-I/II/III "
        "ground-truth columns. Predicted types are in the predictions CSV."
    )
    md.append(
        "- **Suitability accuracy:** not scored — no suitability labels in dataset."
    )
    md.append("")

    md.append("## Model train / eval stats (reference)")
    md.append("")
    tms = report["train_meta_summary"]
    mes = report["model_eval_summary"]
    md.append("| Source | Metric | Value |")
    md.append("|---|---|---:|")
    md.append(f"| `train_meta.json` | layers in lookup | {tms.get('layers')} |")
    md.append(
        f"| `train_meta.json` | Eg holdout MAE (eV) | "
        f"{tms.get('eg_holdout_mae_eV')} |"
    )
    md.append(
        f"| `train_meta.json` | Eg CV MAE (eV) | {tms.get('eg_cv_mae_eV')} |"
    )
    md.append(f"| `train_meta.json` | Eg n | {tms.get('eg_n')} |")
    md.append(
        f"| `train_meta.json` | formula Eg holdout MAE | "
        f"{tms.get('formula_eg_holdout_mae_eV')} |"
    )
    md.append(
        f"| `train_meta.json` | Type ETL/HTL holdout acc | "
        f"{tms.get('type_etl_holdout_acc')} / {tms.get('type_htl_holdout_acc')} |"
    )
    md.append(
        f"| `model_eval_report.json` | Eg holdout MAE / RMSE / R² | "
        f"{mes.get('eg_holdout_MAE_eV')} / {mes.get('eg_holdout_RMSE_eV')} / "
        f"{mes.get('eg_holdout_R2')} |"
    )
    md.append(
        f"| `model_eval_report.json` | Type ETL/HTL holdout acc | "
        f"{mes.get('type_etl_holdout_acc')} / {mes.get('type_htl_holdout_acc')} |"
    )
    md.append("")

    md.append("## Worst / best tool Eg predictions (scored rows)")
    md.append("")
    md.append("### Worst 5")
    md.append("")
    md.append("| Material | Actual | Tool | Error | Source | Lookup? |")
    md.append("|---|---:|---:|---:|---|---|")
    for _, row in worst.iterrows():
        md.append(
            f"| {row['material']} | {row['actual_Eg_eV']:.3f} | "
            f"{row['tool_Eg_eV']:.3f} | {row['tool_Eg_error']:+.3f} | "
            f"{row['tool_Eg_source']} | {row['in_lookup']} |"
        )
    md.append("")
    md.append("### Best 5")
    md.append("")
    md.append("| Material | Actual | Tool | Error | Source | Lookup? |")
    md.append("|---|---:|---:|---:|---|---|")
    for _, row in best.iterrows():
        md.append(
            f"| {row['material']} | {row['actual_Eg_eV']:.3f} | "
            f"{row['tool_Eg_eV']:.3f} | {row['tool_Eg_error']:+.3f} | "
            f"{row['tool_Eg_source']} | {row['in_lookup']} |"
        )
    md.append("")

    md.append("## Per-row results")
    md.append("")
    md.append(
        "| # | Material | Actual Eg | Tool Eg | |err| | Source | ML Eg | "
        "Lookup? | ETL | HTL |"
    )
    md.append("|---:|---|---:|---:|---:|---|---:|---|---|---|")
    for _, row in out.iterrows():
        md.append(
            f"| {row['row_idx']} | {row['material']} | {row['actual_Eg_eV']:.3f} | "
            f"{row['tool_Eg_eV']:.3f} | {abs(row['tool_Eg_error']):.3f} | "
            f"{row['tool_Eg_source']} | {row['ml_predict_eg_eV']:.3f} | "
            f"{row['in_lookup']} | {row['etl']} | {row['htl']} |"
        )
    md.append("")

    md.append("## Notes")
    md.append("")
    md.append(
        f"- Default ETL/HTL when unspecified: `{DEFAULT_ETL}` / `{DEFAULT_HTL}` "
        "(Eg prediction does not depend on contacts; Type labels may)."
    )
    md.append(
        "- Dual Eg values: HSE preferred over PBE; experimental preferred when labeled; "
        "ranges use midpoint. Filled Eg rows cite `eg_fill_doi` / `completion_notes`."
    )
    md.append(
        "- Dataset completed 2026-07-23: previously skipped rows now have single-absorber "
        "formulas + literature Eg (see `gap_method` / `eg_fill_doi` columns)."
    )
    md.append("")
    md.append("## Files")
    md.append("")
    md.append(f"- Dataset: `{TEST_CSV.relative_to(ROOT).as_posix()}`")
    md.append(f"- Predictions: `{OUT_CSV.relative_to(ROOT).as_posix()}`")
    md.append(f"- This report: `{OUT_MD.relative_to(ROOT).as_posix()}`")
    md.append(f"- JSON: `{OUT_JSON.relative_to(ROOT).as_posix()}`")
    md.append("")
    md.append("## Regenerate")
    md.append("")
    md.append("```bash")
    md.append("python scripts/eval_literature_test_set.py")
    md.append("```")
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
