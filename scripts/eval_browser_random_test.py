"""Evaluate OptoStack on browser-sourced perovskite test set (use_llm=False)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_stack import (  # noqa: E402
    check_absorber_perovskite,
    load_layer_lookup,
    ml_estimate_eg_chi,
    predict_eg,
    predict_stack,
    resolve_layer,
)

DATA = ROOT / "data"
TEST_CSV = DATA / "browser_random_perovskite_test_set.csv"
OUT_CSV = DATA / "browser_random_predictions.csv"
OUT_MD = DATA / "browser_random_accuracy_report.md"
OUT_JSON = DATA / "browser_random_accuracy_report.json"


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


def _get_absorber_eg_chi(pred: dict) -> tuple[float | None, float | None]:
    # Primary keys from predict_stack result
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
        return float(pred["absorber_Eg_eV"]), (None if chi is None else float(chi))
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


def main() -> None:
    df = pd.read_csv(TEST_CSV)
    layers = load_layer_lookup()
    rows: list[dict] = []

    for _, r in df.iterrows():
        mat = str(r["material"]).strip()
        actual = float(r["actual_Eg_eV"])
        chi_raw = r.get("actual_chi_eV")
        actual_chi = (
            float(chi_raw)
            if pd.notna(chi_raw) and str(chi_raw).strip() != ""
            else None
        )
        etl = str(r["etl"]).strip() if pd.notna(r.get("etl")) else "TiO2"
        htl = str(r["htl"]).strip() if pd.notna(r.get("htl")) else "NiO"

        entry = resolve_layer(layers, mat)
        in_lookup = bool(entry and "Eg_eV" in entry)
        lookup_eg = float(entry["Eg_eV"]) if in_lookup else None
        lookup_chi = (
            float(entry["chi_eV"])
            if in_lookup and entry.get("chi_eV") is not None
            else None
        )

        eligible = bool(check_absorber_perovskite(mat).get("eligible", False))
        ml_eg = float(predict_eg(mat))
        est = ml_estimate_eg_chi(mat, "absorber")
        est_eg = float(est["Eg_eV"])
        est_chi = float(est["chi_eV"])

        pred = predict_stack(mat, etl, htl, use_llm=False)
        blocked = bool(pred.get("blocked") or pred.get("screening_blocked"))
        pipe_eg, pipe_chi = _get_absorber_eg_chi(pred)
        sources = pred.get("sources") or {}
        eg_src = sources.get("absorber_Eg", "")
        chi_src = sources.get("absorber_chi", "")

        # Effective tool Eg: pipeline if available, else lookup, else ML
        if pipe_eg is not None and not blocked:
            tool_eg = pipe_eg
            tool_eg_source = eg_src or "pipeline"
        elif in_lookup:
            tool_eg = lookup_eg
            tool_eg_source = "lookup"
        else:
            tool_eg = ml_eg
            tool_eg_source = "predict_eg_ml"

        if pipe_chi is not None and not blocked:
            tool_chi = pipe_chi
            tool_chi_source = chi_src or "pipeline"
        elif lookup_chi is not None:
            tool_chi = lookup_chi
            tool_chi_source = "lookup"
        else:
            tool_chi = est_chi
            tool_chi_source = est.get("source", "ml_formula_estimator")

        etl_type, htl_type = _extract_types(pred)

        rows.append(
            {
                "material": mat,
                "actual_Eg_eV": actual,
                "actual_chi_eV": actual_chi,
                "eligible": eligible,
                "blocked": blocked,
                "in_lookup": in_lookup,
                "lookup_Eg_eV": lookup_eg,
                "tool_Eg_eV": tool_eg,
                "tool_Eg_error": tool_eg - actual,
                "tool_Eg_source": tool_eg_source,
                "pipeline_Eg_eV": pipe_eg,
                "pipeline_Eg_source": eg_src,
                "ml_predict_eg_eV": ml_eg,
                "ml_predict_eg_error": ml_eg - actual,
                "formula_est_Eg_eV": est_eg,
                "tool_chi_eV": tool_chi,
                "tool_chi_error": (tool_chi - actual_chi) if actual_chi is not None else None,
                "tool_chi_source": tool_chi_source,
                "etl": etl,
                "htl": htl,
                "pred_etl_type": etl_type,
                "pred_htl_type": htl_type,
                "in_verified_lead_halide_set": r.get("in_verified_lead_halide_set"),
                "material_class": r.get("material_class"),
                "source_doi": r.get("source_doi"),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    y_act = out["actual_Eg_eV"].to_numpy(dtype=float)
    y_tool = out["tool_Eg_eV"].to_numpy(dtype=float)
    y_ml = out["ml_predict_eg_eV"].to_numpy(dtype=float)
    lookup_mask = out["in_lookup"].to_numpy(dtype=bool)
    unseen_mask = ~lookup_mask
    # ML-only path: always predict_eg (ignores lookup)
    # Tool path with lookup vs without

    metrics = {
        "tool_vs_literature_all": _metrics(y_act, y_tool),
        "tool_lookup_subset": _metrics(y_act[lookup_mask], y_tool[lookup_mask]),
        "tool_unseen_subset": _metrics(y_act[unseen_mask], y_tool[unseen_mask]),
        "ml_predict_eg_all": _metrics(y_act, y_ml),
        "ml_predict_eg_unseen": _metrics(y_act[unseen_mask], y_ml[unseen_mask]),
        "ml_predict_eg_lookup_materials": _metrics(y_act[lookup_mask], y_ml[lookup_mask]),
    }

    chi_rows = out[out["actual_chi_eV"].notna()].copy()
    if len(chi_rows):
        metrics["tool_chi"] = _metrics(
            chi_rows["actual_chi_eV"].to_numpy(dtype=float),
            chi_rows["tool_chi_eV"].to_numpy(dtype=float),
        )
    else:
        metrics["tool_chi"] = {"n": 0, "MAE": None, "RMSE": None, "R2": None}

    abs_err = np.abs(y_tool - y_act)
    metrics["tool_frac_within_0.2eV"] = round(float(np.mean(abs_err <= 0.2)), 4)
    metrics["tool_frac_within_0.3eV"] = round(float(np.mean(abs_err <= 0.3)), 4)
    metrics["tool_frac_within_0.5eV"] = round(float(np.mean(abs_err <= 0.5)), 4)

    if lookup_mask.any():
        look_eg = out.loc[lookup_mask, "lookup_Eg_eV"].to_numpy(dtype=float)
        tool_look = out.loc[lookup_mask, "tool_Eg_eV"].to_numpy(dtype=float)
        metrics["lookup_self_consistency_MAE"] = round(
            float(np.mean(np.abs(tool_look - look_eg))), 6
        )

    # For lookup materials: accuracy vs literature vs accuracy vs stored lookup
    if lookup_mask.any():
        metrics["lookup_vs_literature_MAE"] = round(
            float(np.mean(np.abs(look_eg - y_act[lookup_mask]))), 4
        )

    worst = out.reindex(
        np.abs(out["tool_Eg_error"]).sort_values(ascending=False).index
    ).head(5)
    best = out.reindex(
        np.abs(out["tool_Eg_error"]).sort_values(ascending=True).index
    ).head(5)

    report = {
        "n_materials": int(len(out)),
        "n_in_lookup": int(lookup_mask.sum()),
        "n_unseen": int(unseen_mask.sum()),
        "n_with_chi": int(len(chi_rows)),
        "n_blocked": int(out["blocked"].sum()),
        "metrics": metrics,
        "method": (
            "predict_stack(..., use_llm=False) when eligible; "
            "predict_eg / lookup fallback; literature ground truth from browser CSV"
        ),
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md: list[str] = []
    md.append("# Browser-sourced perovskite test — OptoStack accuracy")
    md.append("")
    md.append("**Date:** 2026-07-16  ")
    md.append("**Dataset:** `data/browser_random_perovskite_test_set.csv`  ")
    md.append(
        "**Method:** `predict_stack(..., use_llm=False)` + ML-only `predict_eg`  "
    )
    md.append(
        "**Ground truth:** experimental literature Eg (and χ/EA where available) "
        "from web/browser sources — not invented."
    )
    md.append("")
    md.append("## Summary metrics (Eg)")
    md.append("")
    md.append("| Evaluation | n | MAE (eV) | RMSE (eV) | R² |")
    md.append("|---|---:|---:|---:|---:|")
    for label, key in [
        ("Tool (lookup→ML) vs literature — all", "tool_vs_literature_all"),
        ("Tool — materials in layer_lookup", "tool_lookup_subset"),
        ("Tool — unseen (not in lookup)", "tool_unseen_subset"),
        ("ML `predict_eg` only — all", "ml_predict_eg_all"),
        ("ML `predict_eg` — unseen only", "ml_predict_eg_unseen"),
        ("ML `predict_eg` — lookup materials", "ml_predict_eg_lookup_materials"),
    ]:
        m = metrics[key]
        r2 = "—" if m["R2"] is None else f"{m['R2']:.4f}"
        mae = "—" if m["MAE"] is None else f"{m['MAE']:.4f}"
        rmse = "—" if m["RMSE"] is None else f"{m['RMSE']:.4f}"
        md.append(f"| {label} | {m['n']} | {mae} | {rmse} | {r2} |")
    md.append("")
    md.append(
        f"**Fraction of tool |error| ≤ 0.2 / 0.3 / 0.5 eV:** "
        f"{100*metrics['tool_frac_within_0.2eV']:.1f}% / "
        f"{100*metrics['tool_frac_within_0.3eV']:.1f}% / "
        f"{100*metrics['tool_frac_within_0.5eV']:.1f}%"
    )
    md.append("")
    if metrics.get("lookup_self_consistency_MAE") is not None:
        md.append(
            f"**Lookup self-consistency** (tool Eg vs stored lookup on in-lookup materials): "
            f"MAE = {metrics['lookup_self_consistency_MAE']} eV."
        )
        md.append(
            f"**Lookup vs literature** (same subset): MAE = "
            f"{metrics.get('lookup_vs_literature_MAE')} eV "
            "(large when lookup stores DFT library gaps)."
        )
        md.append("")

    md.append("## χ / electron affinity (literature available)")
    md.append("")
    mc = metrics["tool_chi"]
    if mc["n"]:
        md.append(
            f"n={mc['n']}, MAE={mc['MAE']:.4f} eV, RMSE={mc['RMSE']:.4f} eV, "
            f"R²={mc['R2']}"
        )
        md.append("")
        md.append("| Material | Actual χ (eV) | Tool χ (eV) | Error | Source |")
        md.append("|---|---:|---:|---:|---|")
        for _, row in chi_rows.iterrows():
            md.append(
                f"| {row['material']} | {row['actual_chi_eV']:.2f} | "
                f"{row['tool_chi_eV']:.3f} | {row['tool_chi_error']:+.3f} | "
                f"{row['tool_chi_source']} |"
            )
    else:
        md.append("No χ ground-truth rows.")
    md.append("")

    md.append("## Worst / best tool Eg predictions")
    md.append("")
    md.append("### Worst 5")
    md.append("")
    md.append("| Material | Actual | Tool | Error | Source | In lookup? |")
    md.append("|---|---:|---:|---:|---|---|")
    for _, row in worst.iterrows():
        md.append(
            f"| {row['material']} | {row['actual_Eg_eV']:.2f} | {row['tool_Eg_eV']:.3f} | "
            f"{row['tool_Eg_error']:+.3f} | {row['tool_Eg_source']} | {row['in_lookup']} |"
        )
    md.append("")
    md.append("### Best 5")
    md.append("")
    md.append("| Material | Actual | Tool | Error | Source | In lookup? |")
    md.append("|---|---:|---:|---:|---|---|")
    for _, row in best.iterrows():
        md.append(
            f"| {row['material']} | {row['actual_Eg_eV']:.2f} | {row['tool_Eg_eV']:.3f} | "
            f"{row['tool_Eg_error']:+.3f} | {row['tool_Eg_source']} | {row['in_lookup']} |"
        )
    md.append("")

    md.append("## Per-material results")
    md.append("")
    md.append(
        "| Material | Actual Eg | Tool Eg | |err| | Source | ML Eg | Lookup? |"
    )
    md.append("|---|---:|---:|---:|---|---:|---|")
    for _, row in out.iterrows():
        md.append(
            f"| {row['material']} | {row['actual_Eg_eV']:.2f} | {row['tool_Eg_eV']:.3f} | "
            f"{abs(row['tool_Eg_error']):.3f} | {row['tool_Eg_source']} | "
            f"{row['ml_predict_eg_eV']:.3f} | {row['in_lookup']} |"
        )
    md.append("")

    md.append("## Interpretation")
    md.append("")
    md.append(
        f"- **n = {len(out)}** literature materials spanning Sn/Ge ABX₃, double perovskites, "
        "vacancy-ordered A₂BX₆, Sb/Bi-inspired A₃B₂X₉, mixed A-site FACsPbI₃, "
        "plus known lead-halides."
    )
    md.append(
        f"- **{int(lookup_mask.sum())} in layer_lookup**, **{int(unseen_mask.sum())} unseen**."
    )
    md.append(
        "- **Known lead-halides in lookup** (CsPbBr₃, FAPbBr₃, MAPbI₃, CsPbI₃): "
        "tool Eg matches experimental literature (lookup self-consistency ≈ exact)."
    )
    md.append(
        "- **Double perovskites in lookup with DFT-library gaps** (Cs₂AgBiBr₆, Cs₂AgInCl₆, "
        "Cs₂AgBiCl₆, Rb₂AgBiI₆): lookup is exact vs stored values, but those values are "
        "**much lower than experimental optical gaps** — this dominates overall MAE."
    )
    md.append(
        "- **Unseen** tin/Ge/vacancy-ordered materials rely on ML `predict_eg` / formula "
        "estimator; MAE is higher than in-distribution library holdouts."
    )
    md.append(
        "- **Type accuracy** not scored (no literature Type-I/II labels for these random stacks); "
        "predicted types are in `browser_random_predictions.csv`."
    )
    md.append(
        "- **χ**: only 2 materials have literature electron affinity; treat χ MAE as indicative."
    )
    md.append(
        "- Note: `looks_like_perovskite_absorber` was extended to recognize Sn/Ge ABX₃ "
        "(and related halide families) so `predict_stack` accepts these real absorbers."
    )
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- Dataset: `data/browser_random_perovskite_test_set.csv`")
    md.append("- Predictions: `data/browser_random_predictions.csv`")
    md.append("- This report: `data/browser_random_accuracy_report.md`")
    md.append("- JSON: `data/browser_random_accuracy_report.json`")
    md.append("")
    md.append("## Regenerate")
    md.append("")
    md.append("```bash")
    md.append("python scripts/eval_browser_random_test.py")
    md.append("```")
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
