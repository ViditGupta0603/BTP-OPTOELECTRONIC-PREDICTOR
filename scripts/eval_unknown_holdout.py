"""Force formula/rules path on browser set (ignore lookup) — unknown-material accuracy."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formula_estimator import estimate_eg_chi  # noqa: E402

DATA = ROOT / "data"
TEST_CSV = DATA / "browser_random_perovskite_test_set.csv"
OUT_JSON = DATA / "unknown_material_holdout_report.json"


def metrics(yt, yp):
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    hit03 = float(np.mean(np.abs(err) < 0.3))
    hit02 = float(np.mean(np.abs(err) < 0.2))
    return {
        "n": int(len(yt)),
        "MAE": round(mae, 4),
        "hit_0.2": round(hit02, 4),
        "hit_0.3": round(hit03, 4),
    }


def main() -> None:
    df = pd.read_csv(TEST_CSV)
    rows = []
    for _, r in df.iterrows():
        mat = str(r["material"]).strip()
        actual = float(r["actual_Eg_eV"])
        est = estimate_eg_chi(mat, "absorber")
        pred = float(est["Eg_eV"])
        rows.append(
            {
                "material": mat,
                "actual": actual,
                "pred": pred,
                "abs_err": abs(pred - actual),
                "source": est.get("source"),
                "family_id": est.get("family_id"),
                "confidence": est.get("confidence"),
            }
        )
    out = pd.DataFrame(rows)
    snge = out["material"].str.contains(r"SnI3|GeI3|SnBr|SnCl|GeBr", regex=True)
    mixed = out["material"].str.contains(r"FA0\.|0\.\d", regex=True)
    vac = out["family_id"] == "vacancy_ordered_a2bx6"
    abx3 = out["family_id"].str.startswith("abx3")

    report = {
        "all": metrics(out["actual"].to_numpy(), out["pred"].to_numpy()),
        "abx3_families": metrics(
            out.loc[abx3, "actual"].to_numpy(), out.loc[abx3, "pred"].to_numpy()
        ),
        "sn_ge_abx3": metrics(
            out.loc[snge, "actual"].to_numpy(), out.loc[snge, "pred"].to_numpy()
        ),
        "vacancy_a2bx6": metrics(
            out.loc[vac, "actual"].to_numpy(), out.loc[vac, "pred"].to_numpy()
        ),
        "mixed_a_or_fractional": metrics(
            out.loc[mixed, "actual"].to_numpy(), out.loc[mixed, "pred"].to_numpy()
        ),
        "rows": rows,
        "before_reference": {
            "note": "Prior browser report (lookup→ML, Jul 16) / iterative holdout",
            "browser_all_MAE": 0.7869,
            "browser_unseen_MAE": 0.8386,
            "browser_hit_0.3": 0.364,
            "sn_ge_unseen_MAE": 0.3414,
            "iterative_ml_holdout_MAE": 0.6936,
            "iterative_ml_holdout_hit_0.3": 0.25,
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
