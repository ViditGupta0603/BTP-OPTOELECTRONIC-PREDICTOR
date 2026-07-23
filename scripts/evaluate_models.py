"""Evaluate OptoStack ML models with train/test splits.

Reports accuracy, F1, precision, recall (Type) and MAE/RMSE/R2 (Eg).

  python scripts/evaluate_models.py

Writes: data/model_eval_report.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_stack import _feature_frame  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "model_eval_report.json"


def eval_eg() -> dict:
    rows = list(csv.DictReader((DATA / "perovskite_absorber_library.csv").open(encoding="utf-8")))
    formulas = [r["material_absorber"] for r in rows]
    y = np.array([float(r["absorber_band_gap_eV"]) for r in rows])
    _, X = _feature_frame(formulas)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(
        n_estimators=300, max_depth=24, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    holdout = {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "MAE_eV": float(mean_absolute_error(yte, pred)),
        "RMSE_eV": float(np.sqrt(mean_squared_error(yte, pred))),
        "R2": float(r2_score(yte, pred)),
        "median_AE_eV": float(np.median(np.abs(yte - pred))),
    }
    cv_mae = -cross_val_score(reg, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_r2 = cross_val_score(reg, X, y, cv=5, scoring="r2", n_jobs=-1)
    return {
        "model": "RandomForestRegressor",
        "features": "bag-of-elements from formula",
        "dataset": "perovskite_absorber_library.csv",
        "n_total": int(len(y)),
        "holdout_20pct": holdout,
        "cv_5fold": {
            "folds": 5,
            "MAE_mean_eV": float(cv_mae.mean()),
            "MAE_std_eV": float(cv_mae.std()),
            "R2_mean": float(cv_r2.mean()),
            "R2_std": float(cv_r2.std()),
        },
    }


def eval_type(target: str) -> dict:
    rows: list[dict] = list(
        csv.DictReader((DATA / "perovskite_stack_dataset.csv").open(encoding="utf-8"))
    )
    scaps = DATA / "opto_literature_dataset_scaps_only.csv"
    if scaps.exists():
        rows.extend(csv.DictReader(scaps.open(encoding="utf-8")))

    X_mat, y, groups = [], [], []
    for r in rows:
        if not r.get(target):
            continue
        partner = r["material_etl"] if "etl" in target else r["material_htl"]
        eg_p = float(r["etl_band_gap_eV"] if "etl" in target else r["htl_band_gap_eV"])
        X_mat.append(
            {
                "absorber": r["material_absorber"],
                "partner": partner,
                "eg_a": float(r["absorber_band_gap_eV"]),
                "eg_p": eg_p,
            }
        )
        y.append(r[target])
        groups.append(r["material_absorber"])

    df = pd.DataFrame(X_mat)
    y_arr = np.array(y)
    groups_arr = np.array(groups)
    clf = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["absorber", "partner"]),
                        ("num", StandardScaler(), ["eg_a", "eg_p"]),
                    ]
                ),
            ),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]
    )

    Xtr, Xte, ytr, yte = train_test_split(
        df, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    labels = sorted(set(y_arr.tolist()))
    holdout = {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "accuracy": float(accuracy_score(yte, yp)),
        "f1_macro": float(f1_score(yte, yp, average="macro")),
        "f1_weighted": float(f1_score(yte, yp, average="weighted")),
        "precision_macro": float(precision_score(yte, yp, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(yte, yp, average="macro", zero_division=0)),
        "per_class": classification_report(yte, yp, output_dict=True, zero_division=0),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(yte, yp, labels=labels).tolist(),
        },
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in skf.split(df, y_arr):
        clf.fit(df.iloc[tr], y_arr[tr])
        p = clf.predict(df.iloc[te])
        accs.append(accuracy_score(y_arr[te], p))
        f1s.append(f1_score(y_arr[te], p, average="macro"))

    n_groups = len(set(groups_arr.tolist()))
    gkf = GroupKFold(n_splits=min(3, n_groups))
    g_acc, g_f1 = [], []
    for tr, te in gkf.split(df, y_arr, groups=groups_arr):
        if len(set(y_arr[tr].tolist())) < 2:
            continue
        clf.fit(df.iloc[tr], y_arr[tr])
        p = clf.predict(df.iloc[te])
        g_acc.append(accuracy_score(y_arr[te], p))
        g_f1.append(f1_score(y_arr[te], p, average="macro", zero_division=0))

    maj = Counter(y).most_common(1)[0]
    return {
        "model": "GradientBoosting + OneHot(names)+Eg",
        "target": target,
        "n_samples": int(len(y_arr)),
        "class_counts": dict(Counter(y)),
        "majority_baseline": {"majority_class": maj[0], "majority_frac": float(maj[1] / len(y))},
        "holdout_20pct": holdout,
        "cv_stratified_5fold": {
            "folds": 5,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)),
        },
        "cv_leave_absorber_group": {
            "folds": len(g_acc),
            "split": "GroupKFold by absorber (harder / more realistic)",
            "accuracy_mean": float(np.mean(g_acc)) if g_acc else None,
            "accuracy_std": float(np.std(g_acc)) if g_acc else None,
            "f1_macro_mean": float(np.mean(g_f1)) if g_f1 else None,
            "f1_macro_std": float(np.std(g_f1)) if g_f1 else None,
        },
    }


def main() -> None:
    report = {
        "eg_regressor": eval_eg(),
        "type_etl_classifier": eval_type("absorber_etl_type"),
        "type_htl_classifier": eval_type("absorber_htl_type"),
        "note": (
            "Random/stratified Type scores near 1.0 can overstate performance "
            "(material name memorization). Prefer GroupKFold-by-absorber metrics "
            "for new-material realism."
        ),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
