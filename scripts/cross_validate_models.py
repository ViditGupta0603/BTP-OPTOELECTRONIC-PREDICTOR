"""Cross-validate OptoStack production models and write report + figures.

Models match scripts/predict_stack.py and scripts/formula_estimator.py:
  - Eg absorber: RandomForestRegressor (n_estimators=500, …)
  - Formula Eg/χ: RF + GradientBoostingRegressor
  - Type ETL/HTL: GradientBoostingClassifier pipelines

Usage (Windows):
  set PYTHONIOENCODING=utf-8
  python scripts/cross_validate_models.py

Writes:
  data/cross_validation_report.json
  data/cross_validation_report.md
  data/figures/*.png
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formula_estimator import (  # noqa: E402
    _load_training_rows,
    _matrix,
    composition_vector,
    train_estimators,
)
from formula_parse import base_name  # noqa: E402
from predict_stack import (  # noqa: E402
    ABS_PATH,
    EG_MODEL,
    META_OUT,
    STACK_PATH,
    TYPE_MODEL,
    VERIFIED_EG_OVERSAMPLE,
    _feature_frame,
    formula_features,
    train_eg_model,
    train_type_models,
)

DATA = ROOT / "data"
FIGS = DATA / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
REPORT_JSON = DATA / "cross_validation_report.json"
REPORT_MD = DATA / "cross_validation_report.md"
N_SPLITS = 5
RANDOM_STATE = 42

# Production hyperparameters (explicit; match train_* in predict_stack / formula_estimator)
EG_PARAMS = {
    "class": "RandomForestRegressor",
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
FORMULA_EG_PARAMS = {
    "class": "RandomForestRegressor",
    "n_estimators": 400,
    "max_depth": 20,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
FORMULA_CHI_PARAMS = {
    "class": "GradientBoostingRegressor",
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
}
TYPE_CLF_PARAMS = {
    "class": "GradientBoostingClassifier",
    "n_estimators": 100,  # sklearn default (not overridden in train_type_models)
    "learning_rate": 0.1,  # sklearn default
    "max_depth": 3,  # sklearn default
    "random_state": RANDOM_STATE,
    "pipeline": "OneHotEncoder(absorber,partner) + StandardScaler(eg_*)",
}


def _mean_std(vals: list[float | None]) -> dict:
    clean: list[float] = []
    for v in vals:
        if v is None:
            continue
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            continue
        clean.append(fv)
    if not clean:
        return {"mean": None, "std": None, "values": [], "n_defined": 0}
    arr = np.asarray(clean, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "values": clean,
        "n_defined": len(clean),
    }


def _ensure_artifacts() -> dict:
    """Train missing joblibs with production train_* (random_state=42)."""
    actions: dict[str, str] = {}
    from formula_estimator import EG_CHI_MODEL

    if not EG_CHI_MODEL.exists():
        actions["formula"] = "trained"
        train_estimators()
    else:
        actions["formula"] = "present"
    if not EG_MODEL.exists():
        actions["eg"] = "trained"
        train_eg_model()
    else:
        actions["eg"] = "present"
    if not TYPE_MODEL.exists():
        actions["type"] = "trained"
        train_type_models()
    else:
        actions["type"] = "present"
    return actions


def _load_eg_xy() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = list(csv.DictReader(ABS_PATH.open(encoding="utf-8")))
    expanded: list[dict] = []
    for r in rows:
        expanded.append(r)
        if r.get("record_type") == "verified_external":
            expanded.extend([r] * (VERIFIED_EG_OVERSAMPLE - 1))
    formulas = [r["material_absorber"] for r in expanded]
    y = np.array([float(r["absorber_band_gap_eV"]) for r in expanded], dtype=float)
    groups = np.array([base_name(f) for f in formulas])
    keys, X = _feature_frame(formulas)
    return X, y, groups, keys


def _make_eg_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=EG_PARAMS["n_estimators"],
        max_depth=EG_PARAMS["max_depth"],
        min_samples_leaf=EG_PARAMS["min_samples_leaf"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def cv_eg_regressor() -> dict:
    X, y, groups, keys = _load_eg_xy()
    n_groups = len(set(groups.tolist()))
    n_splits = min(N_SPLITS, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics: list[dict] = []
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    fold_r2: list[float] = []

    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        model = _make_eg_model()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        mae = float(mean_absolute_error(y[te], pred))
        rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
        r2 = float(r2_score(y[te], pred))
        fold_metrics.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_test_groups": int(len(set(groups[te].tolist()))),
                "MAE_eV": mae,
                "RMSE_eV": rmse,
                "R2": r2,
            }
        )
        fold_r2.append(r2)
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(pred.tolist())

    y_true_all_a = np.asarray(y_true_all)
    y_pred_all_a = np.asarray(y_pred_all)
    pooled_r2 = float(r2_score(y_true_all_a, y_pred_all_a))

    # --- figures ---
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true_all_a, y_pred_all_a, alpha=0.35, s=18, c="#1f4e79", edgecolors="none")
    lo = float(min(y_true_all_a.min(), y_pred_all_a.min()))
    hi = float(max(y_true_all_a.max(), y_pred_all_a.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.set_xlabel("Actual Eg (eV)")
    ax.set_ylabel("Predicted Eg (eV)")
    ax.set_title(f"Eg RandomForest — predicted vs actual\nGroupKFold R² pooled={pooled_r2:.3f}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    scatter_path = FIGS / "eg_r2_scatter.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    folds = [m["fold"] for m in fold_metrics]
    ax.bar(folds, fold_r2, color="#2a6f97", edgecolor="black", linewidth=0.4)
    ax.axhline(np.mean(fold_r2), color="#c1121f", ls="--", label=f"mean R²={np.mean(fold_r2):.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title("Eg regressor — fold-wise R² (GroupKFold by absorber name)")
    ax.set_xticks(folds)
    ax.legend()
    fig.tight_layout()
    bar_path = FIGS / "eg_r2_fold_bars.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    resid = y_pred_all_a - y_true_all_a
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(y_pred_all_a, resid, alpha=0.35, s=18, c="#495057", edgecolors="none")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Predicted Eg (eV)")
    ax.set_ylabel("Residual (pred − actual) eV")
    ax.set_title("Eg regressor — residual plot")
    fig.tight_layout()
    resid_path = FIGS / "eg_residuals.png"
    fig.savefig(resid_path, dpi=150)
    plt.close(fig)

    return {
        "model": EG_PARAMS,
        "dataset": str(ABS_PATH.relative_to(ROOT)),
        "n_rows_expanded": int(len(y)),
        "n_unique_absorbers": int(n_groups),
        "n_features": int(len(keys)),
        "protocol": {
            "splitter": "GroupKFold",
            "group_by": "absorber base_name",
            "n_splits": n_splits,
            "rationale": "Hold out entire material families by name so folds do not leak the same absorber via oversampling.",
        },
        "folds": fold_metrics,
        "summary": {
            "MAE_eV": _mean_std([m["MAE_eV"] for m in fold_metrics]),
            "RMSE_eV": _mean_std([m["RMSE_eV"] for m in fold_metrics]),
            "R2": _mean_std(fold_r2),
            "pooled_R2": pooled_r2,
        },
        "figures": {
            "scatter": str(scatter_path.relative_to(ROOT)).replace("\\", "/"),
            "fold_r2_bars": str(bar_path.relative_to(ROOT)).replace("\\", "/"),
            "residuals": str(resid_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }


def _type_frame(target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows: list[dict] = []
    for p in (STACK_PATH, DATA / "opto_literature_dataset_scaps_only.csv"):
        if p.exists():
            rows.extend(csv.DictReader(p.open(encoding="utf-8")))

    X_mat, y, groups = [], [], []
    for r in rows:
        if not r.get(target):
            continue
        eg_a = float(r["absorber_band_gap_eV"])
        eg_p = float(r["etl_band_gap_eV"] if "etl" in target else r["htl_band_gap_eV"])
        partner = r["material_etl"] if "etl" in target else r["material_htl"]
        fa = formula_features(r["material_absorber"])
        X_mat.append(
            {
                "absorber": r["material_absorber"],
                "partner": partner,
                "eg_a": eg_a,
                "eg_p": eg_p,
                "eg_diff": eg_a - eg_p,
                "abs_frac_I": fa.get("frac_I", 0.0),
                "abs_has_organic": fa.get("has_organic", 0.0),
            }
        )
        y.append(r[target])
        groups.append(base_name(r["material_absorber"]))
    return pd.DataFrame(X_mat), np.array(y), np.array(groups)


def _make_type_clf() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["absorber", "partner"]),
            (
                "num",
                StandardScaler(),
                ["eg_a", "eg_p", "eg_diff", "abs_frac_I", "abs_has_organic"],
            ),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=TYPE_CLF_PARAMS["n_estimators"],
                    learning_rate=TYPE_CLF_PARAMS["learning_rate"],
                    max_depth=TYPE_CLF_PARAMS["max_depth"],
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _ovr_roc_auc(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float | None:
    """Macro-average one-vs-rest AUC over classes present (with both labels) in y_true."""
    if len(set(y_true.tolist())) < 2:
        return None
    scores: list[float] = []
    for i, cls in enumerate(classes):
        y_bin = (np.asarray(y_true) == cls).astype(int)
        n_pos = int(y_bin.sum())
        if n_pos == 0 or n_pos == len(y_bin):
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
            scores.append(float(auc(fpr, tpr)))
        except ValueError:
            continue
    if not scores:
        return None
    return float(np.mean(scores))


def _plot_roc_ovr(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list,
    title: str,
    out_path: Path,
) -> None:
    y_bin = label_binarize(y_true, classes=classes)
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#1d3557", "#e63946", "#2a9d8f", "#f4a261"]
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0 or (1 - y_bin[:, i]).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"Type {cls} (AUC={roc_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion_heatmap(
    cm: np.ndarray, labels: list[str], title: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([f"Type {l}" for l in labels])
    ax.set_yticklabels([f"Type {l}" for l in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cv_type_classifier(target: str, side: str) -> dict:
    df, y, groups = _type_frame(target)
    classes = sorted(set(y.tolist()))
    n_groups = len(set(groups.tolist()))
    n_splits_g = min(N_SPLITS, n_groups)

    # Primary: GroupKFold by absorber
    gkf = GroupKFold(n_splits=n_splits_g)
    g_folds: list[dict] = []
    y_true_g: list[str] = []
    y_pred_g: list[str] = []
    proba_g: list[np.ndarray] = []
    classes_order: list | None = None

    for fold_i, (tr, te) in enumerate(gkf.split(df, y, groups), start=1):
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = _make_type_clf()
        clf.fit(df.iloc[tr], y[tr])
        pred = clf.predict(df.iloc[te])
        proba = clf.predict_proba(df.iloc[te])
        if classes_order is None:
            classes_order = list(clf.named_steps["clf"].classes_)
        # Align proba to global class order
        aligned = np.zeros((len(te), len(classes)))
        for j, c in enumerate(clf.named_steps["clf"].classes_):
            if c in classes:
                aligned[:, classes.index(c)] = proba[:, j]
        acc = float(accuracy_score(y[te], pred))
        f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
        auc_v = _ovr_roc_auc(y[te], aligned, classes)
        g_folds.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": acc,
                "f1_macro": f1,
                "roc_auc_macro_ovr": auc_v,
            }
        )
        y_true_g.extend(y[te].tolist())
        y_pred_g.extend(pred.tolist())
        proba_g.append(aligned)

    # Secondary: StratifiedKFold for comparison
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    s_folds: list[dict] = []
    for fold_i, (tr, te) in enumerate(skf.split(df, y), start=1):
        clf = _make_type_clf()
        clf.fit(df.iloc[tr], y[tr])
        pred = clf.predict(df.iloc[te])
        proba = clf.predict_proba(df.iloc[te])
        aligned = np.zeros((len(te), len(classes)))
        for j, c in enumerate(clf.named_steps["clf"].classes_):
            if c in classes:
                aligned[:, classes.index(c)] = proba[:, j]
        s_folds.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": float(accuracy_score(y[te], pred)),
                "f1_macro": float(f1_score(y[te], pred, average="macro", zero_division=0)),
                "roc_auc_macro_ovr": _ovr_roc_auc(y[te], aligned, classes),
            }
        )

    y_true_a = np.asarray(y_true_g)
    y_pred_a = np.asarray(y_pred_g)
    cm = confusion_matrix(y_true_a, y_pred_a, labels=classes)
    proba_all = np.vstack(proba_g) if proba_g else np.zeros((0, len(classes)))

    roc_path = FIGS / f"type_{side}_roc_ovr.png"
    cm_path = FIGS / f"type_{side}_confusion_heatmap.png"
    if len(y_true_a) and proba_all.shape[0] == len(y_true_a):
        _plot_roc_ovr(
            y_true_a,
            proba_all,
            classes,
            f"Type {side.upper()} — OvR ROC (GroupKFold pooled OOF)",
            roc_path,
        )
    _plot_confusion_heatmap(
        cm,
        classes,
        f"Type {side.upper()} — confusion matrix (GroupKFold OOF)",
        cm_path,
    )

    def _summ(folds: list[dict], key: str) -> dict:
        vals = [f[key] for f in folds if f.get(key) is not None]
        return _mean_std(vals)

    return {
        "model": TYPE_CLF_PARAMS,
        "target": target,
        "side": side,
        "n_samples": int(len(y)),
        "class_counts": dict(Counter(y.tolist())),
        "labels": classes,
        "protocol_primary": {
            "splitter": "GroupKFold",
            "group_by": "absorber base_name",
            "n_splits": n_splits_g,
            "rationale": "More realistic generalization to unseen absorbers than random stratified splits.",
        },
        "protocol_secondary": {
            "splitter": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "note": "Optimistic vs GroupKFold — material-name features can memorize train absorbers.",
        },
        "group_kfold": {
            "folds": g_folds,
            "summary": {
                "accuracy": _summ(g_folds, "accuracy"),
                "f1_macro": _summ(g_folds, "f1_macro"),
                "roc_auc_macro_ovr": _summ(g_folds, "roc_auc_macro_ovr"),
            },
            "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        },
        "stratified_kfold": {
            "folds": s_folds,
            "summary": {
                "accuracy": _summ(s_folds, "accuracy"),
                "f1_macro": _summ(s_folds, "f1_macro"),
                "roc_auc_macro_ovr": _summ(s_folds, "roc_auc_macro_ovr"),
            },
        },
        "figures": {
            "roc": str(roc_path.relative_to(ROOT)).replace("\\", "/"),
            "confusion_heatmap": str(cm_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }


def cv_formula_estimators() -> dict:
    rows = _load_training_rows()
    feat_rows = [composition_vector(r["material"], r["role"]) for r in rows]
    keys, X = _matrix(feat_rows)
    y_eg = np.array([r["Eg_eV"] for r in rows], dtype=float)
    groups = np.array([base_name(r["material"]) for r in rows])
    n_groups = len(set(groups.tolist()))
    n_splits = min(N_SPLITS, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    eg_folds: list[dict] = []
    y_t, y_p = [], []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y_eg, groups), start=1):
        model = RandomForestRegressor(
            n_estimators=FORMULA_EG_PARAMS["n_estimators"],
            max_depth=FORMULA_EG_PARAMS["max_depth"],
            min_samples_leaf=FORMULA_EG_PARAMS["min_samples_leaf"],
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model.fit(X[tr], y_eg[tr])
        pred = model.predict(X[te])
        eg_folds.append(
            {
                "fold": fold_i,
                "MAE_eV": float(mean_absolute_error(y_eg[te], pred)),
                "RMSE_eV": float(np.sqrt(mean_squared_error(y_eg[te], pred))),
                "R2": float(r2_score(y_eg[te], pred)),
                "n_test": int(len(te)),
            }
        )
        y_t.extend(y_eg[te].tolist())
        y_p.extend(pred.tolist())

    # Chi CV
    chi_rows = [r for r in rows if r.get("chi_eV") is not None]
    feat_chi = [composition_vector(r["material"], r["role"]) for r in chi_rows]
    _k, Xc = _matrix(feat_chi, keys)
    eg_chi = np.array([r["Eg_eV"] for r in chi_rows], dtype=float).reshape(-1, 1)
    Xc = np.hstack([Xc, eg_chi])
    y_chi = np.array([float(r["chi_eV"]) for r in chi_rows], dtype=float)
    g_chi = np.array([base_name(r["material"]) for r in chi_rows])
    n_g_chi = len(set(g_chi.tolist()))
    n_splits_chi = min(N_SPLITS, n_g_chi)
    chi_folds: list[dict] = []
    for fold_i, (tr, te) in enumerate(
        GroupKFold(n_splits=n_splits_chi).split(Xc, y_chi, g_chi), start=1
    ):
        model = GradientBoostingRegressor(
            n_estimators=FORMULA_CHI_PARAMS["n_estimators"],
            max_depth=FORMULA_CHI_PARAMS["max_depth"],
            learning_rate=FORMULA_CHI_PARAMS["learning_rate"],
            random_state=RANDOM_STATE,
        )
        model.fit(Xc[tr], y_chi[tr])
        pred = model.predict(Xc[te])
        chi_folds.append(
            {
                "fold": fold_i,
                "MAE_eV": float(mean_absolute_error(y_chi[te], pred)),
                "RMSE_eV": float(np.sqrt(mean_squared_error(y_chi[te], pred))),
                "R2": float(r2_score(y_chi[te], pred)),
                "n_test": int(len(te)),
            }
        )

    yt, yp = np.asarray(y_t), np.asarray(y_p)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(yt, yp, alpha=0.35, s=18, c="#264653", edgecolors="none")
    lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    pooled = float(r2_score(yt, yp))
    ax.set_xlabel("Actual Eg (eV)")
    ax.set_ylabel("Predicted Eg (eV)")
    ax.set_title(f"Formula Eg RF — predicted vs actual\nGroupKFold pooled R²={pooled:.3f}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    scatter_path = FIGS / "formula_eg_r2_scatter.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    return {
        "eg_model": FORMULA_EG_PARAMS,
        "chi_model": FORMULA_CHI_PARAMS,
        "n_eg": int(len(y_eg)),
        "n_chi": int(len(y_chi)),
        "n_unique_materials_eg": int(n_groups),
        "protocol": {
            "splitter": "GroupKFold",
            "group_by": "material base_name",
            "n_splits_eg": n_splits,
            "n_splits_chi": n_splits_chi,
        },
        "eg": {
            "folds": eg_folds,
            "summary": {
                "MAE_eV": _mean_std([f["MAE_eV"] for f in eg_folds]),
                "RMSE_eV": _mean_std([f["RMSE_eV"] for f in eg_folds]),
                "R2": _mean_std([f["R2"] for f in eg_folds]),
                "pooled_R2": pooled,
            },
        },
        "chi": {
            "folds": chi_folds,
            "summary": {
                "MAE_eV": _mean_std([f["MAE_eV"] for f in chi_folds]),
                "RMSE_eV": _mean_std([f["RMSE_eV"] for f in chi_folds]),
                "R2": _mean_std([f["R2"] for f in chi_folds]),
            },
        },
        "figures": {
            "eg_scatter": str(scatter_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }


def _fmt_ms(d: dict | None, digits: int = 4) -> str:
    if not d or d.get("mean") is None:
        return "n/a"
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def write_markdown(report: dict) -> None:
    eg = report["eg_absorber_regressor"]
    fe = report["formula_eg_chi"]
    etl = report["type_etl"]
    htl = report["type_htl"]
    meta = report.get("train_meta") or {}
    stack_n = report["dataset_sizes"]["stack_table_rows"]

    lines = [
        "# OptoStack cross-validation report",
        "",
        f"**Date:** {report['date']}",
        "",
        f"**Stack table size:** {stack_n} rows (`perovskite_stack_dataset.csv`"
        + (f"; train_meta type n_etl={meta.get('type', {}).get('n_etl')}" if meta.get("type") else "")
        + ").",
        f"**Absorber library:** {report['dataset_sizes']['absorber_library_rows']} rows"
        f" (Eg training expands verified_external ×{VERIFIED_EG_OVERSAMPLE} → "
        f"{eg['n_rows_expanded']} samples).",
        "",
        "## Models in use",
        "",
        "| Role | sklearn class | Artifact |",
        "|------|---------------|----------|",
        "| Absorber Eg | `RandomForestRegressor` | `data/models/perovskite_eg_regressor.joblib` |",
        "| Formula Eg | `RandomForestRegressor` | `data/models/formula_eg_chi_estimator.joblib` |",
        "| Formula χ | `GradientBoostingRegressor` | same joblib |",
        "| Type ETL | `GradientBoostingClassifier` (Pipeline) | `data/models/stack_type_classifier.joblib` |",
        "| Type HTL | `GradientBoostingClassifier` (Pipeline) | same joblib |",
        "",
        "One-line summary: **Eg RF (500 trees) + formula RF/GBR + Type ETL/HTL GBC pipelines** "
        "(all `random_state=42`).",
        "",
        "## Hyperparameters",
        "",
        "| Model | Parameters |",
        "|-------|------------|",
        f"| Absorber Eg RF | n_estimators={EG_PARAMS['n_estimators']}, max_depth={EG_PARAMS['max_depth']}, "
        f"min_samples_leaf={EG_PARAMS['min_samples_leaf']}, random_state=42, n_jobs=-1 |",
        f"| Formula Eg RF | n_estimators={FORMULA_EG_PARAMS['n_estimators']}, max_depth={FORMULA_EG_PARAMS['max_depth']}, "
        f"min_samples_leaf={FORMULA_EG_PARAMS['min_samples_leaf']}, random_state=42 |",
        f"| Formula χ GBR | n_estimators={FORMULA_CHI_PARAMS['n_estimators']}, max_depth={FORMULA_CHI_PARAMS['max_depth']}, "
        f"learning_rate={FORMULA_CHI_PARAMS['learning_rate']}, random_state=42 |",
        f"| Type GBC | n_estimators={TYPE_CLF_PARAMS['n_estimators']}, learning_rate={TYPE_CLF_PARAMS['learning_rate']}, "
        f"max_depth={TYPE_CLF_PARAMS['max_depth']}, random_state=42; "
        f"features: OneHot(absorber,partner)+scaled Eg diffs / organic flags |",
        "",
        "## CV protocol",
        "",
        "- **Eg (absorber + formula):** `GroupKFold` (k=5) grouped by material `base_name` — "
        "avoids leakage from verified-row oversampling and duplicate names.",
        "- **Type ETL/HTL (primary):** `GroupKFold` by absorber name (realistic for new absorbers).",
        "- **Type (secondary):** `StratifiedKFold` (k=5, shuffle, seed=42) — typically optimistic "
        "because categorical material names can be memorized.",
        "- Metrics: Eg → MAE, RMSE, R² per fold + mean±std; Type → accuracy, macro F1, macro OvR ROC-AUC.",
        "",
        "## Eg absorber regressor metrics",
        "",
        f"| Metric | mean ± std (GroupKFold) |",
        f"|--------|-------------------------|",
        f"| MAE (eV) | {_fmt_ms(eg['summary']['MAE_eV'])} |",
        f"| RMSE (eV) | {_fmt_ms(eg['summary']['RMSE_eV'])} |",
        f"| R² | {_fmt_ms(eg['summary']['R2'])} |",
        f"| Pooled OOF R² | {eg['summary']['pooled_R2']:.4f} |",
        "",
        "### Per-fold Eg",
        "",
        "| Fold | n_test | MAE | RMSE | R² |",
        "|------|--------|-----|------|----|",
    ]
    for f in eg["folds"]:
        lines.append(
            f"| {f['fold']} | {f['n_test']} | {f['MAE_eV']:.4f} | {f['RMSE_eV']:.4f} | {f['R2']:.4f} |"
        )

    lines += [
        "",
        "## Formula Eg / χ metrics",
        "",
        f"| Target | MAE | RMSE | R² |",
        f"|--------|-----|------|----|",
        f"| Formula Eg | {_fmt_ms(fe['eg']['summary']['MAE_eV'])} | "
        f"{_fmt_ms(fe['eg']['summary']['RMSE_eV'])} | {_fmt_ms(fe['eg']['summary']['R2'])} |",
        f"| Formula χ | {_fmt_ms(fe['chi']['summary']['MAE_eV'])} | "
        f"{_fmt_ms(fe['chi']['summary']['RMSE_eV'])} | {_fmt_ms(fe['chi']['summary']['R2'])} |",
        "",
        "## Type classification metrics",
        "",
        "### GroupKFold by absorber (preferred)",
        "",
        "| Side | Accuracy | macro F1 | macro OvR ROC-AUC |",
        "|------|----------|----------|-------------------|",
        f"| ETL | {_fmt_ms(etl['group_kfold']['summary']['accuracy'])} | "
        f"{_fmt_ms(etl['group_kfold']['summary']['f1_macro'])} | "
        f"{_fmt_ms(etl['group_kfold']['summary']['roc_auc_macro_ovr'])} |",
        f"| HTL | {_fmt_ms(htl['group_kfold']['summary']['accuracy'])} | "
        f"{_fmt_ms(htl['group_kfold']['summary']['f1_macro'])} | "
        f"{_fmt_ms(htl['group_kfold']['summary']['roc_auc_macro_ovr'])} |",
        "",
        "### StratifiedKFold (comparison — often optimistic)",
        "",
        "| Side | Accuracy | macro F1 | macro OvR ROC-AUC |",
        "|------|----------|----------|-------------------|",
        f"| ETL | {_fmt_ms(etl['stratified_kfold']['summary']['accuracy'])} | "
        f"{_fmt_ms(etl['stratified_kfold']['summary']['f1_macro'])} | "
        f"{_fmt_ms(etl['stratified_kfold']['summary']['roc_auc_macro_ovr'])} |",
        f"| HTL | {_fmt_ms(htl['stratified_kfold']['summary']['accuracy'])} | "
        f"{_fmt_ms(htl['stratified_kfold']['summary']['f1_macro'])} | "
        f"{_fmt_ms(htl['stratified_kfold']['summary']['roc_auc_macro_ovr'])} |",
        "",
        "## Figures",
        "",
        "| Figure | Path |",
        "|--------|------|",
        f"| Eg predicted vs actual (R²) | `{eg['figures']['scatter']}` |",
        f"| Eg fold-wise R² bars | `{eg['figures']['fold_r2_bars']}` |",
        f"| Eg residuals | `{eg['figures']['residuals']}` |",
        f"| Formula Eg scatter | `{fe['figures']['eg_scatter']}` |",
        f"| Type ETL ROC (OvR) | `{etl['figures']['roc']}` |",
        f"| Type HTL ROC (OvR) | `{htl['figures']['roc']}` |",
        f"| Type ETL confusion heatmap | `{etl['figures']['confusion_heatmap']}` |",
        f"| Type HTL confusion heatmap | `{htl['figures']['confusion_heatmap']}` |",
        "",
        "## Caveats",
        "",
        "- **Lookup vs ML:** Runtime prefers literature / stack-table lookup and physics Type from Eg+χ. "
        "ML is used when library values are missing; these CV scores describe the ML fallbacks, not the lookup path.",
        "- **GroupKFold vs random:** Stratified/random Type accuracy can approach ~1.0 via name memorization; "
        "GroupKFold-by-absorber is the realistic number for novel absorbers.",
        "- **Suitability is not CV'd:** YES/MARGINAL/NO comes from deterministic Anderson rules on Types + offsets, "
        "not a supervised model.",
        "- **Formula estimator** blends family/Vegard priors with ML at inference; CV here evaluates the ML "
        "regressor heads only.",
        "- Holdout MAE in `train_meta.json` uses a single random split and may differ from GroupKFold means.",
        "",
        "## Artifact status at run time",
        "",
        "```json",
        json.dumps(report.get("artifact_status", {}), indent=2),
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Ensuring model artifacts…")
    artifact_status = _ensure_artifacts()

    stack_n = sum(1 for _ in csv.DictReader(STACK_PATH.open(encoding="utf-8")))
    abs_n = sum(1 for _ in csv.DictReader(ABS_PATH.open(encoding="utf-8")))
    train_meta = {}
    if META_OUT.exists():
        train_meta = json.loads(META_OUT.read_text(encoding="utf-8"))

    print("CV: absorber Eg regressor…")
    eg_report = cv_eg_regressor()
    print(
        f"  Eg R² mean={eg_report['summary']['R2']['mean']:.4f} "
        f"± {eg_report['summary']['R2']['std']:.4f}"
    )

    print("CV: formula Eg/χ…")
    formula_report = cv_formula_estimators()
    print(
        f"  Formula Eg R² mean={formula_report['eg']['summary']['R2']['mean']:.4f}; "
        f"χ MAE mean={formula_report['chi']['summary']['MAE_eV']['mean']:.4f}"
    )

    print("CV: Type ETL…")
    etl_report = cv_type_classifier("absorber_etl_type", "etl")
    etl_auc = etl_report["group_kfold"]["summary"]["roc_auc_macro_ovr"]["mean"]
    print(
        f"  ETL GroupKFold acc={etl_report['group_kfold']['summary']['accuracy']['mean']:.4f} "
        f"AUC={etl_auc if etl_auc is not None else 'n/a'}"
    )

    print("CV: Type HTL…")
    htl_report = cv_type_classifier("absorber_htl_type", "htl")
    htl_auc = htl_report["group_kfold"]["summary"]["roc_auc_macro_ovr"]["mean"]
    print(
        f"  HTL GroupKFold acc={htl_report['group_kfold']['summary']['accuracy']['mean']:.4f} "
        f"AUC={htl_auc if htl_auc is not None else 'n/a'}"
    )

    report = {
        "date": str(date.today()),
        "models_summary": (
            "Absorber Eg: RandomForestRegressor(500); "
            "Formula Eg: RandomForestRegressor(400/depth20); "
            "Formula χ: GradientBoostingRegressor(300/depth4/lr0.05); "
            "Type ETL/HTL: GradientBoostingClassifier Pipeline (defaults + random_state=42)"
        ),
        "artifact_status": artifact_status,
        "dataset_sizes": {
            "stack_table_rows": stack_n,
            "absorber_library_rows": abs_n,
        },
        "train_meta": train_meta,
        "hyperparameters": {
            "eg_absorber": EG_PARAMS,
            "formula_eg": FORMULA_EG_PARAMS,
            "formula_chi": FORMULA_CHI_PARAMS,
            "type_classifier": TYPE_CLF_PARAMS,
        },
        "eg_absorber_regressor": eg_report,
        "formula_eg_chi": formula_report,
        "type_etl": etl_report,
        "type_htl": htl_report,
        "caveats": [
            "Lookup/physics path preferred at runtime; CV scores ML fallbacks.",
            "GroupKFold preferred over stratified for Type realism.",
            "Suitability verdict not cross-validated.",
        ],
    }

    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report)
    print(f"\nWrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"Figures under {FIGS}")


if __name__ == "__main__":
    main()
