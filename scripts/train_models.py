"""
Phase 3 — Train ML baselines with material-grouped evaluation.

Outputs:
  data/ml_benchmark.csv           — single 80/20 material holdout
  data/ml_cv_results.csv          — 5-fold material-grouped CV (mean ± std)
  data/ml_feature_ablation.csv    — descriptors-only vs full features
  data/ml_oof_predictions.csv     — out-of-fold predictions (best model)
  data/ml_error_by_gas.csv        — MAE / R2 grouped by gas
  data/ml_error_by_mat_group.csv  — MAE / R2 grouped by material family
  data/ml_error_by_functional.csv — MAE grouped by DFT functional
  data/models/best_model.joblib   — fitted pipeline for screening
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from ml_common import (
  BEST_MODEL_PATH,
  DATA_DIR,
  GROUP_COL,
  META_PATH,
  MODEL_DIR,
  TARGET,
  build_preprocessor,
  get_models,
  load_frame,
  material_group_kfold,
  material_group_split,
  num_features_for_set,
)

OUTPUT_BENCHMARK = DATA_DIR / "ml_benchmark.csv"
OUTPUT_CV = DATA_DIR / "ml_cv_results.csv"
OUTPUT_ABLATION = DATA_DIR / "ml_feature_ablation.csv"
OUTPUT_OOF = DATA_DIR / "ml_oof_predictions.csv"
OUTPUT_ERR_GAS = DATA_DIR / "ml_error_by_gas.csv"
OUTPUT_ERR_GROUP = DATA_DIR / "ml_error_by_mat_group.csv"
OUTPUT_ERR_FUNC = DATA_DIR / "ml_error_by_functional.csv"

N_CV_FOLDS = 5
DEFAULT_FEATURE_SET = "full"


def evaluate_split(pipe, train_df, test_df) -> tuple[float, float, np.ndarray]:
  X_train = train_df.drop(columns=[TARGET])
  y_train = train_df[TARGET].astype(float)
  X_test = test_df.drop(columns=[TARGET])
  y_test = test_df[TARGET].astype(float)
  pipe.fit(X_train, y_train)
  pred = pipe.predict(X_test)
  return (
    mean_absolute_error(y_test, pred),
    r2_score(y_test, pred),
    pred,
  )


def run_holdout_benchmark(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
  train_df, test_df, held_out = material_group_split(df)
  num_feats = num_features_for_set(feature_set)
  preprocessor = build_preprocessor(train_df.copy(), num_feats)
  rows = []
  for name, model in get_models().items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    mae, r2, _ = evaluate_split(pipe, train_df, test_df)
    rows.append(
      {
        "Model": name,
        "Feature_set": feature_set,
        "MAE_eV": round(mae, 4),
        "R2": round(r2, 4),
        "Train_rows": len(train_df),
        "Test_rows": len(test_df),
        "Held_out_materials": "; ".join(held_out),
      }
    )
  return pd.DataFrame(rows).sort_values("R2", ascending=False)


def run_material_cv(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
  materials = sorted(df[GROUP_COL].unique())
  num_feats = num_features_for_set(feature_set)
  rows = []
  for name, model in get_models().items():
    fold_maes, fold_r2s = [], []
    for train_mats, test_mats in material_group_kfold(materials, N_CV_FOLDS):
      train_df = df[df[GROUP_COL].isin(train_mats)]
      test_df = df[df[GROUP_COL].isin(test_mats)]
      preprocessor = build_preprocessor(train_df.copy(), num_feats)
      pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
      mae, r2, _ = evaluate_split(pipe, train_df, test_df)
      fold_maes.append(mae)
      fold_r2s.append(r2)
    rows.append(
      {
        "Model": name,
        "Feature_set": feature_set,
        "MAE_mean_eV": round(float(np.mean(fold_maes)), 4),
        "MAE_std_eV": round(float(np.std(fold_maes)), 4),
        "R2_mean": round(float(np.mean(fold_r2s)), 4),
        "R2_std": round(float(np.std(fold_r2s)), 4),
        "CV_folds": N_CV_FOLDS,
      }
    )
  return pd.DataFrame(rows).sort_values("R2_mean", ascending=False)


def run_oof_predictions(
  df: pd.DataFrame, model_name: str, feature_set: str
) -> pd.DataFrame:
  materials = sorted(df[GROUP_COL].unique())
  num_feats = num_features_for_set(feature_set)
  models = get_models()
  if model_name not in models:
    model_name = "LightGBM" if "LightGBM" in models else list(models.keys())[0]

  oof_rows = []
  for fold_idx, (train_mats, test_mats) in enumerate(
    material_group_kfold(materials, N_CV_FOLDS)
  ):
    train_df = df[df[GROUP_COL].isin(train_mats)]
    test_df = df[df[GROUP_COL].isin(test_mats)]
    preprocessor = build_preprocessor(train_df.copy(), num_feats)
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", models[model_name])])
    _, _, pred = evaluate_split(pipe, train_df, test_df)
    chunk = test_df.copy()
    chunk["Predicted_Energy_eV"] = pred
    chunk["Residual_eV"] = chunk[TARGET].astype(float) - pred
    chunk["CV_fold"] = fold_idx + 1
    chunk["Model"] = model_name
    oof_rows.append(chunk)

  out = pd.concat(oof_rows, ignore_index=True)
  out["Abs_Error_eV"] = out["Residual_eV"].abs()
  return out


def error_breakdown(oof: pd.DataFrame, group_col: str, min_rows: int = 3) -> pd.DataFrame:
  rows = []
  for key, grp in oof.groupby(group_col):
    if len(grp) < min_rows:
      continue
    y = grp[TARGET].astype(float)
    p = grp["Predicted_Energy_eV"].astype(float)
    rows.append(
      {
        group_col: key,
        "N_rows": len(grp),
        "MAE_eV": round(mean_absolute_error(y, p), 4),
        "R2": round(r2_score(y, p), 4),
        "Mean_Actual_eV": round(float(y.mean()), 4),
        "Mean_Predicted_eV": round(float(p.mean()), 4),
      }
    )
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows).sort_values("MAE_eV", ascending=False)


def fit_and_save_best(df: pd.DataFrame, model_name: str, feature_set: str) -> None:
  num_feats = num_features_for_set(feature_set)
  preprocessor = build_preprocessor(df.copy(), num_feats)
  models = get_models()
  model = models[model_name]
  pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
  X = df.drop(columns=[TARGET])
  y = df[TARGET].astype(float)
  pipe.fit(X, y)
  MODEL_DIR.mkdir(parents=True, exist_ok=True)
  joblib.dump(pipe, BEST_MODEL_PATH)
  meta = {
    "model": model_name,
    "feature_set": feature_set,
    "target": TARGET,
    "cat_features": [c for c in pipe.named_steps["prep"].transformers[0][2]],
    "num_features": num_feats,
    "train_rows": len(df),
    "materials": int(df[GROUP_COL].nunique()),
    "gases": int(df["Gas"].nunique()),
  }
  META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
  df = load_frame()
  print(f"Loaded {len(df)} rows, {df[GROUP_COL].nunique()} materials, {df['Gas'].nunique()} gases")

  # Feature ablation on holdout
  ablation_frames = []
  for fs in ("descriptors_only", "full"):
    ablation_frames.append(run_holdout_benchmark(df, fs))
  ablation = pd.concat(ablation_frames, ignore_index=True)
  ablation.to_csv(OUTPUT_ABLATION, index=False)

  # Primary holdout (full features)
  benchmark = run_holdout_benchmark(df, DEFAULT_FEATURE_SET)
  benchmark.to_csv(OUTPUT_BENCHMARK, index=False)

  # Material-grouped CV (full features)
  cv_results = run_material_cv(df, DEFAULT_FEATURE_SET)
  cv_results.to_csv(OUTPUT_CV, index=False)

  best_row = cv_results.iloc[0]
  best_model = best_row["Model"]
  print(f"\nBest model by CV R2: {best_model} ({best_row['R2_mean']} ± {best_row['R2_std']})")

  # OOF predictions for error analysis
  oof = run_oof_predictions(df, best_model, DEFAULT_FEATURE_SET)
  keep_cols = [
    "Material", "Gas", "Mat_Group", "Material_Class", "Functional",
    "DFT_Software", TARGET, "Predicted_Energy_eV", "Residual_eV",
    "Abs_Error_eV", "CV_fold", "Model",
  ]
  oof[keep_cols].to_csv(OUTPUT_OOF, index=False)

  error_by_gas = error_breakdown(oof, "Gas", min_rows=5)
  error_by_group = error_breakdown(oof, "Mat_Group", min_rows=5)
  error_by_func = error_breakdown(oof, "Functional", min_rows=5)
  error_by_gas.to_csv(OUTPUT_ERR_GAS, index=False)
  error_by_group.to_csv(OUTPUT_ERR_GROUP, index=False)
  error_by_func.to_csv(OUTPUT_ERR_FUNC, index=False)

  fit_and_save_best(df, best_model, DEFAULT_FEATURE_SET)

  print("\n--- Holdout benchmark (full features) ---")
  print(benchmark.to_string(index=False))
  print("\n--- 5-fold material-grouped CV (full features) ---")
  print(cv_results.to_string(index=False))
  print("\n--- Feature ablation (best model per set) ---")
  for fs in ("descriptors_only", "full"):
    sub = ablation[ablation["Feature_set"] == fs].head(1)
    if not sub.empty:
      r = sub.iloc[0]
      print(f"  {fs}: {r['Model']} R2={r['R2']} MAE={r['MAE_eV']} eV")
  print(f"\nWrote outputs to {DATA_DIR}")
  print(f"Saved best model ({best_model}) -> {BEST_MODEL_PATH}")


if __name__ == "__main__":
  main()
