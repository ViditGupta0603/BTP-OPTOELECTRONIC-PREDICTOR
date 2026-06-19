"""Shared ML utilities for Phase 3 training and screening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
META_PATH = DATA_DIR / "ml_model_meta.json"
ENRICHED_CSV = DATA_DIR / "dft_gas_sensing_dataset_enriched.csv"
ML_CSV = DATA_DIR / "dft_gas_sensing_dataset_ml.csv"

TARGET = "Adsorption_Energy_eV"
GROUP_COL = "Material"

CAT_FEATURES = [
  "Gas",
  "Material_Class",
  "Doping",
  "Functional",
  "DFT_Software",
  "Mat_Group",
]

NUM_DESCRIPTOR_FEATURES = [
  "Gas_MolecularWeight",
  "Gas_XLogP",
  "Gas_TPSA",
  "Gas_HeavyAtomCount",
  "Gas_HBondDonorCount",
  "Gas_HBondAcceptorCount",
  "Mat_Layers",
  "Mat_MetalFraction",
]

NUM_PAPER_FEATURES = [
  "Charge_Transfer_e",
  "Adsorption_Distance_A",
  "Bandgap_Before_eV",
  "Bandgap_After_eV",
  "Bandgap_Change_eV",
]

FEATURE_SETS = {
  "descriptors_only": NUM_DESCRIPTOR_FEATURES,
  "full": NUM_DESCRIPTOR_FEATURES + NUM_PAPER_FEATURES,
}


def load_frame() -> pd.DataFrame:
  path = ENRICHED_CSV if ENRICHED_CSV.exists() else ML_CSV
  if not path.exists():
    raise FileNotFoundError("Run build_dataset.py and enrich_descriptors.py first.")
  df = pd.read_csv(path)
  df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
  return df.dropna(subset=[TARGET]).copy()


def num_features_for_set(feature_set: str) -> list[str]:
  return FEATURE_SETS.get(feature_set, FEATURE_SETS["full"])


def material_group_split(
  df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
  materials = sorted(df[GROUP_COL].unique())
  rng = np.random.default_rng(seed)
  rng.shuffle(materials)
  n_test = max(1, int(round(len(materials) * test_frac)))
  test_materials = set(materials[:n_test])
  test_mask = df[GROUP_COL].isin(test_materials)
  return (
    df.loc[~test_mask].copy(),
    df.loc[test_mask].copy(),
    sorted(test_materials),
  )


def material_group_kfold(
  materials: list[str], n_splits: int = 5, seed: int = 42
):
  mats = sorted(materials)
  rng = np.random.default_rng(seed)
  rng.shuffle(mats)
  folds = np.array_split(mats, n_splits)
  for i in range(n_splits):
    test_mats = set(folds[i].tolist())
    train_mats = set(m for j, fold in enumerate(folds) if j != i for m in fold.tolist())
    yield sorted(train_mats), sorted(test_mats)


def build_preprocessor(
  df: pd.DataFrame, num_features: list[str]
) -> ColumnTransformer:
  cat_cols = [c for c in CAT_FEATURES if c in df.columns]
  num_cols = [c for c in num_features if c in df.columns]
  for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

  transformers = []
  if cat_cols:
    transformers.append(
      (
        "cat",
        Pipeline(
          steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
          ]
        ),
        cat_cols,
      )
    )
  if num_cols:
    transformers.append(
      (
        "num",
        Pipeline(
          steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
          ]
        ),
        num_cols,
      )
    )
  return ColumnTransformer(transformers=transformers)


def get_models() -> dict:
  models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
      n_estimators=300, random_state=42, n_jobs=-1
    ),
  }
  try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
      n_estimators=300,
      learning_rate=0.05,
      max_depth=6,
      subsample=0.9,
      colsample_bytree=0.9,
      random_state=42,
      verbosity=0,
    )
  except ImportError:
    pass
  try:
    from lightgbm import LGBMRegressor

    models["LightGBM"] = LGBMRegressor(
      n_estimators=300,
      learning_rate=0.05,
      random_state=42,
      verbose=-1,
    )
  except ImportError:
    pass
  try:
    from catboost import CatBoostRegressor

    models["CatBoost"] = CatBoostRegressor(
      iterations=300,
      learning_rate=0.05,
      depth=6,
      random_seed=42,
      verbose=False,
    )
  except ImportError:
    pass
  return models
