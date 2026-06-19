"""
Screen material–gas pairs using the trained adsorption-energy model.

Usage:
  python predict_screening.py
  python predict_screening.py --material "MoS2" --gas NO2
  python predict_screening.py --top 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from enrich_descriptors import (
  infer_material_descriptors,
  load_gas_cache,
  pubchem_gas_properties,
  save_gas_cache,
)
from ml_common import BEST_MODEL_PATH, DATA_DIR, META_PATH, load_frame

OUTPUT_SCREEN = DATA_DIR / "ml_screening_ranked.csv"


def load_meta() -> dict:
  if META_PATH.exists():
    return json.loads(META_PATH.read_text(encoding="utf-8"))
  return {}


def build_feature_row(
  material: str,
  gas: str,
  material_class: str,
  doping: str,
  cache: dict,
) -> dict:
  mat = infer_material_descriptors(material, material_class, doping)
  gas_props = pubchem_gas_properties(gas, cache)
  row = {
    "Material": material,
    "Gas": gas,
    "Material_Class": material_class,
    "Doping": doping,
    "Functional": "NA",
    "DFT_Software": "NA",
    "Mat_Group": mat["Mat_Group"],
    "Mat_Layers": mat["Mat_Layers"],
    "Mat_MetalFraction": mat["Mat_MetalFraction"],
    "Charge_Transfer_e": np.nan,
    "Adsorption_Distance_A": np.nan,
    "Bandgap_Before_eV": np.nan,
    "Bandgap_After_eV": np.nan,
    "Bandgap_Change_eV": np.nan,
    **gas_props,
  }
  for key, val in list(row.items()):
    if val == "NA":
      row[key] = np.nan
  return row


def prepare_features(pred_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
  cat_cols = meta.get("cat_features", [])
  num_cols = meta.get("num_features", [])
  for col in cat_cols + num_cols:
    if col not in pred_df.columns:
      pred_df[col] = np.nan if col in num_cols else "NA"
  for col in num_cols:
    pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce")
  return pred_df[cat_cols + num_cols]


def default_screening_grid(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
  rows = []
  seen = set()
  for _, r in df.drop_duplicates(subset=["Material", "Gas"]).iterrows():
    key = (r["Material"], r["Gas"])
    if key in seen:
      continue
    seen.add(key)
    rows.append(
      (
        r["Material"],
        r["Gas"],
        str(r.get("Material_Class", "NA") or "NA"),
        str(r.get("Doping", "NA") or "NA"),
      )
    )
  return rows


def main():
  parser = argparse.ArgumentParser(description="Screen DFT adsorption energies")
  parser.add_argument("--material", help="Material name (must match dataset naming)")
  parser.add_argument("--gas", help="Gas name")
  parser.add_argument("--top", type=int, default=0, help="Export top-N strongest binding (most negative E)")
  args = parser.parse_args()

  if not BEST_MODEL_PATH.exists():
    raise FileNotFoundError(f"Run train_models.py first. Missing {BEST_MODEL_PATH}")

  meta = load_meta()
  pipe = joblib.load(BEST_MODEL_PATH)
  df = load_frame()
  cache = load_gas_cache()

  if args.material and args.gas:
    subset = df[(df["Material"] == args.material) & (df["Gas"] == args.gas)]
    if subset.empty:
      mat_class, doping = "NA", "NA"
    else:
      row = subset.iloc[0]
      mat_class = str(row.get("Material_Class", "NA") or "NA")
      doping = str(row.get("Doping", "NA") or "NA")
    grid = [(args.material, args.gas, mat_class, doping)]
  else:
    grid = default_screening_grid(df)

  feat_rows = [build_feature_row(mat, gas, mc, dop, cache) for mat, gas, mc, dop in grid]
  save_gas_cache(cache)

  pred_df = pd.DataFrame(feat_rows)
  X = prepare_features(pred_df, meta)
  preds = pipe.predict(X)
  pred_df["Predicted_Adsorption_Energy_eV"] = np.round(preds, 4)
  pred_df = pred_df.sort_values("Predicted_Adsorption_Energy_eV")

  if args.top > 0:
    pred_df = pred_df.head(args.top)

  pred_df.to_csv(OUTPUT_SCREEN, index=False)

  print(f"Model: {meta.get('model', 'unknown')} | feature_set: {meta.get('feature_set', 'unknown')}")
  print(f"Screened {len(pred_df)} material–gas pairs")
  print(
    pred_df[["Material", "Gas", "Mat_Group", "Predicted_Adsorption_Energy_eV"]]
    .head(15)
    .to_string(index=False)
  )
  print(f"\nFull rankings -> {OUTPUT_SCREEN}")


if __name__ == "__main__":
  main()
