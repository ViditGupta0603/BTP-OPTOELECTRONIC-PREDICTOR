"""
Merge gas (PubChem) and material descriptors into the ML dataset.

Phase 2 of the B.T.P. pipeline: enriches labels from build_dataset.py with
features suitable for RF / XGBoost / LightGBM training.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INPUT_CSV = DATA_DIR / "dft_gas_sensing_dataset_ml.csv"
OUTPUT_CSV = DATA_DIR / "dft_gas_sensing_dataset_enriched.csv"
GAS_CACHE = DATA_DIR / "gas_descriptor_cache.json"

# Map dataset gas names to PubChem search terms where they differ.
GAS_PUBCHEM_QUERY = {
  "CO": "carbon monoxide",
  "NO": "nitric oxide",
  "NO2": "nitrogen dioxide",
  "O2": "oxygen",
  "N2": "nitrogen",
  "C2H6": "ethane",
  "CH3CHO": "acetaldehyde",
  "C6H6": "benzene",
  "Formaldehyde": "formaldehyde",
  "SOF2": "thionyl fluoride",
  "SO2F2": "sulfuryl fluoride",
  "HF": "hydrogen fluoride",
}

# Gases that must be re-fetched when cache is stale (ambiguous PubChem names).
GAS_CACHE_BYPASS = {"CO", "NO", "NO2"}

MATERIAL_CLASS_DEFAULTS = {
  "2D Carbon": ("carbon_2d", 1, 0.0),
  "2D TMDC": ("tmd", 1, 0.33),
  "MXene": ("mxene", 1, 0.25),
  "Metal Oxide": ("metal_oxide", 1, 0.33),
}

# Explicit overrides for materials where name heuristics are ambiguous.
MATERIAL_DESCRIPTORS = {
  "Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.0},
  "B-doped Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.01},
  "B-doped Graphene (1B)": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.01},
  "N-doped Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.01},
  "Defective Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.0},
  "Ni-doped Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.02},
  "Pd-doped Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.02},
  "N-Ga co-doped Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.04},
  "Ga-doped Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.02},
  "Co@Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.02},
  "Fe@Graphene": {"group": "carbon_2d", "layers": 1, "metal_fraction": 0.02},
  "MoS2": {"group": "tmd", "layers": 1, "metal_fraction": 0.33},
  "Few-Layer MoS2": {"group": "tmd", "layers": 3, "metal_fraction": 0.33},
  "WS2": {"group": "tmd", "layers": 1, "metal_fraction": 0.33},
  "Janus MoSSe": {"group": "tmd_janus", "layers": 1, "metal_fraction": 0.33},
  "Janus MoSSe (MoV)": {"group": "tmd_janus", "layers": 1, "metal_fraction": 0.33},
  "Janus MoSSe (S/SeV)": {"group": "tmd_janus", "layers": 1, "metal_fraction": 0.33},
  "Janus MoSSe (SeV)": {"group": "tmd_janus", "layers": 1, "metal_fraction": 0.33},
  "Ir-modified MoS2": {"group": "tmd_decorated", "layers": 1, "metal_fraction": 0.34},
  "Al-doped MoS2": {"group": "tmd_doped", "layers": 1, "metal_fraction": 0.34},
  "SnO2": {"group": "metal_oxide", "layers": 1, "metal_fraction": 0.33},
  "Ti2CO2": {"group": "mxene", "layers": 1, "metal_fraction": 0.25},
  "V2CO2": {"group": "mxene", "layers": 1, "metal_fraction": 0.25},
  "Nb2CO2": {"group": "mxene", "layers": 1, "metal_fraction": 0.25},
  "Mo2CO2": {"group": "mxene", "layers": 1, "metal_fraction": 0.25},
  "Ti3C2O2": {"group": "mxene", "layers": 1, "metal_fraction": 0.30},
  "Sc2CO2 (O-vacancy)": {"group": "mxene_defect", "layers": 1, "metal_fraction": 0.22},
  "Zigzag Graphene Nanoribbon": {"group": "carbon_1d", "layers": 1, "metal_fraction": 0.0},
  "ZGNR-O (epoxy)": {"group": "carbon_1d", "layers": 1, "metal_fraction": 0.0},
  "ZGNR-OH (hydroxyl)": {"group": "carbon_1d", "layers": 1, "metal_fraction": 0.0},
  "ZGNR-O-OH (epoxy+hydroxyl)": {"group": "carbon_1d", "layers": 1, "metal_fraction": 0.0},
}

MULTI_B_FRACTION = {
  "Multi-B Graphene (1B)": 0.01,
  "Multi-B Graphene (2B-II)": 0.02,
  "Multi-B Graphene (2B-III)": 0.02,
  "Multi-B Graphene (3B-II)": 0.03,
  "Multi-B Graphene (3B-III)": 0.03,
}

GROUP_4B_TMD = re.compile(r"^1[HT]-(Hf|Zr)(S2|Se2|Te2)$")
MOTE2_DECORATED = re.compile(r"^(Co|V|W|Zr|Au|Ag|Cu|Rh)-MoTe2$")
COINAGE_MOS2 = re.compile(r"^(Au|Ag|Cu)(2?)-substituted MoS2$")
GRAPHENE_DOPED = re.compile(
  r"(doped|embedded|anchored|@|Fe-N co-doped|B-pattern|Al-doped|Zn-embedded)",
  re.I,
)


def load_gas_cache() -> dict:
  if GAS_CACHE.exists():
    with GAS_CACHE.open(encoding="utf-8") as f:
      return json.load(f)
  return {}


def save_gas_cache(cache: dict) -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  with GAS_CACHE.open("w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2)


def pubchem_gas_properties(gas: str, cache: dict) -> dict:
  if gas in cache and gas not in GAS_CACHE_BYPASS:
    return cache[gas]

  query = GAS_PUBCHEM_QUERY.get(gas, gas)
  url = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    f"{requests.utils.quote(query)}/property/MolecularWeight,XLogP,"
    "TPSA,HeavyAtomCount,HBondDonorCount,HBondAcceptorCount/JSON"
  )
  props = {
    "Gas_MolecularWeight": "NA",
    "Gas_XLogP": "NA",
    "Gas_TPSA": "NA",
    "Gas_HeavyAtomCount": "NA",
    "Gas_HBondDonorCount": "NA",
    "Gas_HBondAcceptorCount": "NA",
  }
  try:
    resp = requests.get(url, timeout=20)
    if resp.ok:
      block = resp.json()["PropertyTable"]["Properties"][0]
      mapping = {
        "Gas_MolecularWeight": "MolecularWeight",
        "Gas_XLogP": "XLogP",
        "Gas_TPSA": "TPSA",
        "Gas_HeavyAtomCount": "HeavyAtomCount",
        "Gas_HBondDonorCount": "HBondDonorCount",
        "Gas_HBondAcceptorCount": "HBondAcceptorCount",
      }
      for out_key, in_key in mapping.items():
        if in_key in block and block[in_key] is not None:
          props[out_key] = block[in_key]
  except (requests.RequestException, KeyError, IndexError, ValueError):
    pass

  cache[gas] = props
  time.sleep(0.25)
  return props


def infer_material_descriptors(
  material: str,
  material_class: str = "",
  doping: str = "NA",
) -> dict:
  if material in MATERIAL_DESCRIPTORS:
    base = MATERIAL_DESCRIPTORS[material]
    return {
      "Mat_Group": base["group"],
      "Mat_Layers": base["layers"],
      "Mat_MetalFraction": base["metal_fraction"],
    }

  if material in MULTI_B_FRACTION:
    return {
      "Mat_Group": "carbon_2d",
      "Mat_Layers": 1,
      "Mat_MetalFraction": MULTI_B_FRACTION[material],
    }

  if GROUP_4B_TMD.match(material):
    return {"Mat_Group": "tmd_4b", "Mat_Layers": 1, "Mat_MetalFraction": 0.33}

  if MOTE2_DECORATED.match(material):
    return {"Mat_Group": "tmd_decorated", "Mat_Layers": 1, "Mat_MetalFraction": 0.36}

  if COINAGE_MOS2.match(material):
    frac = 0.34 if "2" in material else 0.33
    return {"Mat_Group": "tmd_substituted", "Mat_Layers": 1, "Mat_MetalFraction": frac}

  if "B-pattern Graphene" in material:
    return {"Mat_Group": "carbon_2d", "Mat_Layers": 1, "Mat_MetalFraction": 0.02}

  if "Graphene" in material or "ZGNR" in material:
    frac = 0.02 if GRAPHENE_DOPED.search(material) or (doping and doping != "NA") else 0.0
    group = "carbon_1d" if "ZGNR" in material or "Nanoribbon" in material else "carbon_2d"
    return {"Mat_Group": group, "Mat_Layers": 1, "Mat_MetalFraction": frac}

  if "MoS2" in material or "WS2" in material or "MoSSe" in material or "MoTe2" in material:
    return {"Mat_Group": "tmd", "Mat_Layers": 1, "Mat_MetalFraction": 0.33}

  if material_class in MATERIAL_CLASS_DEFAULTS:
    group, layers, frac = MATERIAL_CLASS_DEFAULTS[material_class]
    return {"Mat_Group": group, "Mat_Layers": layers, "Mat_MetalFraction": frac}

  return {"Mat_Group": "unknown", "Mat_Layers": "NA", "Mat_MetalFraction": "NA"}


def material_properties(material: str, material_class: str = "", doping: str = "NA") -> dict:
  return infer_material_descriptors(material, material_class, doping)


def main():
  if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Run build_dataset.py first. Missing {INPUT_CSV}")

  df = pd.read_csv(INPUT_CSV)
  cache = load_gas_cache()

  gas_rows = []
  for gas in sorted(df["Gas"].dropna().unique()):
    gas_rows.append({"Gas": gas, **pubchem_gas_properties(gas, cache)})
  save_gas_cache(cache)

  gas_df = pd.DataFrame(gas_rows)
  enriched = df.merge(gas_df, on="Gas", how="left")

  mat_props = []
  for material in enriched["Material"].unique():
    subset = enriched.loc[enriched["Material"] == material].iloc[0]
    mat_class = str(subset.get("Material_Class", "") or "")
    doping = str(subset.get("Doping", "NA") or "NA")
    mat_props.append(
      {"Material": material, **material_properties(material, mat_class, doping)}
    )
  mat_df = pd.DataFrame(mat_props)
  enriched = enriched.merge(mat_df, on="Material", how="left")

  enriched.to_csv(OUTPUT_CSV, index=False)
  unknown = (enriched["Mat_Group"] == "unknown").sum()
  print(f"Wrote {len(enriched)} enriched rows to {OUTPUT_CSV}")
  print(f"Gas descriptors cached: {len(cache)} gases in {GAS_CACHE}")
  print(f"Materials with descriptors: {enriched['Material'].nunique()}")
  print(f"Rows with unknown material group: {unknown}")


if __name__ == "__main__":
  main()
