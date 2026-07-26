"""OptoStack prediction pipeline — train + predict Type for absorber/ETL/HTL.

Pipeline order:
  1) Lookup Eg+χ → compute Type (physics)
  2) Else ML/formula estimator for missing Eg/χ → physics Type
  3) Type-ML only if needed; optional --llm (off by default)

Usage:
  python scripts/predict_stack.py --train
  python scripts/predict_stack.py --absorber K2TiI6 --etl TiO2 --htl MoO3
  python scripts/predict_stack.py --absorber Cs2AgBiBr6 --etl TiO2 --htl NiO --llm
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from formula_parse import (  # noqa: E402
    canonicalize_material_alias,
    formula_feature_dict,
    parse_formula_counts as _parse_formula_counts,
)
from literature_bands import Layer, optoelectronic_suitability, stack_row  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = DATA / "models"
MODELS.mkdir(exist_ok=True)

STACK_PATH = DATA / "perovskite_stack_dataset.csv"
ABS_PATH = DATA / "perovskite_absorber_library.csv"
ETL_PATH = DATA / "etl_material_library.csv"
HTL_PATH = DATA / "htl_material_library.csv"
LAYER_PROPS = DATA / "layer_properties.csv"
OPTO_PATH = DATA / "opto_literature_dataset.csv"
EG_MODEL = MODELS / "perovskite_eg_regressor.joblib"
TYPE_MODEL = MODELS / "stack_type_classifier.joblib"
LAYER_CACHE = MODELS / "layer_lookup.json"
META_OUT = MODELS / "train_meta.json"

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_MONOLAYER_PREFIX = re.compile(r"^[12][TH]-", re.I)
_DASHES = str.maketrans("−–—", "---")

PEROVSKITE_ABSORBER_CLASSES = {
    "bulk_thin_film_device",
    "halide_double_perovskite_absorber",
    "oxide_double_perovskite_absorber",
    "vacancy_ordered_halide_double_perovskite",
    "halide_perovskite_alloy",
    "cs3sb2x9_perovskite_inspired",
    "lead_halide_perovskite_absorber",
}
NON_PEROVSKITE_ABSORBER_CLASSES = {
    "2d_monolayer_vdw",
    "2d_exfoliated_mps3",
}
# Extended in perovskite_rules.NON_PEROVSKITE_ABSORBERS; keep local union for fast path
DENYLIST_ABSORBERS = {"BeSiP2", "GaAs", "CdTe", "CdSe", "Si", "Ge", "CIGS", "CZTS", "InP"}

# Known indirect-gap absorbers — flag in notes; do not claim YES without caveat
INDIRECT_GAP_MATERIALS = {
    "Cs2AgBiBr6",
    "Cs2AgBiCl6",
    "Cs2AgBiI6",
    "Cs2TiBr6",
    "Cs3Bi2I9",
}

# Degenerate / metallic polymer HTLs — Eg-based Anderson Type is unreliable
DEGENERATE_HTL_MATERIALS = {
    "PEDOT:PSS",
    "PEDOT",
    "PEDOTPSS",
}

# Compact anatase TiO2 preferred for perovskite device tables (vs BeSiP2 SCAPS 3.4 eV)
TIO2_DEVICE_EG_EV = 3.2
TIO2_DEVICE_CHI_EV = 4.0

# Common contact-layer misplacements (absorber field) — also loaded from ETL/HTL libraries.
KNOWN_ETL_ABSORBERS = {
    "ZnO",
    "TiO2",
    "SnO2",
    "CdS",
    "CdZnS",
    "Nb2O5",
    "PCBM",
    "PC60BM",
    "C60",
    "WS2",
    "ZnSe",
    "SnS2",
    "BaSnO3",
    "LBSO",
}
KNOWN_HTL_ABSORBERS = {
    "MoO3",
    "NiO",
    "CuPc",
    "C6PcH2",
    "CuI",
    "CuSCN",
    "PTAA",
    "Spiro-OMeTAD",
    "PEDOT:PSS",
    "PEDOT",
    "V2O5",
    "CuAlO2",
    "TiO2:N",
    "Cu2O",
}
MATERIAL_ROLE_ALIASES = {
    "cupc": "CuPc",
    "c6pch2": "C6PcH2",
    "lbso": "LBSO",
    "lasno3": "BaSnO3",
    "c60": "C60",
    "spiro": "Spiro-OMeTAD",
    "spiro-ometad": "Spiro-OMeTAD",
    "spiroometad": "Spiro-OMeTAD",
}

_STACK_INDEX: dict[tuple[str, str, str], dict[str, str]] | None = None
_MATERIAL_REGISTRY: dict[str, dict] | None = None
_CONTACT_ROLE_INDEX: dict[str, set[str]] | None = None


def base_name(name: str) -> str:
    """Strip phase suffixes: 'Cs2AgBiBr6 (cubic)' → 'Cs2AgBiBr6'."""
    return re.sub(r"\s*\(.*\)\s*$", "", (name or "").strip())


def normalize_material_name(name: str) -> str:
    """Normalize user input: unicode dashes, FA/MA aliases, 1t- → 1T- monolayer prefix."""
    s = canonicalize_material_alias(base_name((name or "").strip()).translate(_DASHES))
    m = re.match(r"^([12])([THth])-(.*)$", s)
    if m:
        rest = m.group(3)
        s = f"{m.group(1)}{m.group(2).upper()}-{rest}"
    return s


def _put_layer(
    layers: dict[str, dict[str, float]],
    name: str,
    eg: float,
    chi: float | None = None,
    *,
    prefer_literature: bool = False,
) -> None:
    entry: dict[str, float] = {"Eg_eV": float(eg)}
    if chi is not None and not (isinstance(chi, float) and math.isnan(chi)):
        entry["chi_eV"] = float(chi)
    for key in (name, base_name(name)):
        if not key:
            continue
        prev = layers.get(key)
        if prev is None:
            layers[key] = dict(entry)
            continue
        if prefer_literature:
            # Verified experimental / SCAPS always wins for Eg; keep χ if new lacks it
            merged = dict(prev)
            merged["Eg_eV"] = entry["Eg_eV"]
            if "chi_eV" in entry:
                merged["chi_eV"] = entry["chi_eV"]
            layers[key] = merged
            continue
        # Prefer entries that already have χ; do not overwrite literature χ with estimates
        if "chi_eV" in prev and "chi_eV" not in entry:
            continue
        if "chi_eV" not in prev:
            merged = dict(prev)
            merged["Eg_eV"] = entry["Eg_eV"]
            if "chi_eV" in entry:
                merged["chi_eV"] = entry["chi_eV"]
            layers[key] = merged


def _holdout_ml_materials() -> set[str]:
    """Materials reserved for ML-only Eg testing (must not enter layer lookup)."""
    path = RAW / "verified_experimental_absorbers.csv"
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("holdout_ml") or "").strip() == "1":
                name = normalize_material_name(row.get("material") or "")
                if name:
                    out.add(name)
                    out.add(base_name(name))
    return out


def _load_verified_experimental(layers: dict[str, dict[str, float]]) -> None:
    """Overwrite DFT library gaps with DOI-backed experimental optical Eg/χ."""
    path = RAW / "verified_experimental_absorbers.csv"
    if not path.exists():
        return
    holdout = _holdout_ml_materials()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = normalize_material_name(row.get("material") or "")
            if not name or not row.get("absorber_band_gap_eV"):
                continue
            if name in holdout or base_name(name) in holdout:
                continue
            chi = float(row["chi_eV"]) if row.get("chi_eV") else None
            _put_layer(
                layers,
                name,
                float(row["absorber_band_gap_eV"]),
                chi,
                prefer_literature=True,
            )
            aliases = {
                "HC(NH2)2PbI3": "FAPbI3",
                "HC(NH2)2PbBr3": "FAPbBr3",
                "HC(NH2)2SnI3": "FASnI3",
                "CH3NH3PbI3": "MAPbI3",
                "CH3NH3PbBr3": "MAPbBr3",
                "CH3NH3SnI3": "MASnI3",
                "CH3NH3GeI3": "MAGeI3",
            }
            if name in aliases:
                _put_layer(
                    layers,
                    aliases[name],
                    float(row["absorber_band_gap_eV"]),
                    chi,
                    prefer_literature=True,
                )

def _load_verified_contacts(layers: dict[str, dict[str, float]]) -> None:
    path = RAW / "verified_contact_layers.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = normalize_material_name(row.get("material") or "")
            if not name or not row.get("Eg_eV"):
                continue
            chi = float(row["chi_eV"]) if row.get("chi_eV") else None
            _put_layer(layers, name, float(row["Eg_eV"]), chi, prefer_literature=True)


def load_layer_lookup() -> dict[str, dict[str, float]]:
    """Build Eg+χ lookup from libraries + raw SCAPS (stable, name-normalized)."""
    layers: dict[str, dict[str, float]] = {}

    # Prefer prebuilt unified table when present
    if LAYER_PROPS.exists():
        holdout = _holdout_ml_materials()
        with LAYER_PROPS.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("material") or not row.get("Eg_eV"):
                    continue
                nname = normalize_material_name(row["material"])
                if nname in holdout or base_name(nname) in holdout:
                    continue
                chi = float(row["chi_eV"]) if row.get("chi_eV") else None
                _put_layer(layers, nname, float(row["Eg_eV"]), chi)

    if ABS_PATH.exists():
        holdout = _holdout_ml_materials()
        with ABS_PATH.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("material_absorber")
                if not name or not row.get("absorber_band_gap_eV"):
                    continue
                nname = normalize_material_name(name)
                if nname in holdout or base_name(nname) in holdout:
                    continue
                chi = float(row["chi_eV"]) if row.get("chi_eV") else None
                prefer = (row.get("record_type") or "") == "verified_external" or (
                    row.get("gap_method") or ""
                ).startswith("experimental")
                _put_layer(
                    layers,
                    nname,
                    float(row["absorber_band_gap_eV"]),
                    chi,
                    prefer_literature=prefer,
                )

    for path in (ETL_PATH, HTL_PATH):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("material") or not row.get("Eg_eV"):
                    continue
                chi = float(row["chi_eV"]) if row.get("chi_eV") else None
                prefer = (row.get("record_type") or "") in (
                    "verified_contact",
                    "etl_library",
                    "htl_library",
                )
                _put_layer(
                    layers,
                    normalize_material_name(row["material"]),
                    float(row["Eg_eV"]),
                    chi,
                    prefer_literature=prefer,
                )

    for fname in (
        "paper4_scaps_materials.csv",
        "paper_cs_pb_scaps_materials.csv",
        "paper_cs3sb2br9_scaps_materials.csv",
        "paper_besip2_scaps_materials.csv",
        "paper_k2gei6_dft_absorber.csv",
    ):
        path = RAW / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("material") or not row.get("Eg_eV"):
                    continue
                chi = float(row["chi_eV"]) if row.get("chi_eV") else None
                _put_layer(
                    layers,
                    normalize_material_name(row["material"]),
                    float(row["Eg_eV"]),
                    chi,
                    prefer_literature=bool(chi is not None),
                )

    _load_monolayer_tables(layers)
    _load_opto_literature_layers(layers)
    # Experimental optical gaps overwrite DFT library values last
    _load_verified_experimental(layers)
    _load_verified_contacts(layers)

    # Prefer compact-anatase TiO2 Eg≈3.2 eV for perovskite device screening.
    # BeSiP2 SCAPS tables used 3.4 eV (alternate phase/table) — keep that in raw CSV.
    for key in ("TiO2",):
        if key in layers:
            layers[key] = {
                **layers[key],
                "Eg_eV": TIO2_DEVICE_EG_EV,
                "chi_eV": float(layers[key].get("chi_eV", TIO2_DEVICE_CHI_EV)),
            }

    # Alias FA/MA for verified lead-halides — always overwrite short keys from
    # canonical formulas so a stale FAPbBr3≠FAPbI3 (halide) collision cannot stick.
    for full, short in (
        ("HC(NH2)2PbI3", "FAPbI3"),
        ("HC(NH2)2PbBr3", "FAPbBr3"),
        ("HC(NH2)2PbCl3", "FAPbCl3"),
        ("CH3NH3PbI3", "MAPbI3"),
        ("CH3NH3PbBr3", "MAPbBr3"),
        ("CH3NH3PbCl3", "MAPbCl3"),
        ("HC(NH2)2SnI3", "FASnI3"),
        ("CH3NH3SnI3", "MASnI3"),
    ):
        if full in layers:
            layers[short] = dict(layers[full])

    LAYER_CACHE.write_text(json.dumps(layers, indent=2), encoding="utf-8")
    return layers


def _load_monolayer_tables(layers: dict[str, dict[str, float]]) -> None:
    """Index Paper1 / Özçelik HSE06 monolayers (Eg + χ from VBM)."""
    for fname, vbm_col, eg_col in (
        ("paper1_table2_monolayers.csv", "ev_vbm_eV", "eg_hse06_eV"),
        ("ozcelik_prb2016_monolayers.csv", "ev_vbm_hse_eV", "eg_hse_eV"),
    ):
        path = RAW / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("material") or not row.get(eg_col) or not row.get(vbm_col):
                    continue
                name = normalize_material_name(row["material"])
                eg = float(row[eg_col])
                vbm = float(row[vbm_col])
                chi = float(Layer.from_vbm_eg(name, vbm, eg).chi)
                _put_layer(layers, name, eg, chi, prefer_literature=True)


def _load_opto_literature_layers(layers: dict[str, dict[str, float]]) -> None:
    """Index unique Eg/χ per material from the full literature stack dataset."""
    if not OPTO_PATH.exists():
        return
    eg_by_mat: dict[str, list[float]] = {}
    with OPTO_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for mat_col, eg_col in (
                ("material_absorber", "absorber_band_gap_eV"),
                ("material_etl", "etl_band_gap_eV"),
                ("material_htl", "htl_band_gap_eV"),
            ):
                mat = normalize_material_name(row.get(mat_col) or "")
                if not mat or not row.get(eg_col):
                    continue
                eg_by_mat.setdefault(mat, []).append(float(row[eg_col]))
    for mat, egs in eg_by_mat.items():
        eg = float(np.median(egs))
        prev = layers.get(mat)
        if prev and "chi_eV" in prev:
            _put_layer(layers, mat, eg, float(prev["chi_eV"]), prefer_literature=True)
        elif prev is None:
            _put_layer(layers, mat, eg, None)


def load_material_registry() -> dict[str, dict]:
    """Material metadata for perovskite eligibility (library + literature + SCAPS)."""
    global _MATERIAL_REGISTRY
    if _MATERIAL_REGISTRY is not None:
        return _MATERIAL_REGISTRY

    reg: dict[str, dict] = {}

    def _touch(name: str) -> dict:
        key = normalize_material_name(name)
        return reg.setdefault(key, {})

    if ABS_PATH.exists():
        with ABS_PATH.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_material_name(row.get("material_absorber") or "")
                if not name:
                    continue
                entry = _touch(name)
                entry["in_perovskite_library"] = True
                entry["is_perovskite_absorber"] = True
                entry["material_class"] = row.get("material_class") or entry.get("material_class")
                entry["perovskite_family"] = row.get("perovskite_family") or entry.get(
                    "perovskite_family"
                )

    for fname in (
        "paper4_scaps_materials.csv",
        "paper_cs_pb_scaps_materials.csv",
        "paper_cs3sb2br9_scaps_materials.csv",
        "paper_besip2_scaps_materials.csv",
        "paper_k2gei6_dft_absorber.csv",
    ):
        path = RAW / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_material_name(row.get("material") or "")
                if not name:
                    continue
                role = (row.get("layer_role") or "").strip().lower()
                if role == "absorber" or name == "K2GeI6":
                    entry = _touch(name)
                    entry["scaps_absorber"] = True
                    if name not in DENYLIST_ABSORBERS:
                        entry["is_perovskite_absorber"] = True
                    entry.setdefault("material_class", "bulk_thin_film_device")

    if OPTO_PATH.exists():
        with OPTO_PATH.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mclass = (row.get("material_class") or "").strip()
                mclass_l = mclass.lower()
                for mat_col, role in (
                    ("material_absorber", "absorber"),
                    ("material_etl", "etl"),
                    ("material_htl", "htl"),
                ):
                    name = normalize_material_name(row.get(mat_col) or "")
                    if not name:
                        continue
                    entry = _touch(name)
                    roles = entry.setdefault("roles", set())
                    roles.add(role)
                    if role == "absorber" and mclass:
                        entry["material_class"] = mclass
                        if mclass_l in NON_PEROVSKITE_ABSORBER_CLASSES:
                            entry["is_perovskite_absorber"] = False
                        elif mclass_l in {c.lower() for c in PEROVSKITE_ABSORBER_CLASSES}:
                            entry["is_perovskite_absorber"] = True

    for fname in ("paper1_table2_monolayers.csv", "ozcelik_prb2016_monolayers.csv"):
        path = RAW / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_material_name(row.get("material") or "")
                if not name:
                    continue
                entry = _touch(name)
                entry["material_class"] = "2D_monolayer_vdW"
                entry["is_perovskite_absorber"] = False

    for path, role in ((ETL_PATH, "etl"), (HTL_PATH, "htl")):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_material_name(row.get("material") or "")
                if not name:
                    continue
                entry = _touch(name)
                roles = entry.setdefault("roles", set())
                roles.add(role)
                entry[f"is_{role}_contact"] = True

    verified_abs = RAW / "verified_experimental_absorbers.csv"
    if verified_abs.exists():
        with verified_abs.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_material_name(row.get("material") or "")
                if not name:
                    continue
                entry = _touch(name)
                entry["is_perovskite_absorber"] = True
                entry["in_perovskite_library"] = True
                entry["material_class"] = row.get("material_class") or entry.get("material_class")
                entry["perovskite_family"] = row.get("perovskite_family") or entry.get(
                    "perovskite_family"
                )
                if row.get("gap_type"):
                    entry["gap_type"] = row["gap_type"]

    _MATERIAL_REGISTRY = reg
    return reg


def parse_formula_counts(formula: str) -> dict[str, float]:
    return _parse_formula_counts(formula)


def _canonical_contact_key(name: str) -> str:
    key = base_name(normalize_material_name(name)).lower().replace(" ", "")
    return MATERIAL_ROLE_ALIASES.get(key, base_name(normalize_material_name(name)))


def load_contact_role_index() -> dict[str, set[str]]:
    """Materials indexed as ETL/HTL contact layers (never perovskite absorbers)."""
    global _CONTACT_ROLE_INDEX
    if _CONTACT_ROLE_INDEX is not None:
        return _CONTACT_ROLE_INDEX

    roles: dict[str, set[str]] = {}
    for mat in KNOWN_ETL_ABSORBERS:
        roles.setdefault(_canonical_contact_key(mat), set()).add("etl")
    for mat in KNOWN_HTL_ABSORBERS:
        roles.setdefault(_canonical_contact_key(mat), set()).add("htl")

    for path, role in ((ETL_PATH, "etl"), (HTL_PATH, "htl")):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_material_name(row.get("material") or "")
                if name:
                    roles.setdefault(_canonical_contact_key(name), set()).add(role)

    for fname in (
        "paper4_scaps_materials.csv",
        "paper_cs_pb_scaps_materials.csv",
        "paper_cs3sb2br9_scaps_materials.csv",
        "paper_besip2_scaps_materials.csv",
    ):
        path = RAW / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                layer_role = (row.get("layer_role") or "").strip().lower()
                if layer_role not in ("etl", "htl"):
                    continue
                name = normalize_material_name(row.get("material") or "")
                if name:
                    roles.setdefault(_canonical_contact_key(name), set()).add(layer_role)

    _CONTACT_ROLE_INDEX = roles
    return roles


def _is_simple_oxide_semiconductor(formula: str) -> bool:
    """Binary/ternary oxides without halides — typical ETLs, not perovskite absorbers."""
    counts = parse_formula_counts(formula)
    if not counts or counts.get("O", 0) <= 0:
        return False
    halogens = sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    if halogens > 0:
        return False
    metals = {el for el in counts if el not in ("H", "C", "N", "O", "F", "Cl", "Br", "I")}
    o_count = counts.get("O", 0)
    # Binary oxide (ZnO, TiO2, SnO2, MoO3, NiO)
    if len(counts) == 2 and o_count > 0:
        return True
    # Simple ternary oxide without double-perovskite stoichiometry (BaSnO3, not A2BB'O6)
    if len(metals) <= 2 and o_count <= 3:
        return True
    return False


def looks_like_perovskite_absorber(formula: str) -> bool:
    """Heuristic: formula matches known perovskite-inspired absorber families."""
    try:
        from perovskite_rules import looks_like_perovskite_absorber as _rules_looks

        return _rules_looks(formula)
    except Exception:
        pass

    counts = parse_formula_counts(formula)
    if not counts:
        return False

    name_clean = base_name(formula).replace(".", "").replace(" ", "")
    halogens = sum(counts.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    pb = counts.get("Pb", 0)
    o_count = counts.get("O", 0)
    metals = {el for el in counts if el not in ("H", "C", "N", "O", "F", "Cl", "Br", "I")}

    if counts.get("C", 0) > 0 and pb == 0 and halogens == 0:
        return False

    b_site = counts.get("Pb", 0) + counts.get("Sn", 0) + counts.get("Ge", 0)
    if b_site > 0 and halogens >= 2 and o_count == 0:
        return True

    if halogens >= 6 and (
        counts.get("Ag", 0)
        or counts.get("Bi", 0)
        or counts.get("Cu", 0)
        or counts.get("In", 0)
        or counts.get("Sb", 0)
        or counts.get("Ti", 0)
        or counts.get("Sn", 0)
    ):
        return True

    if re.search(r".*3.*2.*9", name_clean) and halogens >= 6:
        return True

    if halogens >= 6 and re.search(r".*2.*6$", name_clean):
        return True

    if o_count >= 6 and len(metals) >= 4:
        return True

    return False


def _contact_layer_block(
    absorber: str, layers: dict[str, dict[str, float]] | None = None
) -> dict | None:
    """Block when absorber is a known ETL/HTL contact material."""
    bn = base_name(normalize_material_name(absorber))
    key = _canonical_contact_key(bn)
    contact_roles = load_contact_role_index().get(key, set())
    if not contact_roles:
        return None

    layers = layers if layers is not None else load_layer_lookup()
    entry = resolve_layer(layers, bn) or {}
    eg = entry.get("Eg_eV")
    eg_note = f" (Eg≈{float(eg):.1f} eV)" if eg is not None else ""

    if "etl" in contact_roles and "htl" not in contact_roles:
        oxide = _is_simple_oxide_semiconductor(bn) or bn.endswith("O") or "O" in parse_formula_counts(bn)
        kind = "oxide ETL" if oxide else "ETL"
        return {
            "eligible": False,
            "not_perovskite": True,
            "message": f"{bn} is {('an' if kind[0] in 'aeiou' else 'a')} {kind}{eg_note}, not a perovskite absorber",
            "hint": f"Try entering {bn} in the ETL field instead of Absorber.",
            "misplaced_role": "etl",
        }
    if "htl" in contact_roles:
        kind = "HTL" if bn not in KNOWN_ETL_ABSORBERS else "contact layer"
        return {
            "eligible": False,
            "not_perovskite": True,
            "message": f"{bn} is {('an' if kind[0] in 'aeiou' else 'a')} {kind}{eg_note}, not a perovskite absorber",
            "hint": f"Try entering {bn} in the HTL field instead of Absorber.",
            "misplaced_role": "htl",
        }
    return None


def check_absorber_perovskite(absorber: str) -> dict:
    """Return eligibility for perovskite-stack screening (absorber must be perovskite-class)."""
    name = normalize_material_name(absorber)
    bn = base_name(name)

    # Taxonomy gate (families, contacts, non-perovskites, 2D RP/DJ)
    try:
        from perovskite_rules import classify_family

        fam = classify_family(bn)
        if fam.family_id == "non_perovskite" or bn in DENYLIST_ABSORBERS:
            thin_film = bn in {"CZTS", "CIGS", "CdTe", "GaAs", "Si", "Ge", "InP", "BeSiP2"}
            extra = (
                " Non-perovskite thin-film / III-V / chalcogenide absorbers (e.g. CZTS, CIGS, CdTe) "
                "are intentionally blocked -- this tool screens perovskite absorbers only."
                if thin_film
                else ""
            )
            return {
                "eligible": False,
                "not_perovskite": True,
                "family_id": fam.family_id,
                "message": (
                    f"{bn} is not a perovskite absorber -- screening is for perovskite stacks only."
                    f"{extra}"
                ),
            }
        if fam.family_id == "rp_dj_2d" or _MONOLAYER_PREFIX.match(bn):
            return {
                "eligible": False,
                "not_perovskite": True,
                "family_id": fam.family_id,
                "message": (
                    f"{bn} is a 2D monolayer / RP–DJ phase, not a standard 3D perovskite absorber — "
                    "screening is for perovskite stacks only"
                ),
            }
        if fam.family_id == "contact_etl":
            return {
                "eligible": False,
                "not_perovskite": True,
                "family_id": fam.family_id,
                "message": f"{bn} is an ETL contact layer, not a perovskite absorber",
                "hint": f"Try entering {bn} in the ETL field instead of Absorber.",
                "misplaced_role": "etl",
            }
        if fam.family_id == "contact_htl":
            return {
                "eligible": False,
                "not_perovskite": True,
                "family_id": fam.family_id,
                "message": f"{bn} is an HTL contact layer, not a perovskite absorber",
                "hint": f"Try entering {bn} in the HTL field instead of Absorber.",
                "misplaced_role": "htl",
            }
    except Exception:
        if bn in DENYLIST_ABSORBERS:
            thin_film = bn in {"CZTS", "CIGS", "CdTe", "GaAs", "Si", "Ge", "InP", "BeSiP2"}
            extra = (
                " Non-perovskite thin-film / III-V / chalcogenide absorbers (e.g. CZTS, CIGS, CdTe) "
                "are intentionally blocked -- this tool screens perovskite absorbers only."
                if thin_film
                else ""
            )
            return {
                "eligible": False,
                "not_perovskite": True,
                "message": (
                    f"{bn} is not a perovskite absorber -- screening is for perovskite stacks only."
                    f"{extra}"
                ),
            }
        if _MONOLAYER_PREFIX.match(bn):
            return {
                "eligible": False,
                "not_perovskite": True,
                "message": (
                    f"{bn} is a 2D monolayer (vdW), not a perovskite absorber — "
                    "screening is for perovskite stacks only"
                ),
            }

    reg = load_material_registry()
    info = reg.get(name) or reg.get(bn) or {}
    mclass = (info.get("material_class") or "").lower()

    if mclass in NON_PEROVSKITE_ABSORBER_CLASSES:
        label = info.get("material_class") or mclass
        return {
            "eligible": False,
            "not_perovskite": True,
            "message": (
                f"{bn} ({label}) is not a perovskite absorber — "
                "screening is for perovskite stacks only"
            ),
        }

    contact_block = _contact_layer_block(absorber)
    if contact_block:
        return contact_block

    roles = info.get("roles") or set()
    if roles == {"etl"} or (roles & {"etl", "htl"} and "absorber" not in roles):
        if "htl" in roles and "etl" not in roles:
            return _contact_layer_block(absorber) or {
                "eligible": False,
                "not_perovskite": True,
                "message": f"{bn} is a contact-layer HTL, not a perovskite absorber",
                "hint": f"Try entering {bn} in the HTL field instead of Absorber.",
            }
        return _contact_layer_block(absorber) or {
            "eligible": False,
            "not_perovskite": True,
            "message": f"{bn} is a contact-layer ETL, not a perovskite absorber",
            "hint": f"Try entering {bn} in the ETL field instead of Absorber.",
        }

    if info.get("is_perovskite_absorber") or info.get("in_perovskite_library"):
        return {"eligible": True}
    if info.get("scaps_absorber"):
        return {"eligible": True}
    if mclass in {c.lower() for c in PEROVSKITE_ABSORBER_CLASSES}:
        return {"eligible": True}

    # Oxide perovskites (BaTiO3, double oxides): eligible with caution — not primary halide screening
    try:
        from perovskite_rules import classify_family

        fam = classify_family(bn)
        if fam.family_id == "oxide_perovskite" and fam.absorber_eligible:
            return {
                "eligible": True,
                "family_id": fam.family_id,
                "warning": (
                    f"{bn} looks like an oxide perovskite (wide-gap). "
                    "Eligible for estimates, but not a primary halide-absorber screening target."
                ),
            }
    except Exception:
        pass

    if _is_simple_oxide_semiconductor(bn):
        layers = load_layer_lookup()
        entry = resolve_layer(layers, bn) or {}
        eg = entry.get("Eg_eV")
        eg_note = f" (Eg≈{float(eg):.1f} eV)" if eg is not None else ""
        return {
            "eligible": False,
            "not_perovskite": True,
            "message": (
                f"{bn} is an oxide semiconductor{eg_note}, not a perovskite absorber"
            ),
            "hint": (
                f"If {bn} is your electron-transport layer, enter it in the ETL field."
            ),
            "misplaced_role": "etl",
        }

    if looks_like_perovskite_absorber(bn):
        fam_id = None
        try:
            from perovskite_rules import classify_family

            fam_id = classify_family(bn).family_id
        except Exception:
            pass
        out = {
            "eligible": True,
            "warning": (
                f"{bn} is not in the perovskite absorber library; "
                "predictions are estimates only."
            ),
        }
        if fam_id:
            out["family_id"] = fam_id
            if fam_id == "oxide_perovskite":
                out["warning"] += " Oxide perovskite — wide-gap; low confidence for halide screening."
        return out

    return {
        "eligible": False,
        "not_perovskite": True,
        "message": (
            f"{bn} does not match perovskite absorber families (ABX₃, A₂BB′X₆, "
            "A₂BX₆ halides, etc.) — screening is for perovskite stacks only"
        ),
    }


def load_literature_stack_index() -> dict[tuple[str, str, str], dict[str, str]]:
    """Exact (absorber, ETL, HTL) rows from literature / SCAPS stack tables."""
    global _STACK_INDEX
    if _STACK_INDEX is not None:
        return _STACK_INDEX

    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in (
        OPTO_PATH,
        STACK_PATH,
        DATA / "opto_literature_dataset_scaps_only.csv",
    ):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("material_absorber") or not row.get("absorber_etl_type"):
                    continue
                key = (
                    normalize_material_name(row["material_absorber"]),
                    normalize_material_name(row["material_etl"]),
                    normalize_material_name(row["material_htl"]),
                )
                # Prefer first occurrence; SCAPS/stack tables are authoritative for Types
                if key not in index:
                    index[key] = row
    _STACK_INDEX = index
    return index


def lookup_literature_stack(absorber: str, etl: str, htl: str) -> dict[str, str] | None:
    key = (
        normalize_material_name(absorber),
        normalize_material_name(etl),
        normalize_material_name(htl),
    )
    return load_literature_stack_index().get(key)


def resolve_layer(
    layers: dict[str, dict[str, float]], name: str
) -> dict[str, float] | None:
    """Exact name, normalized name, base-name, then case-insensitive exact match.

    Never prefix-matches (FAPb must not resolve to FAPbI3 when querying FAPbBr3).
    """
    name = normalize_material_name(name)
    if name in layers:
        return layers[name]
    bn = base_name(name)
    if bn in layers:
        return layers[bn]
    low = name.lower()
    bn_low = bn.lower()
    for k, v in layers.items():
        kl = k.lower()
        if kl == low or base_name(k).lower() == bn_low:
            return v
    return None


def _is_degenerate_htl(name: str) -> bool:
    key = base_name(normalize_material_name(name)).upper().replace(" ", "").replace("-", "")
    canon = {m.upper().replace(" ", "").replace("-", "").replace(":", "") for m in DEGENERATE_HTL_MATERIALS}
    key_nocolon = key.replace(":", "")
    if key_nocolon in canon or key in canon:
        return True
    return "PEDOT" in key


def _apply_degenerate_htl_caveat(result: dict, htl: str) -> None:
    """Annotate stacks whose HTL is a degenerate/metallic polymer (PEDOT:PSS)."""
    if not _is_degenerate_htl(htl):
        return
    note = (
        f"gap_type: degenerate/metallic HTL — {base_name(normalize_material_name(htl))} "
        "is a highly doped polymer; Eg-based Anderson Type is unreliable for this contact."
    )
    notes = list(result.get("notes") or [])
    if note not in notes:
        notes.append(note)
    result["notes"] = notes
    opto = result.get("optoelectronic") or {}
    opto["gap_type"] = opto.get("gap_type") or "degenerate_htl"
    opto["htl_caveat"] = "degenerate_metallic"
    reason = opto.get("reason") or ""
    caveat = (
        f" Caveat: {base_name(normalize_material_name(htl))} is a degenerate/metallic HTL — "
        "Eg-based junction Type and YES/MARGINAL suitability should be treated with low confidence."
    )
    if "degenerate/metallic HTL" not in reason:
        opto["reason"] = reason + caveat
    # Soften Type-ML confidence markers when present
    if "predicted_absorber_htl_proba" in result:
        result["predicted_absorber_htl_proba_note"] = (
            "Type-ML confidence suppressed for degenerate HTL (PEDOT family)."
        )
        result.pop("predicted_absorber_htl_proba", None)
    result["optoelectronic"] = opto
    result["caution"] = True

def formula_features(formula: str) -> dict[str, float]:
    return formula_feature_dict(formula)


def _feature_frame(formulas: list[str]) -> tuple[list[str], np.ndarray]:
    dicts = [formula_features(f) for f in formulas]
    keys = sorted({k for d in dicts for k in d})
    X = np.zeros((len(dicts), len(keys)), dtype=float)
    for i, d in enumerate(dicts):
        for j, k in enumerate(keys):
            X[i, j] = d.get(k, 0.0)
    return keys, X


VERIFIED_EG_OVERSAMPLE = 10


def train_eg_model() -> dict:
    rows = list(csv.DictReader(ABS_PATH.open(encoding="utf-8")))
    expanded: list[dict] = []
    for r in rows:
        expanded.append(r)
        if r.get("record_type") == "verified_external":
            expanded.extend([r] * (VERIFIED_EG_OVERSAMPLE - 1))
    formulas = [r["material_absorber"] for r in expanded]
    y = np.array([float(r["absorber_band_gap_eV"]) for r in expanded], dtype=float)
    keys, X = _feature_frame(formulas)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)
    mae = float(mean_absolute_error(yte, model.predict(Xte)))
    cv = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    payload = {
        "model": model,
        "element_keys": keys,
        "holdout_mae": mae,
        "cv_mae": float(-cv.mean()),
    }
    joblib.dump(payload, EG_MODEL)
    return {
        "holdout_mae_eV": mae,
        "cv_mae_eV": float(-cv.mean()),
        "n": len(y),
        "path": str(EG_MODEL),
    }


def train_type_models() -> dict:
    """Train Type classifiers on expanded perovskite stacks (+ optional SCAPS)."""
    rows: list[dict] = []
    for p in (STACK_PATH, DATA / "opto_literature_dataset_scaps_only.csv"):
        if p.exists():
            rows.extend(csv.DictReader(p.open(encoding="utf-8")))

    def pack(target: str):
        X_mat, y = [], []
        for r in rows:
            if not r.get(target):
                continue
            eg_a = float(r["absorber_band_gap_eV"])
            eg_p = float(
                r["etl_band_gap_eV"] if "etl" in target else r["htl_band_gap_eV"]
            )
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
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["absorber", "partner"]),
                ("num", StandardScaler(), ["eg_a", "eg_p", "eg_diff", "abs_frac_I", "abs_has_organic"]),
            ]
        )
        clf = Pipeline(
            [
                ("pre", pre),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        )
        df = pd.DataFrame(X_mat)
        ys = np.array(y)
        Xtr, Xte, ytr, yte = train_test_split(
            df, ys, test_size=0.2, random_state=42, stratify=ys
        )
        clf.fit(Xtr, ytr)
        acc = float(accuracy_score(yte, clf.predict(Xte)))
        clf.fit(df, ys)
        return clf, acc, len(ys)

    etl_clf, etl_acc, n_etl = pack("absorber_etl_type")
    htl_clf, htl_acc, n_htl = pack("absorber_htl_type")
    payload = {
        "etl": etl_clf,
        "htl": htl_clf,
        "etl_holdout_acc": etl_acc,
        "htl_holdout_acc": htl_acc,
        "n_etl": n_etl,
        "n_htl": n_htl,
        "train_stacks": str(STACK_PATH),
    }
    joblib.dump(payload, TYPE_MODEL)
    return {
        "etl_holdout_acc": etl_acc,
        "htl_holdout_acc": htl_acc,
        "n_etl": n_etl,
        "n_htl": n_htl,
        "path": str(TYPE_MODEL),
    }


def predict_eg(formula: str) -> float:
    if not EG_MODEL.exists():
        raise FileNotFoundError("Train first: python scripts/predict_stack.py --train")
    payload = joblib.load(EG_MODEL)
    keys: list[str] = payload["element_keys"]
    feats = formula_features(formula)
    x = np.array([[feats.get(k, 0.0) for k in keys]], dtype=float)
    return float(payload["model"].predict(x)[0])


def ml_type(absorber: str, partner: str, eg_a: float, eg_p: float, side: str) -> dict:
    if not TYPE_MODEL.exists():
        raise FileNotFoundError("Train first: python scripts/predict_stack.py --train")
    payload = joblib.load(TYPE_MODEL)
    clf = payload["etl" if side == "etl" else "htl"]
    fa = formula_features(absorber)
    df = pd.DataFrame(
        [
            {
                "absorber": absorber,
                "partner": partner,
                "eg_a": eg_a,
                "eg_p": eg_p,
                "eg_diff": eg_a - eg_p,
                "abs_frac_I": fa.get("frac_I", 0.0),
                "abs_has_organic": fa.get("has_organic", 0.0),
            }
        ]
    )
    pred = str(clf.predict(df)[0])
    out: dict = {"type": pred}
    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(df)[0]
            classes = list(clf.classes_)
            out["proba"] = {str(c): float(p) for c, p in zip(classes, proba)}
        except Exception:
            pass
    return out


def llm_predict_eg_chi(material: str, role: str = "absorber") -> dict:
    """Optional LLM: formula-only calculator (no web). Prefer ML instead."""
    from llm_literature_assist import predict_material

    data = predict_material(material, role, provider="auto")
    pred = data["prediction"]
    return {
        "Eg_eV": pred.get("Eg_eV"),
        "chi_eV": pred.get("chi_eV"),
        "confidence": pred.get("confidence"),
        "method": pred.get("method"),
        "provider": data["_meta"].get("provider"),
        "saved": data["_meta"].get("saved"),
    }


def ml_estimate_eg_chi(material: str, role: str) -> dict:
    from formula_estimator import estimate_eg_chi

    return estimate_eg_chi(material, role)


def _has_chi(entry: dict | None) -> bool:
    if not entry:
        return False
    chi = entry.get("chi_eV")
    return chi is not None and not (isinstance(chi, float) and math.isnan(chi))


def _src_kind(src: str | None) -> str:
    if not src:
        return "missing"
    if src in ("lookup", "literature", "literature_SCAPS", "literature_stack_row"):
        return "lookup"
    if src == "user_override":
        return "user"
    return "predicted"


def _lookup_field_labels() -> dict[str, str]:
    return {
        k: "lookup"
        for k in (
            "absorber_Eg",
            "absorber_chi",
            "etl_Eg",
            "etl_chi",
            "htl_Eg",
            "htl_chi",
            "absorber_etl_type",
            "absorber_htl_type",
            "optoelectronic",
        )
    }


def _result_from_literature_row(
    lit_row: dict[str, str],
    absorber: str,
    etl: str,
    htl: str,
    *,
    not_perovskite: bool = False,
    message: str | None = None,
    layers: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Build response from an exact opto_literature_dataset row."""
    layers = layers or load_layer_lookup()

    def _chi_for(material: str, role: str) -> tuple[float | None, str]:
        entry = resolve_layer(layers, material)
        if entry and _has_chi(entry):
            return float(entry["chi_eV"]), "lookup"
        est = ml_estimate_eg_chi(material, role)
        return float(est["chi_eV"]), est.get("source", "ml_formula_estimator")

    abs_chi, abs_chi_src = _chi_for(absorber, "absorber")
    etl_chi, etl_chi_src = _chi_for(etl, "etl")
    htl_chi, htl_chi_src = _chi_for(htl, "htl")

    sources = {
        "llm_mode": "off",
        "llm_used": False,
        "absorber_Eg": "literature_stack_row",
        "etl_Eg": "literature_stack_row",
        "htl_Eg": "literature_stack_row",
        "absorber_chi": abs_chi_src,
        "etl_chi": etl_chi_src,
        "htl_chi": htl_chi_src,
        "etl": "literature_stack_row" if etl_chi_src == "literature_stack_row" else f"literature_stack_row/{etl_chi_src}",
        "htl": "literature_stack_row" if htl_chi_src == "literature_stack_row" else f"literature_stack_row/{htl_chi_src}",
    }
    field_labels = {
        "absorber_Eg": "lookup",
        "etl_Eg": "lookup",
        "htl_Eg": "lookup",
        "absorber_chi": _src_kind(abs_chi_src),
        "etl_chi": _src_kind(etl_chi_src),
        "htl_chi": _src_kind(htl_chi_src),
        "absorber_etl_type": "lookup",
        "absorber_htl_type": "lookup",
        "optoelectronic": "lookup",
    }

    etl_type = str(lit_row["absorber_etl_type"])
    htl_type = str(lit_row["absorber_htl_type"])
    # Recompute Anderson Types from Eg+χ when affinities are available (fixes
    # legacy straddling/staggered label swap in older curated rows).
    if abs_chi is not None and etl_chi is not None and htl_chi is not None:
        from literature_bands import junction_type as _jtype

        a = Layer(absorber, float(lit_row["absorber_band_gap_eV"]), float(abs_chi))
        e = Layer(etl, float(lit_row["etl_band_gap_eV"]), float(etl_chi))
        h = Layer(htl, float(lit_row["htl_band_gap_eV"]), float(htl_chi))
        etl_type = _jtype(a, e)
        htl_type = _jtype(a, h)
    opto = optoelectronic_suitability(etl_type, htl_type)
    opto["label"] = "lookup"

    notes: list[str] = []
    abs_bn = base_name(absorber)
    if abs_bn in INDIRECT_GAP_MATERIALS:
        notes.append(
            f"gap_type: indirect — {abs_bn} is a known indirect-gap absorber; "
            "suitability YES should be read with an optical-absorption caveat."
        )
        opto["gap_type"] = "indirect"
        if opto.get("verdict") == "YES":
            opto["reason"] = (
                opto.get("reason", "")
                + f" Caveat: {abs_bn} is an indirect-gap absorber — optical absorption may be weaker than a direct-gap YES."
            )

    result: dict = {
        "material_absorber": absorber,
        "material_etl": etl,
        "material_htl": htl,
        "absorber_band_gap_eV": float(lit_row["absorber_band_gap_eV"]),
        "etl_band_gap_eV": float(lit_row["etl_band_gap_eV"]),
        "htl_band_gap_eV": float(lit_row["htl_band_gap_eV"]),
        "absorber_chi_eV": abs_chi,
        "etl_chi_eV": etl_chi,
        "htl_chi_eV": htl_chi,
        "cbo_eV": float(lit_row["cbo_eV"]),
        "vbo_eV": float(lit_row["vbo_eV"]),
        "absorber_etl_type": etl_type,
        "absorber_htl_type": htl_type,
        "method": "literature_stack_row",
        "sources": sources,
        "field_labels": field_labels,
        "notes": notes,
        "optoelectronic": opto,
        "literature_reference": {
            "source_doi": lit_row.get("source_doi"),
            "source_paper": lit_row.get("source_paper"),
            "material_class": lit_row.get("material_class"),
            "gap_method": lit_row.get("gap_method"),
        },
    }
    if not_perovskite:
        result["not_perovskite"] = True
        result["blocked"] = True
        result["screening_blocked"] = True
        result["message"] = message
        result["optoelectronic"] = {
            **opto,
            "verdict": "BLOCKED",
            "suitable": False,
            "reason": (
                f"{message} Literature junction types: absorber–ETL {etl_type}, "
                f"absorber–HTL {htl_type} (reference only)."
            ),
            "label": "lookup",
        }
    _apply_degenerate_htl_caveat(result, htl)
    return result


def _blocked_absorber_result(
    absorber: str,
    etl: str,
    htl: str,
    perovskite_check: dict,
) -> dict:
    """Blocked response — no absorber Eg/χ ML predictions."""
    message = perovskite_check["message"]
    result: dict = {
        "material_absorber": absorber,
        "material_etl": etl,
        "material_htl": htl,
        "not_perovskite": True,
        "blocked": True,
        "screening_blocked": True,
        "message": message,
        "method": "blocked_non_perovskite",
        "notes": [message],
        "optoelectronic": {
            "verdict": "BLOCKED",
            "suitable": False,
            "reason": message,
            "label": "blocked",
        },
    }
    if perovskite_check.get("hint"):
        result["hint"] = perovskite_check["hint"]
        result["notes"].append(perovskite_check["hint"])
    if perovskite_check.get("misplaced_role"):
        result["misplaced_role"] = perovskite_check["misplaced_role"]
    return result


def predict_stack(
    absorber: str,
    etl: str,
    htl: str,
    eg: float | None = None,
    chi: float | None = None,
    use_llm: bool | None = None,
) -> dict:
    """Predict Type. Order: perovskite check → literature stack → layer lookup → ML."""
    absorber = normalize_material_name(absorber)
    etl = normalize_material_name(etl)
    htl = normalize_material_name(htl)

    layers = load_layer_lookup()
    perovskite_check = check_absorber_perovskite(absorber)

    if not perovskite_check.get("eligible", True):
        return _blocked_absorber_result(absorber, etl, htl, perovskite_check)

    lit_row = lookup_literature_stack(absorber, etl, htl)

    if lit_row and eg is None and chi is None:
        return _result_from_literature_row(lit_row, absorber, etl, htl, layers=layers)

    notes: list[str] = []
    if perovskite_check.get("warning"):
        notes.append(perovskite_check["warning"])

    abs_bn = base_name(absorber)
    if abs_bn in INDIRECT_GAP_MATERIALS:
        notes.append(
            f"gap_type: indirect — {abs_bn} is a known indirect-gap absorber; "
            "suitability YES should be read with an optical-absorption caveat."
        )

    sources: dict = {}
    use_llm_flag = use_llm is True
    sources["llm_mode"] = "on" if use_llm_flag else "off"
    sources["llm_used"] = False

    abs_entry = dict(resolve_layer(layers, absorber) or {})
    estimate_meta: dict = {}

    if eg is not None:
        abs_eg = float(eg)
        sources["absorber_Eg"] = "user_override"
    elif "Eg_eV" in abs_entry:
        abs_eg = float(abs_entry["Eg_eV"])
        sources["absorber_Eg"] = "lookup"
        estimate_meta["confidence"] = "high"
        estimate_meta["caution"] = False
    else:
        est = ml_estimate_eg_chi(absorber, "absorber")
        abs_eg = float(est["Eg_eV"])
        sources["absorber_Eg"] = est.get("source", "ml_formula_estimator")
        estimate_meta.update(
            {
                k: est[k]
                for k in (
                    "confidence",
                    "caution",
                    "family_id",
                    "blend_weight_prior",
                    "prior_method",
                )
                if k in est
            }
        )
        notes.append(
            f"Estimated absorber Eg={abs_eg:.3f} eV"
            f" ({est.get('source')}; family={est.get('family_id')}; "
            f"confidence={est.get('confidence')})"
        )
        if est.get("caution"):
            notes.append(
                "Low confidence (OOD / unusual family) — treat band gap as approximate."
            )

    if chi is not None:
        abs_chi = float(chi)
        sources["absorber_chi"] = "user_override"
    elif _has_chi(abs_entry):
        abs_chi = float(abs_entry["chi_eV"])
        sources["absorber_chi"] = "lookup"
    else:
        est = ml_estimate_eg_chi(absorber, "absorber")
        abs_chi = float(est["chi_eV"])
        sources["absorber_chi"] = est.get("source", "ml_formula_estimator")
        for k in ("confidence", "caution", "family_id"):
            if k in est and k not in estimate_meta:
                estimate_meta[k] = est[k]
        # χ is used internally for Type; omit from user-facing notes

    def resolve_contact(name: str, role: str):
        entry = resolve_layer(layers, name)
        if entry and "Eg_eV" in entry and _has_chi(entry):
            return float(entry["Eg_eV"]), float(entry["chi_eV"]), "lookup", "lookup"
        if entry and "Eg_eV" in entry:
            eg_v = float(entry["Eg_eV"])
            est = ml_estimate_eg_chi(name, role)
            return eg_v, float(est["chi_eV"]), "lookup", est.get("source", "ml_formula_estimator")
        est = ml_estimate_eg_chi(name, role)
        eg_v, chi_v = float(est["Eg_eV"]), float(est["chi_eV"])
        src = est.get("source", "ml_formula_estimator")
        notes.append(f"ML predicted {role} {name}: Eg={eg_v:.3f} eV")
        if use_llm_flag:
            llm = llm_predict_eg_chi(name, role)
            sources[f"llm_{role}"] = llm
            sources["llm_used"] = True
        return eg_v, chi_v, src, src

    etl_eg, etl_chi, etl_eg_src, etl_chi_src = resolve_contact(etl, "etl")
    htl_eg, htl_chi, htl_eg_src, htl_chi_src = resolve_contact(htl, "htl")
    sources["etl_Eg"] = etl_eg_src
    sources["etl_chi"] = etl_chi_src
    sources["htl_Eg"] = htl_eg_src
    sources["htl_chi"] = htl_chi_src
    sources["etl"] = etl_eg_src if etl_eg_src == etl_chi_src else f"{etl_eg_src}/{etl_chi_src}"
    sources["htl"] = htl_eg_src if htl_eg_src == htl_chi_src else f"{htl_eg_src}/{htl_chi_src}"

    field_labels = {
        "absorber_Eg": _src_kind(sources.get("absorber_Eg")),
        "absorber_chi": _src_kind(sources.get("absorber_chi")),
        "etl_Eg": _src_kind(etl_eg_src),
        "etl_chi": _src_kind(etl_chi_src),
        "htl_Eg": _src_kind(htl_eg_src),
        "htl_chi": _src_kind(htl_chi_src),
    }

    result: dict = {
        "material_absorber": absorber,
        "material_etl": etl,
        "material_htl": htl,
        "absorber_band_gap_eV": abs_eg,
        "etl_band_gap_eV": etl_eg,
        "htl_band_gap_eV": htl_eg,
        "absorber_chi_eV": abs_chi,
        "etl_chi_eV": etl_chi,
        "htl_chi_eV": htl_chi,
        "sources": sources,
        "field_labels": field_labels,
        "notes": notes,
    }
    if estimate_meta:
        result["estimate_meta"] = estimate_meta
        if estimate_meta.get("confidence"):
            result["confidence"] = estimate_meta["confidence"]
        if estimate_meta.get("caution"):
            result["caution"] = True
        if estimate_meta.get("family_id"):
            result["perovskite_family"] = estimate_meta["family_id"]
    elif perovskite_check.get("family_id"):
        result["perovskite_family"] = perovskite_check["family_id"]

    if abs_chi is not None and etl_chi is not None and htl_chi is not None:
        a = Layer(absorber, abs_eg, float(abs_chi))
        e = Layer(etl, float(etl_eg), float(etl_chi))
        h = Layer(htl, float(htl_eg), float(htl_chi))
        row = stack_row(a, e, h, source_doi="pipeline", source_paper="computed")
        any_pred = any(field_labels[k] == "predicted" for k in field_labels)
        type_label = "predicted" if any_pred else "lookup"
        result.update(
            {
                "cbo_eV": row["cbo_eV"],
                "vbo_eV": row["vbo_eV"],
                "absorber_etl_type": row["absorber_etl_type"],
                "absorber_htl_type": row["absorber_htl_type"],
                "method": "compute_from_Eg_chi",
            }
        )
        field_labels["absorber_etl_type"] = type_label
        field_labels["absorber_htl_type"] = type_label
        opto = optoelectronic_suitability(row["absorber_etl_type"], row["absorber_htl_type"])
        opto["label"] = type_label
        if abs_bn in INDIRECT_GAP_MATERIALS:
            opto["gap_type"] = "indirect"
            if opto.get("verdict") == "YES":
                opto["reason"] = (
                    opto.get("reason", "")
                    + f" Caveat: {abs_bn} is an indirect-gap absorber — optical absorption may be weaker than a direct-gap YES."
                )
            else:
                opto["reason"] = (
                    opto.get("reason", "")
                    + f" Note: {abs_bn} has an indirect gap."
                )
        result["optoelectronic"] = opto
        field_labels["optoelectronic"] = type_label
        result["field_labels"] = field_labels
        _apply_degenerate_htl_caveat(result, htl)
        return result

    result["method"] = "ml_type_from_names_and_Eg"
    etl_type = htl_type = None
    if etl_eg is not None:
        etl_pred = ml_type(absorber, etl, abs_eg, float(etl_eg), "etl")
        etl_type = etl_pred["type"]
        result["predicted_absorber_etl_type"] = etl_type
        field_labels["absorber_etl_type"] = "predicted"
        if "proba" in etl_pred and not _is_degenerate_htl(htl):
            result["predicted_absorber_etl_proba"] = etl_pred["proba"]
    if htl_eg is not None:
        htl_pred = ml_type(absorber, htl, abs_eg, float(htl_eg), "htl")
        htl_type = htl_pred["type"]
        result["predicted_absorber_htl_type"] = htl_type
        field_labels["absorber_htl_type"] = "predicted"
        if "proba" in htl_pred and not _is_degenerate_htl(htl):
            result["predicted_absorber_htl_proba"] = htl_pred["proba"]
    opto = optoelectronic_suitability(etl_type, htl_type)
    opto["label"] = "predicted"
    if abs_bn in INDIRECT_GAP_MATERIALS:
        opto["gap_type"] = "indirect"
        if opto.get("verdict") == "YES":
            opto["reason"] = (
                opto.get("reason", "")
                + f" Caveat: {abs_bn} is an indirect-gap absorber — optical absorption may be weaker than a direct-gap YES."
            )
    result["optoelectronic"] = opto
    field_labels["optoelectronic"] = "predicted"
    result["field_labels"] = field_labels
    _apply_degenerate_htl_caveat(result, htl)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Train / predict stack Type (lookup + ML)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--absorber", type=str)
    ap.add_argument("--etl", type=str)
    ap.add_argument("--htl", type=str)
    ap.add_argument("--eg", type=float, default=None)
    ap.add_argument("--chi", type=float, default=None)
    ap.add_argument("--llm", action="store_true", help="Optional formula-only LLM (off by default)")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--predict-eg-only", action="store_true")
    ap.add_argument("--list-materials", action="store_true")
    args = ap.parse_args()

    if args.train:
        layers = load_layer_lookup()
        from formula_estimator import train_estimators

        formula_stats = train_estimators()
        eg_stats = train_eg_model()
        type_stats = train_type_models()
        meta = {"layers": len(layers), "formula": formula_stats, "eg": eg_stats, "type": type_stats}
        META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        return

    if args.list_materials:
        print(json.dumps(sorted(load_layer_lookup().keys()), indent=2))
        return

    if args.predict_eg_only:
        if not args.absorber:
            ap.error("--absorber required")
        from formula_estimator import estimate_eg_chi

        print(json.dumps({"absorber": args.absorber, **estimate_eg_chi(args.absorber)}, indent=2))
        return

    if not (args.absorber and args.etl and args.htl):
        ap.error("Need --absorber --etl --htl")

    out = predict_stack(
        args.absorber, args.etl, args.htl, eg=args.eg, chi=args.chi, use_llm=bool(args.llm)
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
