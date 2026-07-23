"""Fill χ for absorbers + build ETL/HTL material libraries.

Priority for χ:
  1) Exact SCAPS / literature χ from data/raw
  2) RandomForest estimate from formula features + Eg
  3) Role / family prior

Writes:
  data/perovskite_absorber_library.csv   (+ chi_eV, chi_source)
  data/etl_material_library.csv
  data/htl_material_library.csv
  data/layer_properties.csv
  data/models/chi_regressor.joblib
  data/models/layer_lookup.json

Usage:
  python scripts/enrich_chi_dataset.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from formula_parse import parse_formula_counts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = DATA / "models"
MODELS.mkdir(exist_ok=True)

ABS_PATH = DATA / "perovskite_absorber_library.csv"
ETL_PATH = DATA / "etl_material_library.csv"
HTL_PATH = DATA / "htl_material_library.csv"
STACK_PATH = DATA / "perovskite_stack_dataset.csv"
LAYER_OUT = DATA / "layer_properties.csv"
CHI_MODEL = MODELS / "chi_regressor.joblib"
LAYER_CACHE = MODELS / "layer_lookup.json"

SCAPS_FILES = (
    "paper4_scaps_materials.csv",
    "paper_cs_pb_scaps_materials.csv",
    "paper_cs3sb2br9_scaps_materials.csv",
    "paper_besip2_scaps_materials.csv",
    "paper_k2gei6_dft_absorber.csv",
)

PAPER_LABEL = {
    "paper4_scaps_materials.csv": "K2TiI6 SCAPS device tables",
    "paper_cs_pb_scaps_materials.csv": "CsPbZnIBr2 SCAPS device tables",
    "paper_cs3sb2br9_scaps_materials.csv": "Cs3Sb2Br9 SCAPS device tables",
    "paper_besip2_scaps_materials.csv": "BeSiP2 SCAPS device tables",
    "paper_k2gei6_dft_absorber.csv": "K2GeI6 DFT absorber",
}

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def base_name(name: str) -> str:
    """Strip phase / annotation suffixes: 'Cs2AgBiBr6 (cubic)' → 'Cs2AgBiBr6'."""
    return re.sub(r"\s*\(.*\)\s*$", "", (name or "").strip())


def formula_features(formula: str) -> dict[str, float]:
    return parse_formula_counts(formula)


def feature_matrix(
    formulas: list[str], keys: list[str] | None = None
) -> tuple[list[str], np.ndarray]:
    dicts = [formula_features(f) for f in formulas]
    keys = keys or sorted({k for d in dicts for k in d})
    X = np.zeros((len(dicts), len(keys)), dtype=float)
    for i, d in enumerate(dicts):
        for j, k in enumerate(keys):
            X[i, j] = d.get(k, 0.0)
    return keys, X


def load_literature_layers() -> dict[str, dict]:
    """Known Eg/χ from SCAPS-style raw tables (keyed by material name)."""
    layers: dict[str, dict] = {}
    for fname in SCAPS_FILES:
        path = RAW / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row.get("material") or not row.get("Eg_eV"):
                    continue
                name = row["material"].strip()
                role = (row.get("layer_role") or "unknown").strip().lower()
                if role not in ("absorber", "etl", "htl"):
                    # k2gei6 file has no role
                    role = "absorber" if "absorber" in fname or name == "K2GeI6" else role
                entry = {
                    "material": name,
                    "Eg_eV": float(row["Eg_eV"]),
                    "chi_eV": float(row["chi_eV"]) if row.get("chi_eV") else None,
                    "role": role or "unknown",
                    "chi_source": "literature_SCAPS" if row.get("chi_eV") else None,
                    "source_doi": (row.get("source_doi") or "").strip(),
                    "source_paper": PAPER_LABEL.get(fname, fname),
                    "source_table": (row.get("source_table") or "").strip(),
                    "source_file": fname,
                }
                prev = layers.get(name)
                if prev is None:
                    layers[name] = entry
                elif name.startswith("TiO2") and entry["Eg_eV"] > prev["Eg_eV"]:
                    layers[name] = entry
                elif prev.get("chi_eV") is None and entry.get("chi_eV") is not None:
                    layers[name] = entry

    if "K2GeI6" in layers and layers["K2GeI6"]["chi_eV"] is None:
        layers["K2GeI6"]["chi_eV"] = 4.05
        layers["K2GeI6"]["chi_source"] = "family_prior_vacancy_ordered_iodide"
        layers["K2GeI6"]["role"] = "absorber"

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
                name = row["material"].strip()
                eg = float(row[eg_col])
                vbm = float(row[vbm_col])
                chi = -(vbm + eg)
                entry = {
                    "material": name,
                    "Eg_eV": eg,
                    "chi_eV": chi,
                    "role": "monolayer",
                    "chi_source": "literature_HSE06_monolayer",
                    "source_file": fname,
                }
                prev = layers.get(name)
                if prev is None or prev.get("chi_source", "").startswith("ml"):
                    layers[name] = entry

    return layers


def family_prior_chi(row: dict) -> float:
    """Composition / family priors for halide & oxide perovskite absorbers."""
    family = (row.get("perovskite_family") or "").lower()
    mclass = (row.get("material_class") or "").lower()
    x = (row.get("x_site") or "").strip()
    name = base_name(row.get("material_absorber") or "")

    halide_shift = {"I": -0.05, "Br": 0.0, "Cl": 0.08, "F": 0.12}.get(x, 0.0)

    if "double" in family or "double" in mclass:
        base = 3.95
    elif "vacancy" in family or ("I6" in name):
        base = 4.00
    elif "oxide" in mclass or "oxide" in family:
        base = 4.10
    elif "ABX3" in family or "halide" in mclass:
        base = 3.90
    else:
        base = 3.95

    try:
        eg = float(row.get("absorber_band_gap_eV") or 0)
        if eg > 2.5 and "oxide" in mclass:
            base += 0.1
    except ValueError:
        pass

    return round(base + halide_shift, 3)


def contact_prior_chi(name: str, role: str, eg: float) -> float:
    """Fallback χ priors for ETL/HTL when literature χ is missing."""
    n = name.lower()
    if role == "etl":
        if "tio2" in n:
            return 4.0
        if "sno2" in n:
            return 3.9
        if "zno" in n:
            return 4.0
        if "cds" in n:
            return 4.2
        if "pcb" in n or "pc60" in n:
            return 4.1
        return 4.0 if eg >= 2.5 else 3.9
    # htl
    if "moo3" in n:
        return 2.3
    if "nio" in n:
        return 1.8
    if "ptaa" in n:
        return 2.3
    if "cui" in n:
        return 2.1
    if "cuscn" in n:
        return 1.7
    return 2.5 if eg >= 2.5 else 3.2


def train_chi_model(lit: dict[str, dict]) -> tuple[object, list[str], float]:
    names, y, roles = [], [], []
    for name, e in lit.items():
        if e.get("chi_eV") is None:
            continue
        names.append(name)
        y.append(float(e["chi_eV"]))
        roles.append(
            1.0 if e.get("role") == "etl" else (0.0 if e.get("role") == "htl" else 0.5)
        )
    keys, Xcomp = feature_matrix(names)
    eg = np.array([lit[n]["Eg_eV"] for n in names], dtype=float).reshape(-1, 1)
    role = np.array(roles, dtype=float).reshape(-1, 1)
    X = np.hstack([Xcomp, eg, role])
    y_arr = np.array(y, dtype=float)

    if len(y_arr) < 8:

        class MeanModel:
            def predict(self, X):
                return np.full(len(X), float(np.mean(y_arr)))

        return MeanModel(), keys, 0.0

    Xtr, Xte, ytr, yte = train_test_split(X, y_arr, test_size=0.25, random_state=42)
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1,
    )
    model.fit(Xtr, ytr)
    mae = float(mean_absolute_error(yte, model.predict(Xte))) if len(yte) else 0.0
    model.fit(X, y_arr)
    joblib.dump({"model": model, "element_keys": keys, "holdout_mae": mae}, CHI_MODEL)
    return model, keys, mae


def predict_chi_ml(model, keys: list[str], name: str, eg: float, role_code: float) -> float:
    _, Xcomp = feature_matrix([name], keys)
    X = np.hstack([Xcomp, [[eg]], [[role_code]]])
    return float(model.predict(X)[0])


def enrich_absorber_library(lit: dict[str, dict], model, keys: list[str]) -> dict:
    rows = list(csv.DictReader(ABS_PATH.open(encoding="utf-8")))
    fieldnames = list(rows[0].keys())
    for col in ("chi_eV", "chi_source"):
        if col not in fieldnames:
            fieldnames.append(col)

    stats = {"literature": 0, "ml_estimate": 0}
    for row in rows:
        name = base_name(row["material_absorber"])
        eg = float(row["absorber_band_gap_eV"])
        if name in lit and lit[name].get("chi_eV") is not None:
            row["chi_eV"] = f"{lit[name]['chi_eV']:.4f}"
            row["chi_source"] = lit[name].get("chi_source") or "literature_SCAPS"
            stats["literature"] += 1
            continue
        prior = family_prior_chi(row)
        ml = predict_chi_ml(model, keys, name, eg, 0.5)
        chi = float(np.clip(0.35 * ml + 0.65 * prior, 2.5, 4.8))
        row["chi_eV"] = f"{chi:.4f}"
        row["chi_source"] = "ml_plus_family_prior"
        stats["ml_estimate"] += 1

    with ABS_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return stats


def _stack_contacts() -> dict[str, set[str]]:
    """Unique ETL/HTL names appearing in the stack dataset."""
    out = {"etl": set(), "htl": set()}
    if not STACK_PATH.exists():
        return out
    with STACK_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("material_etl"):
                out["etl"].add(row["material_etl"].strip())
            if row.get("material_htl"):
                out["htl"].add(row["material_htl"].strip())
    return out


def build_contact_library(
    role: str,
    lit: dict[str, dict],
    model,
    keys: list[str],
    out_path: Path,
) -> dict:
    """Build etl_material_library.csv or htl_material_library.csv."""
    stack_names = _stack_contacts()[role]
    names = {
        name
        for name, e in lit.items()
        if e.get("role") == role or name in stack_names
    }
    names |= stack_names

    fieldnames = [
        "material",
        "Eg_eV",
        "chi_eV",
        "chi_source",
        "layer_role",
        "material_class",
        "record_type",
        "source_doi",
        "source_paper",
        "source_table",
    ]
    rows = []
    stats = {"literature": 0, "estimated": 0, "missing_eg": 0}

    for name in sorted(names, key=str.lower):
        e = lit.get(name)
        if e and e.get("Eg_eV") is not None:
            eg = float(e["Eg_eV"])
            doi = e.get("source_doi") or ""
            paper = e.get("source_paper") or ""
            table = e.get("source_table") or ""
            if e.get("chi_eV") is not None:
                chi = float(e["chi_eV"])
                src = e.get("chi_source") or "literature_SCAPS"
                stats["literature"] += 1
            else:
                role_code = 1.0 if role == "etl" else 0.0
                prior = contact_prior_chi(name, role, eg)
                ml = predict_chi_ml(model, keys, name, eg, role_code)
                chi = float(np.clip(0.5 * ml + 0.5 * prior, 1.5, 4.8))
                src = "ml_plus_contact_prior"
                stats["estimated"] += 1
        else:
            # Name only in stacks — should not happen if lit is complete
            stats["missing_eg"] += 1
            continue

        rows.append(
            {
                "material": name,
                "Eg_eV": f"{eg:.4f}",
                "chi_eV": f"{chi:.4f}",
                "chi_source": src,
                "layer_role": role,
                "material_class": "contact_layer",
                "record_type": f"{role}_library",
                "source_doi": doi,
                "source_paper": paper,
                "source_table": table,
            }
        )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    stats["n"] = len(rows)
    stats["path"] = str(out_path)
    return stats


def write_layer_properties(lit: dict[str, dict]) -> dict[str, dict]:
    """Unified lookup: literature + enriched absorber/ETL/HTL libraries."""
    layers: dict[str, dict] = {}

    def _put(name: str, eg: float, chi: float, role: str, chi_source: str) -> None:
        entry = {
            "Eg_eV": eg,
            "chi_eV": chi,
            "role": role,
            "chi_source": chi_source,
        }
        layers[name] = entry
        bn = base_name(name)
        if bn != name and bn not in layers:
            layers[bn] = dict(entry)

    for name, e in lit.items():
        if e.get("chi_eV") is None or e.get("Eg_eV") is None:
            continue
        _put(
            name,
            float(e["Eg_eV"]),
            float(e["chi_eV"]),
            e.get("role", "unknown"),
            e.get("chi_source") or "literature_SCAPS",
        )

    with ABS_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full = row["material_absorber"].strip()
            name = base_name(full)
            if not row.get("chi_eV"):
                continue
            eg = float(row["absorber_band_gap_eV"])
            chi = float(row["chi_eV"])
            src = row.get("chi_source") or "estimated"
            # Prefer literature already stored
            if name not in layers or str(layers[name].get("chi_source", "")).startswith(
                "ml"
            ):
                _put(name, eg, chi, "absorber", src)
            _put(full, eg, chi, "absorber", src)

    for path, role in ((ETL_PATH, "etl"), (HTL_PATH, "htl")):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row["material"].strip()
                _put(
                    name,
                    float(row["Eg_eV"]),
                    float(row["chi_eV"]),
                    role,
                    row.get("chi_source") or "literature_SCAPS",
                )

    with LAYER_OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["material", "Eg_eV", "chi_eV", "role", "chi_source"]
        )
        w.writeheader()
        for name, e in sorted(layers.items(), key=lambda x: x[0].lower()):
            w.writerow(
                {
                    "material": name,
                    "Eg_eV": e["Eg_eV"],
                    "chi_eV": e["chi_eV"],
                    "role": e.get("role", ""),
                    "chi_source": e.get("chi_source", ""),
                }
            )

    cache = {k: {"Eg_eV": v["Eg_eV"], "chi_eV": v["chi_eV"]} for k, v in layers.items()}
    LAYER_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return layers


def main() -> None:
    lit = load_literature_layers()
    model, keys, mae = train_chi_model(lit)
    abs_stats = enrich_absorber_library(lit, model, keys)
    etl_stats = build_contact_library("etl", lit, model, keys, ETL_PATH)
    htl_stats = build_contact_library("htl", lit, model, keys, HTL_PATH)
    layers = write_layer_properties(lit)
    print(
        json.dumps(
            {
                "literature_layers_with_chi": sum(
                    1 for v in lit.values() if v.get("chi_eV") is not None
                ),
                "chi_model_holdout_mae_eV": round(mae, 4),
                "absorber_library": abs_stats,
                "etl_library": etl_stats,
                "htl_library": htl_stats,
                "layer_lookup_entries": len(layers),
                "layer_properties": str(LAYER_OUT),
                "note": (
                    "Estimated χ is for screening only — not measured literature. "
                    "Marked via chi_source (e.g. ml_plus_family_prior)."
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
