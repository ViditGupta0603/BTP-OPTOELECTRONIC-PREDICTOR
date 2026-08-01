"""Deterministic formula -> Eg/chi estimators (ML + perovskite family rules).

No web search. Same formula -> same Eg/chi for perovskite families (role-independent).
Contact ETL/HTL unknowns still use role-specific chi priors.

Train:
  python scripts/formula_estimator.py --train

Predict:
  python scripts/formula_estimator.py --material Cs2AgBiBr6 --role absorber
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from formula_parse import (  # noqa: E402
    base_name,
    formula_feature_dict,
    parse_formula_counts,
)
from perovskite_rules import (  # noqa: E402
    classify_family,
    confidence_for_estimate,
    family_feature_flags,
    family_prior_eg_chi,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = DATA / "models"
MODELS.mkdir(exist_ok=True)

EG_CHI_MODEL = MODELS / "formula_eg_chi_estimator.joblib"
ABS_PATH = DATA / "perovskite_absorber_library.csv"
ETL_PATH = DATA / "etl_material_library.csv"
HTL_PATH = DATA / "htl_material_library.csv"

HALIDE_EG_SHIFT = {"F": 0.45, "Cl": 0.30, "Br": 0.10, "I": 0.0, "O": 0.55}
ROLE_CHI_PRIOR = {"absorber": 3.95, "etl": 4.00, "htl": 2.40}
ROLE_EG_PRIOR = {"absorber": 1.60, "etl": 3.20, "htl": 2.80}

# Named organic / polymer HTLs — ROLE_EG_PRIOR=2.80 is for generic oxides; do not use for these.
ORGANIC_HTL_PRIORS: dict[str, tuple[float, float]] = {
    "P3HT": (1.90, 3.10),
    "MEH-PPV": (2.10, 2.80),
    "PTAA": (2.96, 2.30),
    "PEDOT:PSS": (1.60, 3.30),
    "PEDOT": (1.60, 3.30),
    "Spiro-OMeTAD": (3.00, 2.05),
    "CuPc": (1.70, 3.50),
    "C6PcH2": (1.60, 3.70),
    "nPB": (2.40, 3.00),
    "NPB": (2.40, 3.00),
}

_ABSORBER_FAMILY_PREFIXES = (
    "abx3",
    "halide_double",
    "vacancy_ordered",
    "a3b2x9",
    "oxide_perovskite",
)


def parse_formula(formula: str) -> dict[str, float]:
    return parse_formula_counts(formula)


def composition_vector(formula: str, role: str) -> dict[str, float]:
    """Deterministic feature dict from formula + role (no randomness)."""
    counts = parse_formula(formula)
    total = sum(counts.values()) or 1.0
    feats: dict[str, float] = {f"n_{el}": c for el, c in counts.items()}
    feats["n_atoms"] = total
    feats["n_elements"] = float(len(counts))
    for el, c in counts.items():
        feats[f"frac_{el}"] = c / total

    for h in ("F", "Cl", "Br", "I", "O", "S", "Se", "Te"):
        feats[f"has_{h}"] = 1.0 if counts.get(h, 0) > 0 else 0.0
        feats[f"frac_{h}"] = counts.get(h, 0.0) / total

    halide_shift = 0.0
    for h, w in HALIDE_EG_SHIFT.items():
        halide_shift += feats[f"frac_{h}"] * w
    feats["halide_eg_shift"] = halide_shift

    family = formula_feature_dict(formula)
    feats["is_A2BX6"] = family.get("is_A2BX6", 0.0)
    feats["is_double_like"] = family.get("is_double_like", 0.0)
    feats["is_ABX3"] = family.get("is_ABX3", 0.0)
    feats["is_A3B2X9"] = family.get("is_A3B2X9", 0.0)
    feats["has_Pb"] = family.get("has_Pb", 0.0)
    feats["has_Sn"] = family.get("has_Sn", 0.0)
    feats["has_Ge"] = family.get("has_Ge", 0.0)
    feats["has_Ti"] = 1.0 if counts.get("Ti", 0) else 0.0
    feats["has_organic"] = family.get("has_organic", 0.0)
    feats["frac_organic_CN"] = family.get("frac_organic_CN", 0.0)
    feats["role_absorber"] = 1.0 if role == "absorber" else 0.0
    feats["role_etl"] = 1.0 if role == "etl" else 0.0
    feats["role_htl"] = 1.0 if role == "htl" else 0.0
    feats["role_chi_prior"] = ROLE_CHI_PRIOR.get(role, 3.5)
    feats["role_eg_prior"] = ROLE_EG_PRIOR.get(role, 2.0)

    try:
        flags = family_feature_flags(formula)
        feats.update(flags)
    except Exception:
        pass
    return feats


def _matrix(rows: list[dict[str, float]], keys: list[str] | None = None):
    keys = keys or sorted({k for r in rows for k in r})
    X = np.zeros((len(rows), len(keys)), dtype=float)
    for i, r in enumerate(rows):
        for j, k in enumerate(keys):
            X[i, j] = r.get(k, 0.0)
    return keys, X


def _load_training_rows() -> list[dict]:
    rows: list[dict] = []

    def add(name: str, eg: float, chi: float | None, role: str, eg_src: str, chi_src: str):
        rows.append(
            {
                "material": name,
                "Eg_eV": eg,
                "chi_eV": chi,
                "role": role,
                "eg_source": eg_src,
                "chi_source": chi_src,
            }
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
            for r in csv.DictReader(f):
                if not r.get("material") or not r.get("Eg_eV"):
                    continue
                role = (r.get("layer_role") or "absorber").strip().lower()
                if role not in ("absorber", "etl", "htl"):
                    role = "absorber"
                chi = float(r["chi_eV"]) if r.get("chi_eV") else None
                add(
                    r["material"].strip(),
                    float(r["Eg_eV"]),
                    chi,
                    role,
                    "literature_SCAPS",
                    "literature_SCAPS" if chi is not None else "",
                )

    if ABS_PATH.exists():
        with ABS_PATH.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                name = base_name(r.get("material_absorber") or "")
                if not name or not r.get("absorber_band_gap_eV"):
                    continue
                chi = float(r["chi_eV"]) if r.get("chi_eV") else None
                add(
                    name,
                    float(r["absorber_band_gap_eV"]),
                    chi,
                    "absorber",
                    "absorber_library",
                    r.get("chi_source") or "",
                )

    for path, role in ((ETL_PATH, "etl"), (HTL_PATH, "htl")):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("material") or not r.get("Eg_eV"):
                    continue
                chi = float(r["chi_eV"]) if r.get("chi_eV") else None
                add(
                    r["material"].strip(),
                    float(r["Eg_eV"]),
                    chi,
                    role,
                    "contact_library",
                    r.get("chi_source") or "",
                )

    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (base_name(r["material"]), r["role"])
        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue
        if prev.get("chi_eV") is None and r.get("chi_eV") is not None:
            best[key] = r
        elif r.get("eg_source") == "literature_SCAPS":
            best[key] = r
    return list(best.values())


def heuristic_eg_chi(formula: str, role: str) -> tuple[float, float]:
    """Pure fixed-rule fallback if models missing — family prior first."""
    bn = base_name(formula)
    if role == "htl" and bn in ORGANIC_HTL_PRIORS:
        return ORGANIC_HTL_PRIORS[bn]

    prior = family_prior_eg_chi(formula, role)
    if prior.get("eligible") or prior["family_id"].startswith("abx3") or prior[
        "family_id"
    ] in ("halide_double_a2bbx6", "vacancy_ordered_a2bx6", "a3b2x9_0d"):
        return float(prior["Eg_eV"]), float(prior["chi_eV"])

    feats = composition_vector(formula, role)
    eg = ROLE_EG_PRIOR.get(role, 2.0) + feats["halide_eg_shift"]
    if feats["is_A2BX6"] and role == "absorber":
        eg = 1.4 + feats["halide_eg_shift"]
    if feats.get("has_O", 0) and role == "absorber":
        eg = max(eg, 2.2)
    chi = ROLE_CHI_PRIOR.get(role, 3.5)
    if role == "htl":
        chi = 2.4 - 0.3 * feats.get("has_I", 0)
    eg = float(np.clip(eg, 0.2, 6.0))
    chi = float(np.clip(chi, 1.4, 4.8))
    return eg, chi


def train_estimators() -> dict:
    rows = _load_training_rows()
    feat_rows = [composition_vector(r["material"], r["role"]) for r in rows]
    keys, X = _matrix(feat_rows)
    y_eg = np.array([r["Eg_eV"] for r in rows], dtype=float)

    eg_model = RandomForestRegressor(
        n_estimators=400, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=1
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y_eg, test_size=0.2, random_state=42)
    eg_model.fit(Xtr, ytr)
    eg_mae = float(mean_absolute_error(yte, eg_model.predict(Xte)))
    eg_model.fit(X, y_eg)

    chi_rows = [r for r in rows if r.get("chi_eV") is not None]
    feat_chi = [composition_vector(r["material"], r["role"]) for r in chi_rows]
    _keys_chi, Xc = _matrix(feat_chi, keys)
    eg_chi = np.array([r["Eg_eV"] for r in chi_rows], dtype=float).reshape(-1, 1)
    Xc = np.hstack([Xc, eg_chi])
    y_chi = np.array([float(r["chi_eV"]) for r in chi_rows], dtype=float)

    chi_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    if len(y_chi) >= 8:
        Xtr, Xte, ytr, yte = train_test_split(Xc, y_chi, test_size=0.25, random_state=42)
        chi_model.fit(Xtr, ytr)
        chi_mae = float(mean_absolute_error(yte, chi_model.predict(Xte)))
        chi_model.fit(Xc, y_chi)
    else:
        chi_mae = 0.0
        chi_model.fit(Xc, y_chi)

    payload = {
        "eg_model": eg_model,
        "chi_model": chi_model,
        "feature_keys": keys,
        "eg_holdout_mae": eg_mae,
        "chi_holdout_mae": chi_mae,
        "n_eg": len(y_eg),
        "n_chi": len(y_chi),
    }
    joblib.dump(payload, EG_CHI_MODEL)
    return {
        "eg_holdout_mae_eV": round(eg_mae, 4),
        "chi_holdout_mae_eV": round(chi_mae, 4),
        "n_eg": len(y_eg),
        "n_chi": len(y_chi),
        "path": str(EG_CHI_MODEL),
    }


def _is_perovskite_family(family_id: str) -> bool:
    return any(family_id.startswith(p) or family_id == p for p in _ABSORBER_FAMILY_PREFIXES)


def estimate_eg_chi(formula: str, role: str = "absorber") -> dict:
    """Deterministic Eg/chi: family/Vegard prior blended with ML; clamps by family.

    For perovskite absorber families, Eg/chi are role-independent (same material ->
    same numbers). Contact roles keep role-specific priors when not perovskite.
    """
    role = role if role in ("absorber", "etl", "htl") else "absorber"
    bn = base_name(formula)

    # Named organic HTL priors beat generic ROLE_EG_PRIOR=2.80 (P3HT optical ~1.9 eV)
    if role == "htl" and bn in ORGANIC_HTL_PRIORS:
        eg, chi = ORGANIC_HTL_PRIORS[bn]
        return {
            "Eg_eV": round(eg, 6),
            "chi_eV": round(chi, 6),
            "source": "organic_htl_literature_prior",
            "predicted": True,
            "family_id": "contact_htl",
            "confidence": "high",
            "caution": False,
            "prior_Eg_eV": round(eg, 6),
            "ml_Eg_eV": round(eg, 6),
            "blend_weight_prior": 1.0,
            "prior_method": "organic_htl_named",
        }

    fam = classify_family(formula)
    prior_role = "absorber" if _is_perovskite_family(fam.family_id) else role
    prior = family_prior_eg_chi(formula, prior_role)

    if not EG_CHI_MODEL.exists():
        eg, chi = heuristic_eg_chi(formula, prior_role)
        conf = confidence_for_estimate(
            family_id=fam.family_id,
            vegard=bool(prior.get("vegard")),
            ml_delta=0.0,
        )
        return {
            "Eg_eV": round(eg, 6),
            "chi_eV": round(chi, 6),
            "source": "family_rules" if prior.get("vegard") else "formula_heuristic",
            "predicted": True,
            "family_id": fam.family_id,
            "confidence": conf,
            "caution": conf == "low",
        }

    payload = joblib.load(EG_CHI_MODEL)
    keys: list[str] = payload["feature_keys"]
    feats = composition_vector(formula, prior_role)
    _, X = _matrix([feats], keys)
    ml_eg = float(payload["eg_model"].predict(X)[0])
    Xc = np.hstack([X, [[ml_eg]]])
    ml_chi = float(payload["chi_model"].predict(Xc)[0])

    w = float(prior.get("prior_blend_weight", 0.55))
    ml_delta = abs(ml_eg - float(prior["Eg_eV"]))
    if prior.get("vegard") and ml_delta > 0.8:
        w = min(0.95, w + 0.15)
    if fam.family_id == "vacancy_ordered_a2bx6" and ml_delta > 1.0:
        w = min(0.95, max(w, 0.88))

    eg = w * float(prior["Eg_eV"]) + (1.0 - w) * ml_eg
    chi = w * float(prior["chi_eV"]) + (1.0 - w) * ml_chi

    if not _is_perovskite_family(fam.family_id):
        if role == "etl":
            chi = 0.5 * chi + 0.5 * ROLE_CHI_PRIOR["etl"]
        elif role == "htl":
            chi = 0.5 * chi + 0.5 * ROLE_CHI_PRIOR["htl"]

    eg_min = float(prior.get("eg_min", 0.15))
    eg_max = float(prior.get("eg_max", 6.5))
    eg = float(np.clip(eg, eg_min, eg_max))
    chi = float(np.clip(chi, 1.3, 4.9))
    eg = round(eg, 6)
    chi = round(chi, 6)

    conf = confidence_for_estimate(
        family_id=fam.family_id,
        vegard=bool(prior.get("vegard")),
        ml_delta=ml_delta,
    )
    src = "ml_plus_family_prior"
    if prior.get("vegard"):
        src = "vegard_plus_ml" if w < 0.99 else "vegard_family_rules"

    return {
        "Eg_eV": eg,
        "chi_eV": chi,
        "source": src,
        "predicted": True,
        "family_id": fam.family_id,
        "confidence": conf,
        "caution": conf == "low",
        "prior_Eg_eV": round(float(prior["Eg_eV"]), 6),
        "ml_Eg_eV": round(ml_eg, 6),
        "blend_weight_prior": round(w, 3),
        "prior_method": prior.get("method"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic formula Eg/chi estimator")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--material", type=str)
    ap.add_argument("--role", default="absorber", choices=["absorber", "etl", "htl"])
    args = ap.parse_args()
    if args.train:
        print(json.dumps(train_estimators(), indent=2))
        return
    if not args.material:
        ap.error("Need --train or --material")
    print(
        json.dumps(
            {"material": args.material, "role": args.role, **estimate_eg_chi(args.material, args.role)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
