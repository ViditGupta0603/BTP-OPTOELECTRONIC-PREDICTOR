"""Shared formula parsing for perovskite / contact materials.

Handles organic cation abbreviations (FA/MA/…), mixed A-site fractions,
and parenthesized groups like HC(NH2)2.
"""
from __future__ import annotations

import re

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_GROUP = re.compile(r"\(([A-Za-z0-9.]+)\)(\d*\.?\d*)")
_DASHES = str.maketrans("−–—", "---")

# Stoichiometric atom counts for common organic A-site cations
ORGANIC_CATIONS: dict[str, dict[str, float]] = {
    "FA": {"C": 1.0, "H": 5.0, "N": 2.0},  # formamidinium CH5N2+
    "MA": {"C": 1.0, "H": 6.0, "N": 1.0},  # methylammonium CH3NH3+
    "GA": {"C": 1.0, "H": 6.0, "N": 3.0},  # guanidinium
    "EA": {"C": 2.0, "H": 8.0, "N": 1.0},  # ethylammonium
    "PEA": {"C": 8.0, "H": 12.0, "N": 1.0},  # phenethylammonium (approx)
    "BA": {"C": 4.0, "H": 12.0, "N": 1.0},  # butylammonium
}

# Named organics / polymers — not element-parseable; flag features only
NAMED_ORGANICS = {
    "spiro-ometad",
    "spiroometad",
    "spiro",
    "ptaa",
    "cupc",
    "c6pch2",
    "c60",
    "pcbm",
    "pc60bm",
    "meh-ppv",
    "npb",
}

# Canonical name aliases (user shorthand → library formula)
MATERIAL_ALIASES: dict[str, str] = {
    "fapbi3": "HC(NH2)2PbI3",
    "fapbbr3": "HC(NH2)2PbBr3",
    "fapbcl3": "HC(NH2)2PbCl3",
    "mapbi3": "CH3NH3PbI3",
    "mapbbr3": "CH3NH3PbBr3",
    "mapbcl3": "CH3NH3PbCl3",
    "fasni3": "HC(NH2)2SnI3",
    "masni3": "CH3NH3SnI3",
    "fagei3": "HC(NH2)2GeI3",
    "magei3": "CH3NH3GeI3",
    "spiro": "Spiro-OMeTAD",
    "spiroometad": "Spiro-OMeTAD",
    "spiro-ometad": "Spiro-OMeTAD",
}


def base_name(name: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", (name or "").strip())


def canonicalize_material_alias(name: str) -> str:
    """Map FAPbI3 / MAPbI3 / Spiro shorthand to canonical library names."""
    s = base_name(name).translate(_DASHES).strip()
    key = s.lower().replace(" ", "")
    if key in MATERIAL_ALIASES:
        return MATERIAL_ALIASES[key]
    # FA/MA prefix on halide perovskites: FA0.83Cs0.17PbI3 stays as-is for parsing
    return s


def _add_counts(dst: dict[str, float], src: dict[str, float], mult: float = 1.0) -> None:
    for el, n in src.items():
        dst[el] = dst.get(el, 0.0) + n * mult


def _parse_simple_elements(fragment: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for el, num in _ELEMENT.findall(fragment):
        counts[el] = counts.get(el, 0.0) + (float(num) if num else 1.0)
    return counts


def expand_parentheses(formula: str) -> str:
    """Expand (NH2)2 → N2H4 style by inlining multiplied groups repeatedly."""
    s = formula
    for _ in range(8):
        m = _GROUP.search(s)
        if not m:
            break
        inner, mult_s = m.group(1), m.group(2)
        mult = float(mult_s) if mult_s else 1.0
        inner_counts = _parse_simple_elements(inner)
        # If inner still has letters only partially parsed, fall back to raw repeat
        if not inner_counts:
            replacement = inner * int(round(mult)) if mult == int(mult) else inner
        else:
            parts = []
            for el, n in inner_counts.items():
                total = n * mult
                if abs(total - round(total)) < 1e-9:
                    total_i = int(round(total))
                    parts.append(el if total_i == 1 else f"{el}{total_i}")
                else:
                    parts.append(f"{el}{total:g}")
            replacement = "".join(parts)
        s = s[: m.start()] + replacement + s[m.end() :]
    return s


def _expand_organic_tokens(formula: str) -> tuple[str, dict[str, float]]:
    """Replace FA/MA/… tokens (with optional fractions) by atom counts.

    Returns (residual_formula, organic_counts).
    """
    organic: dict[str, float] = {}
    s = formula
    # Longest cation names first
    cations = sorted(ORGANIC_CATIONS.keys(), key=len, reverse=True)
    for cat in cations:
        # Match FA0.83 or FA at token boundary (not part of another element)
        pat = re.compile(rf"(?<![A-Za-z]){cat}(\d*\.?\d*)(?![a-z])")
        while True:
            m = pat.search(s)
            if not m:
                break
            mult = float(m.group(1)) if m.group(1) else 1.0
            _add_counts(organic, ORGANIC_CATIONS[cat], mult)
            s = s[: m.start()] + s[m.end() :]
    return s, organic


def parse_formula_counts(formula: str) -> dict[str, float]:
    """Parse a chemical formula into element → count (supports FA/MA + parentheses)."""
    clean = base_name(formula).replace(" ", "").translate(_DASHES)
    if not clean:
        return {}

    # Named organics: return sentinel carbon/nitrogen so has_organic fires
    key = clean.lower().replace("-", "")
    if key in NAMED_ORGANICS or clean.lower() in NAMED_ORGANICS:
        return {"C": 1.0, "H": 1.0, "N": 1.0, "_named_organic": 1.0}

    residual, organic = _expand_organic_tokens(clean)
    residual = expand_parentheses(residual)
    # Strip leftover punctuation
    residual = re.sub(r"[^A-Za-z0-9.]", "", residual)
    counts = _parse_simple_elements(residual)
    _add_counts(counts, organic)
    counts.pop("_named_organic", None)
    return counts


def formula_feature_dict(formula: str) -> dict[str, float]:
    """Bag-of-elements + perovskite family flags used by Eg models."""
    clean = base_name(formula).replace(" ", "")
    feats = parse_formula_counts(formula)
    # Drop internal markers from numeric matrix
    feats = {k: v for k, v in feats.items() if not k.startswith("_")}
    total = sum(feats.values()) or 1.0
    halogens = sum(feats.get(x, 0) for x in ("F", "Cl", "Br", "I"))
    pb = feats.get("Pb", 0)
    b_site = pb + feats.get("Sn", 0) + feats.get("Ge", 0)
    feats["has_Pb"] = 1.0 if pb else 0.0
    feats["has_Sn"] = 1.0 if feats.get("Sn") else 0.0
    feats["has_Ge"] = 1.0 if feats.get("Ge") else 0.0
    feats["has_Cs"] = 1.0 if feats.get("Cs") else 0.0
    feats["has_organic"] = (
        1.0
        if (feats.get("C", 0) > 0 and feats.get("N", 0) > 0)
        or any(tok in clean.upper() for tok in ("FA", "MA", "GA", "EA"))
        else 0.0
    )
    feats["is_A2BX6"] = 1.0 if re.search(r".*2.*6$", clean.replace(".", "")) else 0.0
    feats["is_double_like"] = (
        1.0 if feats.get("Ag", 0) or feats.get("Bi", 0) or feats.get("In", 0) else 0.0
    )
    feats["is_A3B2X9"] = (
        1.0 if re.search(r".*3.*2.*9", clean.replace(".", "")) and halogens >= 6 else 0.0
    )
    if b_site > 0 and halogens > 0:
        feats["halogen_per_B"] = halogens / b_site
        feats["is_ABX3"] = (
            1.0
            if abs(halogens - 3 * b_site) < 0.05 and not feats.get("is_double_like")
            else 0.0
        )
    if pb > 0 and halogens > 0:
        feats["halogen_per_Pb"] = halogens / pb
        if "is_ABX3" not in feats:
            feats["is_ABX3"] = (
                1.0
                if abs(halogens - 3 * pb) < 0.01 and not feats.get("is_double_like")
                else 0.0
            )
    for x in ("Cl", "Br", "I"):
        feats[f"frac_{x}"] = feats.get(x, 0) / total
    feats["frac_organic_CN"] = (feats.get("C", 0) + feats.get("N", 0)) / total

    # Align stoichiometry flags with taxonomy when available
    try:
        from perovskite_rules import classify_family, parse_sites

        fam = classify_family(formula)
        sites = parse_sites(formula)
        if fam.family_id.startswith("abx3"):
            feats["is_ABX3"] = 1.0
        if fam.family_id == "vacancy_ordered_a2bx6":
            feats["is_A2BX6"] = 1.0
        if fam.family_id == "halide_double_a2bbx6":
            feats["is_double_like"] = 1.0
            feats["is_A2BX6"] = 1.0
        if fam.family_id == "a3b2x9_0d":
            feats["is_A3B2X9"] = 1.0
        feats["mixed_a_site"] = 1.0 if sites.is_mixed_a else 0.0
        feats["mixed_halide"] = 1.0 if sites.is_mixed_x else 0.0
    except Exception:
        pass
    return feats
