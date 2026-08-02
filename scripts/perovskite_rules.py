"""Perovskite taxonomy, eligibility, and Eg/χ priors for unknown formulas.

Formula + fixed rules only — no LLM / web recall. Used by:
  - check_absorber_perovskite / eligibility
  - formula_estimator (family prior + Vegard blend)
  - formula_parse family flags
  - llm_literature_assist DOMAIN_CONTEXT (mirrored rules)

Families covered:
  3D halide ABX₃, mixed A-site, mixed halide, halide double A₂B′B″X₆,
  vacancy-ordered A₂BX₆, 0D A₃B₂X₉, oxide / double oxide, 2D RP/DJ,
  contact-layer exceptions, non-perovskite blocks (GaAs, CdTe, BeSiP₂, …).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from formula_parse import (
    ORGANIC_CATIONS,
    base_name,
    canonicalize_material_alias,
    parse_formula_counts,
)

_MONOLAYER_PREFIX = re.compile(r"^[12][TH]-", re.I)
_RP_DJ_MARKERS = re.compile(
    r"(PEA|BA|OA|AVA|PDA|BDA|RP|DJ|Ruddlesden|Dion)", re.I
)

# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyRule:
    family_id: str
    eg_min: float
    eg_max: float
    eg_prior: float
    chi_prior: float
    absorber_eligible: bool
    description: str
    # Weight on family/Vegard prior vs ML when blending (higher = more rule trust)
    prior_blend_weight: float = 0.55
    halide_sensitive: bool = True
    notes: str = ""


FAMILIES: dict[str, FamilyRule] = {
    "abx3_halide_3d": FamilyRule(
        "abx3_halide_3d",
        1.0,
        3.2,
        1.60,
        3.90,
        True,
        "3D halide ABX₃ (A=Cs/Rb/K/MA/FA; B=Pb/Sn/Ge; X=I/Br/Cl)",
        prior_blend_weight=0.70,
    ),
    "abx3_mixed_a": FamilyRule(
        "abx3_mixed_a",
        1.0,
        2.8,
        1.55,
        3.90,
        True,
        "Mixed A-site FA/Cs/MA ABX₃ alloy — Eg between end-members",
        prior_blend_weight=0.75,
    ),
    "abx3_mixed_halide": FamilyRule(
        "abx3_mixed_halide",
        1.2,
        3.2,
        1.90,
        3.95,
        True,
        "Mixed-halide ABX₃ — Vegard-like I < Br < Cl Eg shift",
        prior_blend_weight=0.80,
    ),
    "halide_double_a2bbx6": FamilyRule(
        "halide_double_a2bbx6",
        1.5,
        3.8,
        2.20,
        3.95,
        True,
        "Halide double perovskite A₂B′B″X₆ (e.g. Cs₂AgBiBr₆) — often wider/indirect",
        prior_blend_weight=0.75,
    ),
    "vacancy_ordered_a2bx6": FamilyRule(
        "vacancy_ordered_a2bx6",
        1.0,
        5.2,
        1.80,
        4.00,
        True,
        "Vacancy-ordered A₂BX₆ (K₂GeI₆, Cs₂SnI₆) — iodides ~1–2 eV; Cl/Br much wider",
        prior_blend_weight=0.85,
    ),
    "a3b2x9_0d": FamilyRule(
        "a3b2x9_0d",
        1.6,
        3.4,
        2.20,
        3.90,
        True,
        "0D / layered A₃B₂X₉ (Cs₃Sb₂Br₉, Bi analogues)",
        prior_blend_weight=0.70,
    ),
    "oxide_perovskite": FamilyRule(
        "oxide_perovskite",
        2.0,
        5.5,
        3.20,
        4.10,
        True,
        "Oxide perovskite / double oxide — wider Eg; not primary halide screening",
        prior_blend_weight=0.60,
        halide_sensitive=False,
        notes="Eligible but low confidence for opto halide workflows",
    ),
    "rp_dj_2d": FamilyRule(
        "rp_dj_2d",
        1.5,
        3.5,
        2.40,
        3.80,
        False,
        "2D Ruddlesden–Popper / Dion–Jacobson / large-spacer monolayer — not 3D absorber",
        prior_blend_weight=0.5,
        notes="Blocked as standard 3D perovskite absorber",
    ),
    "contact_etl": FamilyRule(
        "contact_etl",
        1.5,
        4.5,
        3.20,
        4.00,
        False,
        "ETL / oxide contact — not an absorber",
        prior_blend_weight=0.9,
        halide_sensitive=False,
    ),
    "contact_htl": FamilyRule(
        "contact_htl",
        1.2,
        4.0,
        2.80,
        2.40,
        False,
        "HTL / organic contact — not an absorber",
        prior_blend_weight=0.9,
        halide_sensitive=False,
    ),
    "non_perovskite": FamilyRule(
        "non_perovskite",
        0.5,
        4.0,
        1.50,
        4.00,
        False,
        "Non-perovskite semiconductor (GaAs, CdTe, BeSiP₂, …)",
        prior_blend_weight=1.0,
        halide_sensitive=False,
    ),
    "unknown": FamilyRule(
        "unknown",
        0.2,
        6.0,
        1.80,
        3.80,
        False,
        "Unrecognized formula — not treated as perovskite absorber",
        prior_blend_weight=0.35,
        halide_sensitive=False,
    ),
}

# Halide Eg shift relative to iodide (family prior / heuristic)
HALIDE_EG_DELTA = {"I": 0.0, "Br": 0.55, "Cl": 1.25, "F": 1.80}
HALIDE_CHI_DELTA = {"I": -0.05, "Br": 0.0, "Cl": 0.08, "F": 0.12}

# Experimental / literature end-member Eg for Vegard (eV)
# Keys: (A_token, B, X) where A_token in Cs,Rb,K,MA,FA
ABX3_ENDMEMBER_EG: dict[tuple[str, str, str], float] = {
    # Pb
    ("Cs", "Pb", "I"): 1.73,
    ("Cs", "Pb", "Br"): 2.36,
    ("Cs", "Pb", "Cl"): 2.98,
    ("MA", "Pb", "I"): 1.55,
    ("MA", "Pb", "Br"): 2.32,
    ("MA", "Pb", "Cl"): 2.88,
    ("FA", "Pb", "I"): 1.48,
    ("FA", "Pb", "Br"): 2.23,
    ("FA", "Pb", "Cl"): 2.90,
    ("Rb", "Pb", "I"): 2.10,
    ("Rb", "Pb", "Br"): 2.50,
    ("K", "Pb", "I"): 2.20,
    # Sn
    ("Cs", "Sn", "I"): 1.30,
    ("Cs", "Sn", "Br"): 1.75,
    ("Cs", "Sn", "Cl"): 2.55,
    ("MA", "Sn", "I"): 1.30,
    ("MA", "Sn", "Br"): 1.75,
    ("MA", "Sn", "Cl"): 2.50,
    ("FA", "Sn", "I"): 1.35,
    ("FA", "Sn", "Br"): 1.80,
    ("FA", "Sn", "Cl"): 2.55,
    ("Rb", "Sn", "I"): 1.40,
    # Ge
    ("Cs", "Ge", "I"): 1.60,
    ("Cs", "Ge", "Br"): 2.15,
    ("Cs", "Ge", "Cl"): 2.70,
    ("MA", "Ge", "I"): 1.90,
    ("MA", "Ge", "Br"): 2.40,
    ("MA", "Ge", "Cl"): 2.95,
    ("FA", "Ge", "I"): 2.20,
    ("FA", "Ge", "Br"): 2.65,
    ("FA", "Ge", "Cl"): 3.15,
}

# Vacancy-ordered A₂BX₆ end-members (optical gaps)
# Halide identity is strict: I < Br < Cl within each (A,B) series.
A2BX6_ENDMEMBER_EG: dict[tuple[str, str, str], float] = {
    ("Cs", "Sn", "I"): 1.35,
    ("Cs", "Sn", "Br"): 3.23,
    ("Cs", "Sn", "Cl"): 4.89,
    # Xiao / Ju Ti vacancy-ordered family — Cs2TiI6 ≈1.56–1.65 (not Br≈1.8)
    ("Cs", "Ti", "I"): 1.58,
    ("Cs", "Ti", "Br"): 1.88,
    ("Cs", "Ti", "Cl"): 2.90,
    ("K", "Ti", "I"): 1.61,
    ("K", "Ti", "Br"): 1.85,
    ("K", "Ti", "Cl"): 2.85,
    ("Rb", "Ti", "I"): 1.60,
    ("Rb", "Ti", "Br"): 1.86,
    ("Rb", "Ti", "Cl"): 2.88,
    ("Cs", "Ge", "I"): 1.55,
    ("Cs", "Ge", "Br"): 2.30,
    ("Cs", "Ge", "Cl"): 3.10,
    ("K", "Ge", "I"): 1.62,
    ("K", "Ge", "Br"): 2.35,
    ("K", "Ge", "Cl"): 3.15,
    ("Rb", "Sn", "I"): 1.40,
    ("Rb", "Sn", "Br"): 3.00,
    ("Rb", "Sn", "Cl"): 4.60,
    # Pd / Pt vacancy-ordered analogues (literature optical / HSE priors)
    ("Cs", "Pd", "I"): 1.20,
    ("Cs", "Pd", "Br"): 1.67,
    ("Cs", "Pd", "Cl"): 2.40,
    ("Cs", "Pt", "I"): 1.40,
    ("Cs", "Pt", "Br"): 1.95,
    ("Cs", "Pt", "Cl"): 2.70,
}

# Halide double A₂B′B″X₆ priors (optical)
A2BBX6_ENDMEMBER_EG: dict[str, float] = {
    "Cs2AgBiBr6": 2.19,
    "Cs2AgBiCl6": 2.77,
    "Cs2AgBiI6": 1.75,
    "Cs2AgInCl6": 3.23,
    "Cs2AgInBr6": 2.50,
    "Rb2AgBiI6": 1.98,
    "Rb2AgBiBr6": 2.30,
    "Cs2AgSbBr6": 2.00,
    "Cs2NaBiCl6": 3.00,
}

# A₃B₂X₉
A3B2X9_ENDMEMBER_EG: dict[str, float] = {
    "Cs3Sb2I9": 2.05,
    "Cs3Sb2Br9": 2.85,
    "Cs3Sb2Cl9": 3.20,
    "Cs3Bi2I9": 2.00,
    "Cs3Bi2Br9": 2.60,
    "Rb3Sb2I9": 2.10,
    "MA3Bi2I9": 2.10,
}

# Non-absorber / non-perovskite blocks
NON_PEROVSKITE_ABSORBERS = {
    "BeSiP2",
    "GaAs",
    "CdTe",
    "CdSe",
    "Si",
    "Ge",
    "CIGS",
    "CZTS",
    "InP",
    "GaN",
    "AlN",
    "SiC",
    "ZnTe",
    "Zn3P2",
    # Metal-halide precursors (not ABX₃) — historically mis-tagged as ABX3 without A-site
    "PbI2",
    "PbBr2",
    "PbCl2",
    "SnI2",
    "SnBr2",
    "SnCl2",
    "GeI2",
    "GeBr2",
    "GeCl2",
}

# Vacuum-referenced Eg/χ for named contacts whose real band edges lie outside the
# generic ETL/HTL priors. Without these, every contact estimate lands in a narrow
# ~2.4–3.6 eV / χ≈2.4–4.0 window and broken-gap (Type III) alignment — the whole
# point of a deep-affinity hole-extraction oxide — is arithmetically unreachable.
#   deep-χ oxides: Meyer et al. Adv. Mater. 24, 5408 (2012) 10.1002/adma.201201630
#                  Kröger et al. Appl. Phys. Lett. 95, 123301 (2009) 10.1063/1.3231928
#   wide-gap insulators: standard optical Eg + UPS/IPES electron affinity
NAMED_CONTACT_BANDS: dict[str, tuple[float, float]] = {
    # material: (Eg_eV, chi_eV)
    "MoO3": (3.00, 6.70),
    "V2O5": (2.80, 6.60),
    "WO3": (3.10, 5.00),
    "MgO": (7.80, 0.85),
    "Al2O3": (8.80, 1.35),
    "SiO2": (9.00, 0.90),
    "HfO2": (5.70, 2.50),
    "ZrO2": (5.80, 2.50),
}

CONTACT_ETL = {
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
    "IGZO",
    "AZO",
    "ITO",
    "FTO",
}
CONTACT_HTL = {
    "MoO3",
    "NiO",
    "CuPc",
    "C6PcH2",
    "CuI",
    "CuSCN",
    "PTAA",
    "Spiro-OMeTAD",
    "Spiro",
    "V2O5",
    "MEH-PPV",
    "NPB",
    "nPB",
    "PEDOT",
    "PEDOT:PSS",
    "P3HT",
}


@dataclass
class SiteParse:
    formula: str
    a_fracs: dict[str, float] = field(default_factory=dict)
    b_primary: str | None = None
    b_secondary: str | None = None
    b_fracs: dict[str, float] = field(default_factory=dict)
    x_fracs: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)
    has_organic: bool = False
    is_mixed_a: bool = False
    is_mixed_x: bool = False
    stoich_hint: str | None = None  # ABX3 | A2BX6 | A2BBX6 | A3B2X9 | oxide | other


def _clean(formula: str) -> str:
    return base_name(canonicalize_material_alias(formula)).replace(" ", "")


def parse_a_site_fractions(formula: str) -> dict[str, float]:
    """Extract A-site cation fractions from formula string (before full expansion)."""
    s = _clean(formula)
    fracs: dict[str, float] = {}

    # Organic tokens FA0.83, MA, …
    for cat in sorted(ORGANIC_CATIONS.keys(), key=len, reverse=True):
        pat = re.compile(rf"(?<![A-Za-z]){cat}(\d*\.?\d*)(?![a-z])")
        for m in pat.finditer(s):
            mult = float(m.group(1)) if m.group(1) else 1.0
            fracs[cat] = fracs.get(cat, 0.0) + mult
        s = pat.sub("", s)

    # Inorganic A: Cs, Rb, K (and residual after organics removed)
    for cat in ("Cs", "Rb", "K"):
        pat = re.compile(rf"(?<![A-Za-z]){cat}(\d*\.?\d*)(?![a-z])")
        for m in pat.finditer(s):
            mult = float(m.group(1)) if m.group(1) else 1.0
            fracs[cat] = fracs.get(cat, 0.0) + mult

    # CH3NH3 / HC(NH2)2 spelled out → MA / FA (unicode already folded by _clean)
    raw = _clean(formula)
    if "CH3NH3" in raw.upper():
        if "MA" not in fracs:
            fracs["MA"] = fracs.get("MA", 0.0) + 1.0
    if "HC(NH2)2" in raw:
        if "FA" not in fracs:
            fracs["FA"] = fracs.get("FA", 0.0) + 1.0

    total = sum(fracs.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in fracs.items()}


def parse_sites(formula: str) -> SiteParse:
    """Composition + stoichiometry cues for family detection."""
    clean = _clean(formula)
    counts = parse_formula_counts(formula)
    a_fracs = parse_a_site_fractions(formula)

    x_raw = {h: counts.get(h, 0.0) for h in ("I", "Br", "Cl", "F") if counts.get(h, 0) > 0}
    x_tot = sum(x_raw.values()) or 1.0
    x_fracs = {k: v / x_tot for k, v in x_raw.items()}

    b_cands = ("Pb", "Sn", "Ge", "Ti", "Ag", "Bi", "Sb", "In", "Cu", "Au", "Na", "Tl")
    b_present = {el: counts.get(el, 0.0) for el in b_cands if counts.get(el, 0) > 0}
    # Primary B for ABX3: Pb/Sn/Ge preferred
    b_primary = None
    for pref in ("Pb", "Sn", "Ge", "Ti", "Sb", "Bi"):
        if pref in b_present:
            b_primary = pref
            break
    if b_primary is None and b_present:
        b_primary = max(b_present, key=b_present.get)

    b_secondary = None
    if len(b_present) >= 2:
        others = [el for el in b_present if el != b_primary]
        b_secondary = others[0] if others else None

    b_tot = sum(b_present.values()) or 1.0
    b_fracs = {k: v / b_tot for k, v in b_present.items()}

    halogens = sum(counts.get(h, 0) for h in ("F", "Cl", "Br", "I"))
    o_count = counts.get("O", 0)
    name_flat = clean.replace(".", "")

    stoich = "other"
    if o_count >= 3 and halogens == 0:
        stoich = "oxide"
    elif re.search(r".*3.*2.*9", name_flat) and halogens >= 6:
        stoich = "A3B2X9"
    elif (
        halogens >= 5.5
        and len(b_present) >= 2
        and any(el in b_present for el in ("Ag", "Cu", "Na", "In", "Bi", "Sb", "Au"))
    ):
        stoich = "A2BBX6"
    elif halogens >= 5.5 and re.search(r".*2.*6$", name_flat):
        # Distinguish A2BX6 (one B) vs double (two B)
        if len(b_present) >= 2 and any(
            el in b_present for el in ("Ag", "Cu", "Na", "In", "Bi", "Sb")
        ):
            stoich = "A2BBX6"
        else:
            stoich = "A2BX6"
    elif (
        b_primary in ("Pb", "Sn", "Ge")
        and o_count == 0
        and bool(a_fracs)  # require A-site (Cs/Rb/K/MA/FA/…); bare BX₂ ≠ perovskite
        and (
            # Standard ABX₃ with X≈3, or BX₂ precursors excluded (X≈2 without A already blocked).
            halogens >= 2.5
            # Common shorthand writes halide *fractions* that sum to 1
            # (…PbBr0.17I0.83 ≡ …Pb(I0.83Br0.17)3). Accept when A≈1 and B≈1.
            or (
                0.85 <= halogens <= 1.15
                and 0.85 <= sum(a_fracs.values()) <= 1.15
                and 0.85 <= float(b_present.get(b_primary, 0)) <= 1.15
                and len(x_fracs) >= 1
            )
        )
    ):
        stoich = "ABX3"

    return SiteParse(
        formula=clean,
        a_fracs=a_fracs,
        b_primary=b_primary,
        b_secondary=b_secondary,
        b_fracs=b_fracs,
        x_fracs=x_fracs,
        counts=counts,
        has_organic=bool(
            a_fracs.keys() & set(ORGANIC_CATIONS)
            or (counts.get("C", 0) > 0 and counts.get("N", 0) > 0)
        ),
        is_mixed_a=len(a_fracs) >= 2,
        is_mixed_x=len(x_fracs) >= 2,
        stoich_hint=stoich,
    )


def _canonical_block_key(name: str) -> str:
    return base_name(name).lower().replace(" ", "").replace("-", "")


def classify_family(formula: str) -> FamilyRule:
    """Return the FamilyRule for a formula (eligibility + priors)."""
    clean = _clean(formula)
    key = _canonical_block_key(clean)

    # Explicit non-perovskites
    for blocked in NON_PEROVSKITE_ABSORBERS:
        if key == _canonical_block_key(blocked):
            return FAMILIES["non_perovskite"]

    for etl in CONTACT_ETL:
        if key == _canonical_block_key(etl):
            return FAMILIES["contact_etl"]
    for htl in CONTACT_HTL:
        if key == _canonical_block_key(htl):
            return FAMILIES["contact_htl"]

    if _MONOLAYER_PREFIX.match(clean):
        return FAMILIES["rp_dj_2d"]

    # 2D RP/DJ spacers (PEA)2PbI4 etc. — large organic + not simple ABX3 A-site only
    if _RP_DJ_MARKERS.search(clean):
        sites = parse_sites(formula)
        # PEA/BA as sole large spacer → 2D
        if any(tok in clean.upper() for tok in ("PEA", "BA", "OA", "PDA", "BDA")):
            # Allow BA token only if butylammonium spacer pattern (BA)2…
            if re.search(r"\(?(PEA|BA|OA|PDA|BDA)\)?\d", clean, re.I) or clean.upper().startswith(
                ("PEA", "(PEA", "BA2", "(BA")
            ):
                return FAMILIES["rp_dj_2d"]
        if sites.stoich_hint == "ABX3" and sites.has_organic:
            # FA/MA alone is 3D — already handled; PEA caught above
            pass

    sites = parse_sites(formula)
    counts = sites.counts
    if not counts:
        # Named organics as HTL
        if key in {_canonical_block_key(x) for x in CONTACT_HTL}:
            return FAMILIES["contact_htl"]
        return FAMILIES["unknown"]

    if sites.stoich_hint == "A3B2X9":
        return FAMILIES["a3b2x9_0d"]
    if sites.stoich_hint == "A2BBX6":
        return FAMILIES["halide_double_a2bbx6"]
    if sites.stoich_hint == "A2BX6":
        return FAMILIES["vacancy_ordered_a2bx6"]
    if sites.stoich_hint == "ABX3":
        if sites.is_mixed_a:
            return FAMILIES["abx3_mixed_a"]
        if sites.is_mixed_x:
            return FAMILIES["abx3_mixed_halide"]
        return FAMILIES["abx3_halide_3d"]
    if sites.stoich_hint == "oxide":
        metals = {
            el
            for el in counts
            if el not in ("H", "C", "N", "O", "F", "Cl", "Br", "I")
        }
        # BaTiO3-like / double oxide A2BB'O6
        if counts.get("O", 0) >= 3 and len(metals) >= 2:
            # Simple contact oxides already caught; remaining oxide perovskites
            if key not in {_canonical_block_key(x) for x in CONTACT_ETL | CONTACT_HTL}:
                return FAMILIES["oxide_perovskite"]

    # Fallback: Pb/Sn/Ge + halide
    b = (counts.get("Pb", 0) + counts.get("Sn", 0) + counts.get("Ge", 0))
    hal = sum(counts.get(h, 0) for h in ("F", "Cl", "Br", "I"))
    if b > 0 and hal >= 2 and counts.get("O", 0) == 0:
        if sites.is_mixed_a:
            return FAMILIES["abx3_mixed_a"]
        if sites.is_mixed_x:
            return FAMILIES["abx3_mixed_halide"]
        return FAMILIES["abx3_halide_3d"]

    return FAMILIES["unknown"]


def looks_like_perovskite_absorber(formula: str) -> bool:
    """True if formula matches an absorber-eligible perovskite family."""
    fam = classify_family(formula)
    return fam.absorber_eligible


def halide_weighted_delta(x_fracs: dict[str, float], table: dict[str, float]) -> float:
    if not x_fracs:
        return 0.0
    return sum(table.get(x, 0.0) * f for x, f in x_fracs.items())


def _endmember_eg(a: str, b: str, x: str) -> float | None:
    return ABX3_ENDMEMBER_EG.get((a, b, x))


def vegard_abx3_eg(sites: SiteParse) -> tuple[float | None, str]:
    """Vegard interpolate ABX₃ Eg from A-site × halide end-members."""
    b = sites.b_primary
    if b not in ("Pb", "Sn", "Ge") or not sites.x_fracs:
        return None, "no_abx3_sites"

    a_fracs = sites.a_fracs or {"Cs": 1.0}
    # If A empty but organic counts suggest MA/FA via spelled formula
    if not sites.a_fracs and sites.has_organic:
        raw = sites.formula.upper()
        if "CH3NH3" in raw.replace(" ", "") or "MA" in raw:
            a_fracs = {"MA": 1.0}
        elif "HC(NH2)2" in sites.formula or "FA" in raw:
            a_fracs = {"FA": 1.0}

    eg = 0.0
    w_sum = 0.0
    missing = 0
    for a, fa in a_fracs.items():
        for x, fx in sites.x_fracs.items():
            val = _endmember_eg(a, b, x)
            if val is None:
                # Fallback: Cs end-member + small A-shift
                base = _endmember_eg("Cs", b, x)
                if base is None:
                    missing += 1
                    continue
                # FA slightly lower than Cs for Pb iodide; MA similar
                a_shift = {"FA": -0.15, "MA": -0.10, "Rb": 0.15, "K": 0.25}.get(a, 0.0)
                if b != "Pb":
                    a_shift *= 0.5
                val = base + a_shift
            eg += fa * fx * val
            w_sum += fa * fx

    if w_sum < 0.5:
        return None, "insufficient_endmembers"
    eg /= w_sum
    note = "vegard_abx3"
    if sites.is_mixed_a:
        note += "+mixed_A"
    if sites.is_mixed_x:
        note += "+mixed_X"
    if missing:
        note += f"+{missing}_fallback"
    return float(eg), note


def vegard_a2bx6_eg(sites: SiteParse) -> tuple[float | None, str]:
    a_fracs = sites.a_fracs or {"Cs": 1.0}
    b = sites.b_primary
    if not b or not sites.x_fracs:
        return None, "no_a2bx6"

    eg = 0.0
    w = 0.0
    for a, fa in a_fracs.items():
        if a not in ("Cs", "Rb", "K"):
            a = "Cs"
        for x, fx in sites.x_fracs.items():
            val = A2BX6_ENDMEMBER_EG.get((a, b, x))
            if val is None:
                # Halide ladder from nearest iodide/bromide
                for x_ref in ("I", "Br", "Cl"):
                    ref = A2BX6_ENDMEMBER_EG.get((a, b, x_ref)) or A2BX6_ENDMEMBER_EG.get(
                        ("Cs", b, x_ref)
                    )
                    if ref is not None:
                        val = ref + (
                            HALIDE_EG_DELTA.get(x, 0) - HALIDE_EG_DELTA.get(x_ref, 0)
                        )
                        # Vacancy-ordered Cl/Br jumps are larger than ABX3 — amplify
                        if x == "Cl" and x_ref == "I":
                            val = max(val, ref + 3.2)
                        elif x == "Br" and x_ref == "I":
                            val = max(val, ref + 1.6)
                        break
            if val is None:
                continue
            eg += fa * fx * val
            w += fa * fx
    if w < 0.3:
        return None, "no_a2bx6_endmembers"
    return float(eg / w), "vegard_a2bx6"


def family_prior_eg_chi(formula: str, role: str = "absorber") -> dict[str, Any]:
    """Rule-based Eg/χ prior (and optional Vegard) for a formula."""
    fam = classify_family(formula)
    sites = parse_sites(formula)
    clean = _clean(formula)

    # Named contacts with measured band edges outside the generic contact window
    if role in ("etl", "htl") and clean in NAMED_CONTACT_BANDS:
        eg, chi = NAMED_CONTACT_BANDS[clean]
        return {
            "family_id": fam.family_id,
            "Eg_eV": eg,
            "chi_eV": chi,
            "prior_blend_weight": 1.0,
            "vegard": False,
            "method": "named_contact_band_edges",
            "eligible": False,
            "eg_min": eg,
            "eg_max": eg,
        }

    # Contact role priors when estimating ETL/HTL unknowns
    if role == "etl" and fam.family_id in ("contact_etl", "unknown", "oxide_perovskite"):
        eg = 3.20 + halide_weighted_delta(sites.x_fracs, HALIDE_EG_DELTA) * 0.3
        if sites.counts.get("O", 0):
            eg = max(eg, 3.0)
        chi = 4.00
        return {
            "family_id": fam.family_id,
            "Eg_eV": float(min(max(eg, fam.eg_min), fam.eg_max)),
            "chi_eV": chi,
            "prior_blend_weight": 0.7,
            "vegard": False,
            "method": "contact_etl_prior",
            "eligible": False,
        }
    if role == "htl" and fam.family_id in ("contact_htl", "unknown"):
        # Named organics / polymers — optical Eg often ~1.6–2.2, not ROLE_EG_PRIOR 2.80
        named = {
            "P3HT": (1.90, 3.10),
            "MEH-PPV": (2.10, 2.80),
            "PTAA": (2.96, 2.30),
            "PEDOT:PSS": (1.60, 3.30),
            "PEDOT": (1.60, 3.30),
            "Spiro-OMeTAD": (3.00, 2.05),
            "CuPc": (1.70, 3.50),
        }
        if clean in named:
            eg, chi = named[clean]
            return {
                "family_id": fam.family_id,
                "Eg_eV": eg,
                "chi_eV": chi,
                "prior_blend_weight": 0.95,
                "vegard": False,
                "method": "organic_htl_named",
                "eligible": False,
            }
        eg = 2.80
        chi = 2.40
        return {
            "family_id": fam.family_id,
            "Eg_eV": eg,
            "chi_eV": chi,
            "prior_blend_weight": 0.7,
            "vegard": False,
            "method": "contact_htl_prior",
            "eligible": False,
        }

    vegard_eg: float | None = None
    vegard_note = ""
    method = "family_prior"

    if fam.family_id in ("abx3_halide_3d", "abx3_mixed_a", "abx3_mixed_halide"):
        vegard_eg, vegard_note = vegard_abx3_eg(sites)
    elif fam.family_id == "vacancy_ordered_a2bx6":
        vegard_eg, vegard_note = vegard_a2bx6_eg(sites)
    elif fam.family_id == "halide_double_a2bbx6":
        if clean in A2BBX6_ENDMEMBER_EG:
            vegard_eg = A2BBX6_ENDMEMBER_EG[clean]
            vegard_note = "exact_double_endmember"
        else:
            # Halide-scaled prior from Bi/Ag bromide baseline
            base = 2.20
            vegard_eg = base + halide_weighted_delta(sites.x_fracs, HALIDE_EG_DELTA)
            if sites.counts.get("In", 0) and "Cl" in sites.x_fracs:
                vegard_eg = max(vegard_eg, 3.0)
            vegard_note = "double_halide_ladder"
    elif fam.family_id == "a3b2x9_0d":
        if clean in A3B2X9_ENDMEMBER_EG:
            vegard_eg = A3B2X9_ENDMEMBER_EG[clean]
            vegard_note = "exact_a3b2x9"
        else:
            base = 2.05 if "I" in sites.x_fracs else 2.40
            vegard_eg = base + halide_weighted_delta(sites.x_fracs, {"I": 0, "Br": 0.8, "Cl": 1.15, "F": 1.5})
            vegard_note = "a3b2x9_halide_ladder"

    if vegard_eg is not None:
        eg = vegard_eg
        method = vegard_note or "vegard"
        blend_w = min(0.92, fam.prior_blend_weight + 0.15)
    else:
        eg = fam.eg_prior + halide_weighted_delta(sites.x_fracs, HALIDE_EG_DELTA)
        if fam.family_id == "vacancy_ordered_a2bx6":
            # Stronger Cl/Br open-gap prior when Vegard missed
            eg = fam.eg_prior + halide_weighted_delta(
                sites.x_fracs, {"I": 0.0, "Br": 1.70, "Cl": 3.40, "F": 4.0}
            )
        blend_w = fam.prior_blend_weight

    eg = float(min(max(eg, fam.eg_min), fam.eg_max))
    chi = fam.chi_prior + halide_weighted_delta(sites.x_fracs, HALIDE_CHI_DELTA)
    if fam.family_id == "vacancy_ordered_a2bx6" and "I" in sites.x_fracs:
        chi = max(chi, 4.0)
    chi = float(min(max(chi, 1.5), 4.9))

    return {
        "family_id": fam.family_id,
        "Eg_eV": eg,
        "chi_eV": chi,
        "prior_blend_weight": blend_w,
        "vegard": vegard_eg is not None,
        "method": method,
        "eligible": fam.absorber_eligible,
        "eg_min": fam.eg_min,
        "eg_max": fam.eg_max,
        "description": fam.description,
    }


def confidence_for_estimate(
    *,
    family_id: str,
    vegard: bool,
    ml_delta: float,
    in_library: bool = False,
) -> str:
    """high | medium | low — UI caution when low (OOD)."""
    if in_library:
        return "high"
    if family_id == "unknown" or family_id.startswith("contact") or family_id == "non_perovskite":
        return "low"
    if family_id == "oxide_perovskite" or family_id == "rp_dj_2d":
        return "low"
    if vegard and abs(ml_delta) < 0.6:
        return "high" if family_id.startswith("abx3") else "medium"
    if vegard:
        return "medium"
    if family_id == "vacancy_ordered_a2bx6":
        return "medium" if abs(ml_delta) < 1.0 else "low"
    if family_id.startswith("abx3"):
        return "medium"
    return "low"


def family_feature_flags(formula: str) -> dict[str, float]:
    """Extra numeric flags for ML feature vectors."""
    fam = classify_family(formula)
    sites = parse_sites(formula)
    flags = {
        "family_abx3": 1.0 if fam.family_id.startswith("abx3") else 0.0,
        "family_double": 1.0 if fam.family_id == "halide_double_a2bbx6" else 0.0,
        "family_a2bx6": 1.0 if fam.family_id == "vacancy_ordered_a2bx6" else 0.0,
        "family_a3b2x9": 1.0 if fam.family_id == "a3b2x9_0d" else 0.0,
        "family_oxide": 1.0 if fam.family_id == "oxide_perovskite" else 0.0,
        "mixed_a": 1.0 if sites.is_mixed_a else 0.0,
        "mixed_x": 1.0 if sites.is_mixed_x else 0.0,
        "prior_eg": family_prior_eg_chi(formula)["Eg_eV"],
        "prior_chi": family_prior_eg_chi(formula)["chi_eV"],
    }
    return flags


def domain_context_rules_text() -> str:
    """Compact rule text for optional LLM formula-only prompts."""
    lines = [
        "PEROVSKITE FAMILY RULES (formula-only; no literature recall):",
        "1) ABX3 halide (A=Cs/Rb/K/MA/FA; B=Pb/Sn/Ge; X=I/Br/Cl): Vegard from end-members; I<Br<Cl Eg.",
        "2) Mixed A-site: Eg = sum_i f_Ai * Eg(AiBX3).",
        "3) Halide double A2B'B''X6: wider/indirect; Br~2.0–2.3, Cl~2.7–3.3 eV priors.",
        "4) Vacancy-ordered A2BX6: iodides ~1.3–1.8 eV; Br/Cl much wider (Cs2SnBr6~3.2, Cs2SnCl6~4.9).",
        "5) A3B2X9: Sb/Bi iodides ~2.0 eV; bromides ~2.6–2.9 eV.",
        "6) Oxide perovskites: wide Eg; not primary halide absorbers.",
        "7) 2D RP/DJ (PEA/BA spacers) and monolayers: NOT standard 3D absorbers — block.",
        "8) Contacts ZnO/TiO2/SnO2/BaSnO3/MoO3/NiO/organics: ETL/HTL only — block as absorber.",
        "9) BeSiP2/GaAs/CdTe: non-perovskite — block.",
        "χ absorber halide ~3.7–4.2; ETL ~4.0; HTL ~1.7–2.5.",
    ]
    return "\n".join(lines)
