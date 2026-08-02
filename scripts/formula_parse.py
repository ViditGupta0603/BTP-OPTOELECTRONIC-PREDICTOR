"""Shared formula parsing for perovskite / contact materials.

Handles organic cation abbreviations (FA/MA/…), mixed A-site fractions,
and parenthesized groups like HC(NH2)2.
"""
from __future__ import annotations

import re
import unicodedata

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")
_GROUP = re.compile(r"\(([A-Za-z0-9.]+)\)(\d*\.?\d*)")

# Unicode spellings folded before any parsing, lookup or eligibility check.
# NFKC already covers sub/superscript digits (CH₃NH₃PbI₃, Cs²), full-width forms
# (ＣＨ３…) and no-break spaces; the characters below have no compatibility
# decomposition and must be mapped explicitly.
_DASH_VARIANTS = "\u2212\u2013\u2014\u2012\u2010\u2011\u2015"  # − – — ‒ ‐ ‑ ―
_INTERPUNCT_VARIANTS = "\u00b7\u2022\u2027\u2219\u22c5\u30fb\uff65"  # · • ‧ ∙ ⋅ ・ ･
_INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\u00ad"

_TEXT_FOLD: dict[int, str | None] = {ord(c): "-" for c in _DASH_VARIANTS}
_TEXT_FOLD.update({ord(c): "." for c in _INTERPUNCT_VARIANTS})
_TEXT_FOLD.update({ord(c): None for c in _INVISIBLE})

# _ELEMENT only matches capitalised symbols, so "nio2" would parse to zero
# elements and the estimator would answer with its training mean instead of a
# number for NiO2. The full table below lets the fold re-case such input.
_ELEMENT_SYMBOLS: frozenset[str] = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu
    Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs
    Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl
    Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh
    Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split()
)

_BARE_FORMULA = re.compile(r"^[A-Za-z0-9().]+$")

# Symbols the re-caser must never produce: no photovoltaic absorber or contact
# contains them, while their spellings collide with common text (Nh in NH3, Cf
# in CFTS, Ts in CNTS), which would turn good input into exotic chemistry.
_RECASE_SYMBOLS: frozenset[str] = _ELEMENT_SYMBOLS - frozenset(
    """
    Tc Pm Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh
    Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og
    """.split()
)

# Stoichiometric atom counts for common organic A-site cations
ORGANIC_CATIONS: dict[str, dict[str, float]] = {
    "FA": {"C": 1.0, "H": 5.0, "N": 2.0},  # formamidinium CH5N2+
    "MA": {"C": 1.0, "H": 6.0, "N": 1.0},  # methylammonium CH3NH3+
    "GA": {"C": 1.0, "H": 6.0, "N": 3.0},  # guanidinium
    "EA": {"C": 2.0, "H": 8.0, "N": 1.0},  # ethylammonium
    "PEA": {"C": 8.0, "H": 12.0, "N": 1.0},  # phenethylammonium (approx)
    "BA": {"C": 4.0, "H": 12.0, "N": 1.0},  # butylammonium
}

# Cations that can be read case-insensitively: "Fa", "Ma", "Ea" and "Pea" are not
# element symbols, so FaSnI3 can only mean formamidinium. BA and GA are excluded
# because "Ba" and "Ga" are barium and gallium — BaSnO3 and GaAs keep their
# elements. Longest first so PEA is tried before EA.
_CASE_SAFE_CATIONS: tuple[str, ...] = tuple(
    sorted(
        (c for c in ORGANIC_CATIONS if c.capitalize() not in _ELEMENT_SYMBOLS),
        key=len,
        reverse=True,
    )
)

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
    "spiro-meotad": "Spiro-OMeTAD",
    "spiromeotad": "Spiro-OMeTAD",
    # Non-stoichiometric nickel oxide spellings → SCAPS NiO (Eg≈3.6 eV)
    "niox": "NiO",
    "nio_x": "NiO",
    "nioₓ": "NiO",
    # Triple-cation mixed-halide (Saliba) — fractional-X shorthand → parenthetical form
    "cs0.06fa0.78ma0.16pbbr0.17i0.83": "Cs0.05(FA0.83MA0.17)0.95Pb(I0.83Br0.17)3",
    "cs0.05fa0.79ma0.16pbbr0.17i0.83": "Cs0.05(FA0.83MA0.17)0.95Pb(I0.83Br0.17)3",
    "cs0.05(fa0.83ma0.17)0.95pb(i0.83br0.17)3": "Cs0.05(FA0.83MA0.17)0.95Pb(I0.83Br0.17)3",
    "csfama": "Cs0.05(FA0.83MA0.17)0.95Pb(I0.83Br0.17)3",
}


def normalize_formula_text(name: str) -> str:
    """Fold every unicode spelling of a formula onto one ASCII canonical form.

    This is the single fold used by the eligibility gate, the parser, alias
    lookup and the estimator, so CH₃NH₃PbI₃, ＣＨ３ＮＨ３ＰｂＩ３ and CH3NH3PbI3
    can never be treated as different materials.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).translate(_TEXT_FOLD)
    return _recase_formula(re.sub(r"\s+", " ", s).strip())


def _recase_formula(s: str) -> str:
    """Rewrite a bare formula onto real element symbols (nio2 / Nio2 → NiO2).

    Anything with a space, dash or colon is a trade name (Spiro-OMeTAD,
    PEDOT:PSS) rather than a formula and is left alone. If any fragment is not a
    real element or organic cation the input is returned untouched, so the tool
    never invents chemistry for a typo; correctly written formulas re-tokenise
    onto themselves.
    """
    letters = [c for c in s if c.isalpha()]
    if not letters or not _BARE_FORMULA.match(s):
        return s

    # Case is meaningful in mixed-case text: Ba/Ga are barium/gallium while
    # BA/GA are butylammonium/guanidinium. If such a string already tokenises
    # as written, it is correct and must be left exactly as the user typed it.
    # A lowercase prefix on a capital is nomenclature (nPB, tBP), not a formula.
    if re.match(r"^[a-z][A-Z]", s):
        return s

    single_case = all(c.islower() for c in letters) or all(c.isupper() for c in letters)
    if not single_case:
        # "Fa"/"Ma" satisfy the [A-Z][a-z] element pattern without being elements,
        # so the orphan check below would accept FaSnI3 and the parser would then
        # invent an element "Fa". Normalise those cations before the check.
        s = _upcase_case_safe_cations(s)
        if _tokenises_as_written(s) or not _has_orphan_letters(s):
            return s

    cations = sorted(ORGANIC_CATIONS, key=len, reverse=True)
    out: list[str] = []
    i = 0
    while i < len(s):
        if not s[i].isalpha():
            out.append(s[i])
            i += 1
            continue
        two, one = s[i : i + 2].capitalize(), s[i].upper()
        # Elements win over cations so basno3 reads as BaSnO3, not BA+SnO3;
        # mapbi3 still reaches MA because "Ma" is not an element.
        if two in _RECASE_SYMBOLS:
            out.append(two)
            i += 2
            continue
        cat = next((c for c in cations if s[i:].upper().startswith(c.upper())), None)
        if cat is not None:
            out.append(cat)
            i += len(cat)
        elif one in _RECASE_SYMBOLS:
            out.append(one)
            i += 1
        else:
            return s
    recased = "".join(out)
    # Long all-caps strings that do not end in an oxide/halide/chalcogenide are
    # trade abbreviations (C6TBTAPH2), not formulas — keep the original spelling.
    if (
        all(c.isupper() for c in letters)
        and len(letters) > 6
        and not re.search(r"(?:O|F|Cl|Br|I|S|Se|Te)\d*\.?\d*$", recased)
    ):
        return s
    return recased


def _upcase_case_safe_cations(s: str) -> str:
    """Rewrite FaSnI3 / MaPbI3 onto FASnI3 / MAPbI3, leaving BaSnO3 and GaAs alone.

    The trailing lookahead must stay case-sensitive (a following lowercase letter
    would mean the capitals belong to an element), so the cation itself is spelled
    as explicit character classes rather than with a global IGNORECASE flag.
    """
    for cat in _CASE_SAFE_CATIONS:
        body = "".join(f"[{c.upper()}{c.lower()}]" for c in cat)
        s = re.sub(rf"(?<![A-Za-z]){body}(?![a-z])", cat, s)
    return s


def _has_orphan_letters(s: str) -> bool:
    """True if _ELEMENT skips letters outright, as in Nio2 where "o" is dropped.

    Abbreviations such as CuPc are consumed whole (Cu + Pc) even though Pc is
    not an element, so they are left alone; only genuinely dropped characters
    justify rewriting what the user typed.
    """
    covered = bytearray(len(s))
    for m in _ELEMENT.finditer(s):
        for k in range(m.start(), m.end()):
            covered[k] = 1
    return any(c.isalpha() and not covered[k] for k, c in enumerate(s))


def _tokenises_as_written(s: str) -> bool:
    """True if s splits cleanly into cations/elements using the case as typed."""
    cations = sorted(ORGANIC_CATIONS, key=len, reverse=True)
    i = 0
    while i < len(s):
        if not s[i].isalpha():
            i += 1
            continue
        # A trailing lowercase letter means the capitals belong to two elements
        # (BAs is B+As, never butylammonium), matching the organic-cation split.
        cat = next(
            (
                c
                for c in cations
                if s.startswith(c, i) and not s[i + len(c) : i + len(c) + 1].islower()
            ),
            None,
        )
        if cat is not None:
            i += len(cat)
        elif s[i : i + 2] in _ELEMENT_SYMBOLS:
            i += 2
        elif s[i] in _ELEMENT_SYMBOLS:
            i += 1
        else:
            return False
    return True


def base_name(name: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", normalize_formula_text(name))


def canonicalize_material_alias(name: str) -> str:
    """Map FAPbI3 / MAPbI3 / Spiro shorthand to canonical library names.

    Matching is exact (after the unicode fold) and halide-sensitive:
    FAPbBr3 must never resolve via an FAPbI3 / FAPb prefix.
    """
    s = base_name(name)
    key = s.lower().replace(" ", "")
    if key in MATERIAL_ALIASES:
        return MATERIAL_ALIASES[key]
    # FA/MA mixed-A formulas (FA0.83Cs0.17PbI3) stay as-is for parsing
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
    clean = base_name(formula).replace(" ", "")
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
