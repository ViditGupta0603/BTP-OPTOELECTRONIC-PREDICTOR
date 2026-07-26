"""Band alignment helpers for literature-curated optoelectronic stacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    name: str
    eg: float
    chi: float  # electron affinity χ (eV), SCAPS / vacuum convention

    @classmethod
    def from_vbm_eg(cls, name: str, ev_vbm: float, eg: float) -> Layer:
        """Build layer from Paper 1-style VBM (eV vs vacuum) and HSE06 Eg."""
        chi = -(ev_vbm + eg)
        return cls(name=name, eg=eg, chi=chi)

    @classmethod
    def from_ionization_potential(cls, name: str, ip: float, eg: float) -> Layer:
        """Experimental UPS/XPS: IP = vacuum − VBM → χ = IP − Eg."""
        return cls(name=name, eg=eg, chi=ip - eg)

    @property
    def cbm(self) -> float:
        return -self.chi

    @property
    def vbm(self) -> float:
        return -(self.chi + self.eg)


def junction_type(a: Layer, b: Layer) -> str:
    """Anderson heterostructure Type from vacuum-referenced band edges.

    Edges use the project convention ``CBM = -χ``, ``VBM = -(χ + Eg)``.

    - **Type I** (straddling): one material's gap contains the other
    - **Type II** (staggered): band edges offset in the same direction
    - **Type III** (broken gap): VBM of one lies at or above CBM of the other
    """
    cb_a, vb_a = a.cbm, a.vbm
    cb_b, vb_b = b.cbm, b.vbm

    # Broken gap — valence band of one overlaps/exceeds conduction band of the other
    if vb_a >= cb_b or vb_b >= cb_a:
        return "Type III"

    # Straddling — A inside B or B inside A (electron energy: higher = less negative)
    a_inside_b = vb_b <= vb_a and cb_a <= cb_b
    b_inside_a = vb_a <= vb_b and cb_b <= cb_a
    if a_inside_b or b_inside_a:
        return "Type I"

    return "Type II"


def optoelectronic_suitability(etl_type: str | None, htl_type: str | None) -> dict:
    """Screen stack for optoelectronic use from junction Types (not PCE).

    Rule (project convention):
      - Type I / Type II → acceptable for charge confinement or separation
      - Type III (broken gap) → generally not preferred for standard opto stacks
      - YES if both interfaces are Type I or Type II
      - MARGINAL if exactly one is Type III
      - NO if both are Type III or a Type is missing
    """
    ok = {"Type I", "Type II"}
    bad = "Type III"
    if not etl_type or not htl_type:
        return {
            "suitable": False,
            "verdict": "UNKNOWN",
            "reason": "Missing junction Type — cannot screen for optoelectronic use.",
        }
    etl_ok = etl_type in ok
    htl_ok = htl_type in ok
    if etl_ok and htl_ok:
        return {
            "suitable": True,
            "verdict": "YES",
            "reason": (
                f"Absorber–ETL is {etl_type} and absorber–HTL is {htl_type}. "
                "Both are Type I/II — suitable for optoelectronic stack screening."
            ),
        }
    if (etl_type == bad) ^ (htl_type == bad):
        return {
            "suitable": False,
            "verdict": "MARGINAL",
            "reason": (
                f"Absorber–ETL {etl_type}, absorber–HTL {htl_type}. "
                "One interface is Type III (broken gap) — usually not preferred; review carefully."
            ),
        }
    return {
        "suitable": False,
        "verdict": "NO",
        "reason": (
            f"Absorber–ETL {etl_type}, absorber–HTL {htl_type}. "
            "Type III broken-gap alignment(s) — not recommended for typical optoelectronic devices."
        ),
    }


def cbo_absorber_etl(absorber: Layer, etl: Layer) -> float:
    return etl.cbm - absorber.cbm


def vbo_absorber_htl(absorber: Layer, htl: Layer) -> float:
    return htl.vbm - absorber.vbm


def stack_row(
    absorber: Layer,
    etl: Layer,
    htl: Layer,
    source_doi: str,
    source_paper: str,
    **meta: str,
) -> dict[str, str | float]:
    row: dict[str, str | float] = {
        "material_absorber": absorber.name,
        "material_etl": etl.name,
        "material_htl": htl.name,
        "absorber_band_gap_eV": absorber.eg,
        "etl_band_gap_eV": etl.eg,
        "htl_band_gap_eV": htl.eg,
        "cbo_eV": round(cbo_absorber_etl(absorber, etl), 4),
        "vbo_eV": round(vbo_absorber_htl(absorber, htl), 4),
        "absorber_etl_type": junction_type(absorber, etl),
        "absorber_htl_type": junction_type(absorber, htl),
        "source_doi": source_doi,
        "source_paper": source_paper,
    }
    row.update(meta)
    return row


def verify_row(row: dict[str, str | float], tol: float = 0.06) -> list[str]:
    """Return list of verification errors (empty = pass)."""
    errors: list[str] = []
    try:
        abs_l = Layer(str(row["material_absorber"]), float(row["absorber_band_gap_eV"]), 0.0)
        etl = Layer(str(row["material_etl"]), float(row["etl_band_gap_eV"]), 0.0)
        htl = Layer(str(row["material_htl"]), float(row["htl_band_gap_eV"]), 0.0)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"parse error: {exc}"]

    # Reconstruct chi from stored offsets if layers share names - need chi in row
    # Verification done in build script with full Layer objects
    if float(row.get("absorber_band_gap_eV", -1)) <= 0:
        errors.append("absorber_band_gap_eV <= 0")
    if float(row.get("etl_band_gap_eV", -1)) <= 0:
        errors.append("etl_band_gap_eV <= 0")
    if float(row.get("htl_band_gap_eV", -1)) <= 0:
        errors.append("htl_band_gap_eV <= 0")
    if not str(row.get("source_doi", "")).startswith("10."):
        errors.append("missing source_doi")
    return errors


def verify_stack(absorber: Layer, etl: Layer, htl: Layer, row: dict, tol: float = 0.06) -> list[str]:
    errors: list[str] = []
    for label, expected, actual in [
        ("cbo_eV", cbo_absorber_etl(absorber, etl), float(row["cbo_eV"])),
        ("vbo_eV", vbo_absorber_htl(absorber, htl), float(row["vbo_eV"])),
        ("absorber_etl_type", junction_type(absorber, etl), str(row["absorber_etl_type"])),
        ("absorber_htl_type", junction_type(absorber, htl), str(row["absorber_htl_type"])),
    ]:
        if isinstance(expected, float):
            if abs(expected - actual) > tol:
                errors.append(f"{label}: expected {expected:.4f}, got {actual}")
        elif expected != actual:
            errors.append(f"{label}: expected {expected}, got {actual}")
    return errors
