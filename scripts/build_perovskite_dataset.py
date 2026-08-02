"""Build a NEW perovskite-only dataset (does not modify opto_literature_dataset.csv).

Sources (already used or previously surveyed in this repo):
  A) K2TiI6 SCAPS stacks — DOI 10.1038/s41598-025-98351-y (40 full stacks)
  B) CsPb0.625Zn0.375IBr2 SCAPS stacks — DOI 10.1038/s41598-024-81797-x (40 full stacks)
  C) Paper 5 Pb-free halide double perovskites — DOI 10.1038/s41524-019-0177-0
     (absorber Eg library; local raw/paper5_double_perovskites.csv)
  D) Pilania et al. oxide double perovskite gaps (matminer `double_perovskites_gap`)
     — DOI 10.1038/srep19375 / figshare (absorber Eg library)
  E) Dubey et al. Cs3Sb2Br9 SCAPS — DOI 10.1016/j.nxmate.2026.102491 (1 full stack;
     verified via scripts/verify_research_papers.py)
  F) Raj et al. K2GeI6 DFT gap — DOI 10.1007/s44291-026-00245-4 (absorber-only;
     stack Table S1 SI not available in main PDF)
  G) Verified lead-halide ABX3 perovskites — literature/DFT Eg with DOI
     (local raw/verified_lead_halide_perovskites.csv; record_type=verified_external)
  H) Expansion absorbers with literature Eg+χ (MAPbI3, FAPbI3, CsPbI3, FASnI3, …)
     crossed with a curated ETL×HTL grid including Type I/II/III corner contacts
  I) CdTe paper (pratyush.pdf / DOI 10.1002/pssb.70269) — training-only corner-case
     stacks tagged non-perovskite; inference still blocks CdTe as absorber

Outputs (new files only):
  data/perovskite_stack_dataset.csv
  data/perovskite_absorber_library.csv
  data/raw/pilania_double_perovskites_gap.csv
  data/perovskite_dataset_build_meta.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from literature_bands import Layer, stack_row, verify_stack  # noqa: E402
from perovskite_rules import NAMED_CONTACT_BANDS  # noqa: E402

RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

STACK_OUT = DATA / "perovskite_stack_dataset.csv"
ABSORBER_OUT = DATA / "perovskite_absorber_library.csv"
PILANIA_RAW = RAW / "pilania_double_perovskites_gap.csv"
META_OUT = DATA / "perovskite_dataset_build_meta.json"
EXPANSION_ABS = RAW / "verified_expansion_absorbers.csv"
CDTE_RAW = RAW / "paper_cdte_scaps_materials.csv"
STACK_COLUMNS = [
    "material_absorber",
    "material_etl",
    "material_htl",
    "absorber_band_gap_eV",
    "etl_band_gap_eV",
    "htl_band_gap_eV",
    "cbo_eV",
    "vbo_eV",
    "absorber_etl_type",
    "absorber_htl_type",
    "gap_method",
    "functional",
    "gap_type",
    "material_class",
    "perovskite_family",
    "record_type",
    "combo_mode",
    "source_doi",
    "source_paper",
]

# Curated contacts for diversity / corner-case expansion (not full SCAPS pool).
# Wide-gap MgO/Al2O3 → Type I; deep-χ MoO3/V2O5 → Type III; ZnO/TiO2/NiO → Type II-ish.
# Keep grid modest so total stacks stay near ~1000 with the SCAPS pool.
EXPANSION_ETL_NAMES = ("TiO2", "SnO2", "MgO", "Al2O3")
EXPANSION_HTL_NAMES = ("NiO", "Spiro-OMeTAD", "MoO3", "V2O5")
# Prefer these absorbers for expansion diversity (subset of verified_expansion file).
EXPANSION_ABSORBER_ALLOW = {
    "CH3NH3PbI3",
    "HC(NH2)2PbI3",
    "CsPbI3",
    "HC(NH2)2SnI3",
    "CsSnBr3",
    "Cs2AgBiBr6",
    "K2GeI6",
}

VERIFIED_LEAD_HALIDE_RAW = RAW / "verified_lead_halide_perovskites.csv"

ABSORBER_COLUMNS = [
    "material_absorber",
    "absorber_band_gap_eV",
    "phase",
    "a_site",
    "a2_site",
    "b1_site",
    "b2_site",
    "x_site",
    "heat_of_formation_eV_per_atom",
    "gap_method",
    "functional",
    "gap_type",
    "material_class",
    "perovskite_family",
    "record_type",
    "source_doi",
    "source_paper",
]


def load_scaps(path: Path) -> tuple[Layer | None, dict[str, Layer], dict[str, Layer], str]:
    absorber = None
    etls: dict[str, Layer] = {}
    htls: dict[str, Layer] = {}
    doi = ""
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doi = row.get("source_doi", doi)
            if not row.get("chi_eV"):
                continue
            layer = Layer(row["material"], float(row["Eg_eV"]), float(row["chi_eV"]))
            if row["layer_role"] == "absorber":
                absorber = layer
            elif row["layer_role"] == "etl":
                etls[layer.name] = layer
            elif row["layer_role"] == "htl":
                htls[layer.name] = layer
    return absorber, etls, htls, doi


def _merge_layers(dst: dict[str, Layer], src: dict[str, Layer]) -> None:
    """Keep first-seen Eg/χ; prefer TiO2 Eg >= 2.9 if replacing."""
    for name, layer in src.items():
        if name not in dst:
            dst[name] = layer
            continue
        if name.startswith("TiO2") and layer.eg >= 2.9 and dst[name].eg < layer.eg:
            dst[name] = layer


def _apply_physical_contact_overrides(etls: dict[str, Layer], htls: dict[str, Layer]) -> None:
    """Replace SCAPS fitting χ for deep-affinity oxides / inject wide-gap buffers.

    SCAPS tables often quote non-physical MoO3/V2O5 χ (~2.3–3.4 eV). UPS/IPES
    literature places them at χ≈6.6–6.7 eV — required for Type III broken-gap
    labels. Wide-gap MgO/Al2O3 are added for genuine Type I straddling cases.
    Keep additions minimal so total stacks stay near ~1000.
    """
    for name in ("MoO3", "V2O5"):
        eg, chi = NAMED_CONTACT_BANDS[name]
        htls[name] = Layer(name, eg, chi)
    for name in ("MgO", "Al2O3"):
        eg, chi = NAMED_CONTACT_BANDS[name]
        etls[name] = Layer(name, eg, chi)
    if "Spiro-OMeTAD" not in htls:
        htls["Spiro-OMeTAD"] = Layer("Spiro-OMeTAD", 3.00, 2.05)
    # PSC-aligned SnO2 χ (BeSiP2 raw had 3.9; ETL library / CdTe paper use 4.0)
    if "SnO2" in etls:
        etls["SnO2"] = Layer("SnO2", etls["SnO2"].eg, 4.00)
    if "CdS" in etls:
        # Keep first-seen Eg; unify χ to CsPb SCAPS 4.18 (CdTe paper had 4.20)
        etls["CdS"] = Layer("CdS", etls["CdS"].eg, 4.18)


def _cross_stacks(
    absorbers: list[tuple[Layer, str, str, str]],
    etls: dict[str, Layer],
    htls: dict[str, Layer],
    *,
    record_type: str,
    combo_mode: str,
    gap_method: str = "SCAPS_literature_table",
    functional: str = "literature_chi_Eg",
    material_class: str = "bulk_thin_film_device",
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    failures: list[dict] = []
    for absorber, label, family, doi in absorbers:
        meta = {
            "gap_method": gap_method,
            "functional": functional,
            "gap_type": "device_simulation_parameter",
            "material_class": material_class,
            "perovskite_family": family,
            "record_type": record_type,
            "combo_mode": combo_mode,
        }
        for etl in etls.values():
            for htl in htls.values():
                row = stack_row(absorber, etl, htl, doi, label, **meta)
                msgs = verify_stack(absorber, etl, htl, row, tol=0.06)
                if msgs:
                    failures.append({"row": row, "msgs": msgs})
                rows.append(row)
    return rows, failures


def _load_pooled_scaps_contacts() -> tuple[dict[str, Layer], dict[str, Layer]]:
    contact_files = [
        RAW / "paper4_scaps_materials.csv",
        RAW / "paper_cs_pb_scaps_materials.csv",
        RAW / "paper_cs3sb2br9_scaps_materials.csv",
        RAW / "paper_besip2_scaps_materials.csv",
    ]
    etls: dict[str, Layer] = {}
    htls: dict[str, Layer] = {}
    for path in contact_files:
        _, e, h, _ = load_scaps(path)
        _merge_layers(etls, e)
        _merge_layers(htls, h)
    _apply_physical_contact_overrides(etls, htls)
    return etls, htls

def build_scaps_pooled_stacks() -> tuple[list[dict], list[dict]]:
    """Each SCAPS perovskite absorber × pooled ETL × pooled HTL (with corner overrides)."""
    absorber_specs = [
        (
            RAW / "paper4_scaps_materials.csv",
            "K2TiI6 SCAPS + pooled contacts",
            "vacancy_ordered_halide_perovskite",
            "10.1038/s41598-025-98351-y",
        ),
        (
            RAW / "paper_cs_pb_scaps_materials.csv",
            "CsPbZnIBr2 SCAPS + pooled contacts",
            "halide_perovskite_alloy",
            "10.1038/s41598-024-81797-x",
        ),
        (
            RAW / "paper_cs3sb2br9_scaps_materials.csv",
            "Cs3Sb2Br9 SCAPS + pooled contacts",
            "Cs3Sb2X9_perovskite_inspired",
            "10.1016/j.nxmate.2026.102491",
        ),
    ]
    absorbers: list[tuple[Layer, str, str, str]] = []
    for path, label, family, expect_doi in absorber_specs:
        abs_l, _, _, doi = load_scaps(path)
        assert abs_l is not None, path
        assert doi == expect_doi, (doi, expect_doi)
        absorbers.append((abs_l, label, family, doi))

    etls, htls = _load_pooled_scaps_contacts()
    return _cross_stacks(
        absorbers,
        etls,
        htls,
        record_type="full_stack",
        combo_mode="pooled_etl_htl",
    )


def build_expansion_absorber_stacks() -> tuple[list[dict], list[dict]]:
    """Diverse ABX3 / A2BX6 absorbers × curated corner-case ETL×HTL grid."""
    if not EXPANSION_ABS.exists():
        return [], []
    pooled_etls, pooled_htls = _load_pooled_scaps_contacts()
    # ZnO for expansion only (not in full SCAPS cross — keeps n≈1000)
    if "ZnO" not in pooled_etls:
        pooled_etls = dict(pooled_etls)
        pooled_etls["ZnO"] = Layer("ZnO", 3.30, 4.00)
    etls = {n: pooled_etls[n] for n in EXPANSION_ETL_NAMES if n in pooled_etls}
    htls = {n: pooled_htls[n] for n in EXPANSION_HTL_NAMES if n in pooled_htls}
    # Ensure deep-affinity / wide-gap present even if pool missing a name
    for n in EXPANSION_ETL_NAMES:
        if n not in etls and n in NAMED_CONTACT_BANDS:
            eg, chi = NAMED_CONTACT_BANDS[n]
            etls[n] = Layer(n, eg, chi)
    for n in EXPANSION_HTL_NAMES:
        if n not in htls and n in NAMED_CONTACT_BANDS:
            eg, chi = NAMED_CONTACT_BANDS[n]
            htls[n] = Layer(n, eg, chi)

    absorbers: list[tuple[Layer, str, str, str]] = []
    with EXPANSION_ABS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = (r.get("material") or "").strip()
            if not name or name not in EXPANSION_ABSORBER_ALLOW:
                continue
            layer = Layer(name, float(r["Eg_eV"]), float(r["chi_eV"]))
            label = f"{name} expansion × curated corner contacts"
            family = r.get("perovskite_family") or "halide_perovskite"
            doi = r.get("source_doi") or "10.1038/nature12340"
            absorbers.append((layer, label, family, doi))

    return _cross_stacks(
        absorbers,
        etls,
        htls,
        record_type="full_stack_expansion",
        combo_mode="curated_corner_etl_htl",
        gap_method="literature_Eg_chi_expansion",
        functional="literature_chi_Eg",
        material_class="bulk_thin_film_device",
    )


def build_cdte_training_stacks() -> tuple[list[dict], list[dict]]:
    """CdTe paper stacks for Type-ML corner cases only (inference still blocks CdTe).

    Uses SnO2/CdS from the paper device family plus deep-χ / standard HTLs so the
    classifier sees extreme offsets outside the perovskite-only absorber set.
    """
    if not CDTE_RAW.exists():
        return [], []
    abs_l, paper_etls, paper_htls, doi = load_scaps(CDTE_RAW)
    assert abs_l is not None and abs_l.name == "CdTe"
    pooled_etls, pooled_htls = _load_pooled_scaps_contacts()
    etls = dict(paper_etls)
    # Keep paper CdS/SnO2; optionally add MgO as wide-gap corner
    if "MgO" in pooled_etls:
        etls["MgO"] = pooled_etls["MgO"]
    htls = dict(paper_htls)
    for name in ("MoO3", "V2O5", "NiO", "CuI", "Spiro-OMeTAD"):
        if name in pooled_htls:
            htls[name] = pooled_htls[name]
        elif name in NAMED_CONTACT_BANDS:
            eg, chi = NAMED_CONTACT_BANDS[name]
            htls[name] = Layer(name, eg, chi)

    absorbers = [
        (
            abs_l,
            "CdTe paper (pratyush) training-only corner stacks",
            "CdTe_chalcogenide_training_only",
            doi or "10.1002/pssb.70269",
        )
    ]
    return _cross_stacks(
        absorbers,
        etls,
        htls,
        record_type="literature_corner_case_non_perovskite",
        combo_mode="cdte_paper_plus_corner_htl",
        gap_method="SCAPS_literature_table",
        functional="literature_chi_Eg",
        material_class="non_perovskite_thin_film_training",
    )


def build_stacks() -> tuple[list[dict], list[dict], dict]:
    """Build ~1000 labeled stacks: SCAPS pool + expansion absorbers + CdTe training."""
    rows_a, fail_a = build_scaps_pooled_stacks()
    rows_b, fail_b = build_expansion_absorber_stacks()
    rows_c, fail_c = build_cdte_training_stacks()

    # Dedup by (absorber, etl, htl) — prefer SCAPS pooled, then expansion, then CdTe
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    for batch in (rows_a, rows_b, rows_c):
        for r in batch:
            key = (
                str(r["material_absorber"]),
                str(r["material_etl"]),
                str(r["material_htl"]),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    failures = fail_a + fail_b + fail_c
    stats = {
        "n_scaps_pooled": len(rows_a),
        "n_expansion": len(rows_b),
        "n_cdte_training": len(rows_c),
        "n_stacks_deduped": len(rows),
    }
    return rows, failures, stats


def build_paper5_absorbers() -> list[dict]:
    doi = "10.1038/s41524-019-0177-0"
    paper = "Li et al. npj Comput Mater 5, 83 (2019) — DFT indirect gap"
    rows: list[dict] = []
    with (RAW / "paper5_double_perovskites.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            eg = float(r["ind_gap"])
            if eg < 0.05:  # drop near-metallic / numerical-noise gaps
                continue
            phase = "cubic" if str(r["cubic"]) in ("1", "1.0") else "orthorhombic"
            formula = f"{r['a_atom']}2{r['b1_atom']}{r['b2_atom']}{r['x_atom']}6"
            rows.append(
                {
                    "material_absorber": f"{formula} ({phase})",
                    "absorber_band_gap_eV": eg,
                    "phase": phase,
                    "a_site": r["a_atom"],
                    "a2_site": "",
                    "b1_site": r["b1_atom"],
                    "b2_site": r["b2_atom"],
                    "x_site": r["x_atom"],
                    "heat_of_formation_eV_per_atom": float(r["heat_of_formation"]),
                    "gap_method": "DFT_indirect_gap",
                    "functional": "DFT_Paper5",
                    "gap_type": "fundamental_quasiparticle",
                    "material_class": "halide_double_perovskite_absorber",
                    "perovskite_family": "A2B1B3X6_halide_double_perovskite",
                    "record_type": "absorber_only",
                    "source_doi": doi,
                    "source_paper": paper,
                }
            )
    return rows


def build_pilania_absorbers() -> list[dict]:
    """Load Pilania oxide double perovskite gaps via matminer; cache to raw/."""
    import pandas as pd

    if PILANIA_RAW.exists():
        df = pd.read_csv(PILANIA_RAW)
    else:
        from matminer.datasets import load_dataset

        df = load_dataset("double_perovskites_gap")
        df.to_csv(PILANIA_RAW, index=False)

    doi = "10.1038/srep19375"
    paper = "Pilania et al. Sci Rep 6, 19375 (2016) — GLLB-SC double perovskite gaps"
    rows: list[dict] = []
    for _, r in df.iterrows():
        eg = float(r["gap gllbsc"])
        if eg <= 0:
            continue
        rows.append(
            {
                "material_absorber": str(r["formula"]),
                "absorber_band_gap_eV": eg,
                "phase": "",
                "a_site": str(r["a_1"]),
                "a2_site": str(r["a_2"]),
                "b1_site": str(r["b_1"]),
                "b2_site": str(r["b_2"]),
                "x_site": "O",
                "heat_of_formation_eV_per_atom": "",
                "gap_method": "DFT_GLLBSC_gap",
                "functional": "GLLB-SC",
                "gap_type": "fundamental_quasiparticle",
                "material_class": "oxide_double_perovskite_absorber",
                "perovskite_family": "A2BBprimeO6_oxide_double_perovskite",
                "record_type": "absorber_only",
                "source_doi": doi,
                "source_paper": paper,
            }
        )
    return rows


def _absorber_merge_key(name: str) -> str:
    """Dedup key: full material label (keeps cubic/orthorhombic twins)."""
    return (name or "").strip().lower()


def build_verified_lead_halide_absorbers() -> list[dict]:
    """Literature-cited ABX3 lead halides (external holdout coverage)."""
    if not VERIFIED_LEAD_HALIDE_RAW.exists():
        return []
    rows: list[dict] = []
    with VERIFIED_LEAD_HALIDE_RAW.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            material = (r.get("material") or "").strip()
            eg = r.get("absorber_band_gap_eV")
            if not material or not eg:
                continue
            rows.append(
                {
                    "material_absorber": material,
                    "absorber_band_gap_eV": float(eg),
                    "phase": r.get("phase") or "",
                    "a_site": r.get("a_site") or "",
                    "a2_site": "",
                    "b1_site": r.get("b1_site") or "Pb",
                    "b2_site": "",
                    "x_site": r.get("x_site") or "",
                    "heat_of_formation_eV_per_atom": "",
                    "gap_method": r.get("gap_method") or "literature_verified",
                    "functional": r.get("functional") or "",
                    "gap_type": r.get("gap_type") or "fundamental_optical",
                    "material_class": r.get("material_class")
                    or "lead_halide_perovskite_absorber",
                    "perovskite_family": r.get("perovskite_family")
                    or "ABX3_lead_halide_perovskite",
                    "record_type": r.get("record_type") or "verified_external",
                    "source_doi": r.get("source_doi") or "",
                    "source_paper": r.get("source_paper") or "",
                }
            )
    return rows


def merge_absorber_libraries(*sources: list[dict]) -> list[dict]:
    """Merge absorber rows; prefer verified_external over absorber_only on name clash."""
    priority = {"verified_external": 3, "absorber_only": 2, "full_stack": 1}
    by_key: dict[str, dict] = {}
    for rows in sources:
        for r in rows:
            key = _absorber_merge_key(r["material_absorber"])
            prev = by_key.get(key)
            if prev is None or priority.get(r.get("record_type"), 0) > priority.get(
                prev.get("record_type"), 0
            ):
                by_key[key] = r
    return list(by_key.values())


def build_k2gei6_absorber() -> list[dict]:
    path = RAW / "paper_k2gei6_dft_absorber.csv"
    with path.open(encoding="utf-8") as f:
        r = next(csv.DictReader(f))
    return [
        {
            "material_absorber": r["material"],
            "absorber_band_gap_eV": float(r["Eg_eV"]),
            "phase": "",
            "a_site": "K",
            "a2_site": "K",
            "b1_site": "Ge",
            "b2_site": "",
            "x_site": "I",
            "heat_of_formation_eV_per_atom": "",
            "gap_method": r["gap_method"],
            "functional": "DFT_paper_kaushiki",
            "gap_type": r["gap_type"],
            "material_class": "vacancy_ordered_halide_double_perovskite",
            "perovskite_family": "A2BX6_halide_double_perovskite",
            "record_type": "absorber_only",
            "source_doi": r["source_doi"],
            "source_paper": "Raj et al. Discover Electronics 2026 — K2GeI6 DFT Eg",
        }
    ]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def main() -> None:
    stacks, failures, stack_stats = build_stacks()
    paper5 = build_paper5_absorbers()
    pilania = build_pilania_absorbers()
    k2gei6 = build_k2gei6_absorber()
    verified_lead = build_verified_lead_halide_absorbers()
    absorbers = merge_absorber_libraries(paper5, pilania, k2gei6, verified_lead)

    write_csv(STACK_OUT, stacks, STACK_COLUMNS)
    write_csv(ABSORBER_OUT, absorbers, ABSORBER_COLUMNS)

    from collections import Counter

    etl_types = Counter(str(r["absorber_etl_type"]) for r in stacks)
    htl_types = Counter(str(r["absorber_htl_type"]) for r in stacks)
    families = Counter(str(r["perovskite_family"]) for r in stacks)
    n_cdte = sum(1 for r in stacks if str(r["material_absorber"]) == "CdTe")

    meta = {
        "note": (
            "Perovskite stack table ~1000 rows with Type I/II/III corner contacts; "
            "CdTe paper stacks are training-only (inference still blocks CdTe absorber). "
            "Does not modify opto_literature_dataset.csv"
        ),
        "stack_dataset": str(STACK_OUT),
        "absorber_library": str(ABSORBER_OUT),
        "pilania_raw_cache": str(PILANIA_RAW),
        "n_stacks": len(stacks),
        "n_absorbers": len(absorbers),
        "n_paper5_absorbers": len(paper5),
        "n_pilania_absorbers": len(pilania),
        "n_k2gei6_absorbers": len(k2gei6),
        "n_verified_lead_halide_absorbers": len(verified_lead),
        "n_absorbers_after_merge": len(absorbers),
        "n_cdte_training_stacks": n_cdte,
        "type_distribution": {
            "absorber_etl_type": dict(etl_types),
            "absorber_htl_type": dict(htl_types),
        },
        "stack_family_counts": dict(families),
        "stack_build_parts": stack_stats,
        "stack_internal_verify_failures": len(failures),
        "stack_all_rows_verified": len(failures) == 0,
        "stack_expansion": {
            "mode": "pooled_etl_htl + curated_corner + cdte_training",
            "note": (
                "SCAPS perovskite absorbers × pooled contacts (physical MoO3/V2O5 χ; "
                "wide-gap MgO/Al2O3). Extra ABX3/A2BX6 absorbers × curated grid. "
                "CdTe paper (10.1002/pssb.70269) stacks tagged training-only; "
                "χ used only to label Type; ML predicts Type from names+Eg."
            ),
            "cdte_policy": (
                "Included in Type-training with perovskite_family="
                "CdTe_chalcogenide_training_only / "
                "record_type=literature_corner_case_non_perovskite. "
                "Product absorber gate still rejects CdTe at inference."
            ),
        },
        "sources": [
            {
                "name": "K2TiI6 × pooled contacts",
                "doi": "10.1038/s41598-025-98351-y",
                "type": "full_stack_expanded",
            },
            {
                "name": "CsPbZnIBr2 × pooled contacts",
                "doi": "10.1038/s41598-024-81797-x",
                "type": "full_stack_expanded",
            },
            {
                "name": "Cs3Sb2Br9 × pooled contacts",
                "doi": "10.1016/j.nxmate.2026.102491",
                "type": "full_stack_expanded",
            },
            {
                "name": "Expansion absorbers × curated corner ETL×HTL",
                "doi": "multiple (see verified_expansion_absorbers.csv)",
                "rows": stack_stats.get("n_expansion", 0),
                "type": "full_stack_expansion",
            },
            {
                "name": "CdTe paper training-only corner stacks",
                "doi": "10.1002/pssb.70269",
                "rows": stack_stats.get("n_cdte_training", 0),
                "type": "literature_corner_case_non_perovskite",
            },
            {
                "name": "Paper5 halide double perovskites",
                "doi": "10.1038/s41524-019-0177-0",
                "rows": len(paper5),
                "type": "absorber_only",
            },
            {
                "name": "Pilania oxide double perovskites",
                "doi": "10.1038/srep19375",
                "rows": len(pilania),
                "type": "absorber_only",
            },
            {
                "name": "K2GeI6 DFT (Raj/kaushiki)",
                "doi": "10.1007/s44291-026-00245-4",
                "rows": len(k2gei6),
                "type": "absorber_only",
            },
            {
                "name": "Verified lead-halide ABX3 perovskites",
                "doi": "multiple (see verified_lead_halide_perovskites.csv)",
                "rows": len(verified_lead),
                "type": "verified_external",
            },
        ],
        "excluded_from_old_master": [
            "BeSiP2 SCAPS (not perovskite)",
            "Paper1/Ozcelik 2D monolayers (not perovskite)",
            "Paper2 MPS3 (not perovskite)",
        ],
        "excluded_research_paper_folder": [
            "tilt ppr.pdf Ca3NI3 (FAIL: CdS Eg=1.64; Ca3NI3 Eg=2.20 vs lit)",
            "DSSC.pdf (not perovskite)",
            "GaInP solar cells.pdf (not perovskite)",
        ],
        "cdte_paper_note": (
            "pratyush.pdf (DOI 10.1002/pssb.70269) PDF was removed from research paper/ "
            "earlier; Eg/χ recovered from prior verification extract "
            "(CdTe Eg=1.547 χ=3.9; SnO2/CdS/CdTe family) into "
            "data/raw/paper_cdte_scaps_materials.csv for Type-training only."
        ),
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if failures:
        print("STACK VERIFY FAILURES:", failures[:3])


if __name__ == "__main__":
    main()
