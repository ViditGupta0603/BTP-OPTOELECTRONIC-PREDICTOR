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

Excluded after verification (see data/research_paper_verification_report.json):
  Ca3NI3 bifacial paper (CdS Eg + Ca3NI3 Eg fail literature), DSSC, GaInP, CdTe.

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

RAW = ROOT / "data" / "raw"
DATA = ROOT / "data"

STACK_OUT = DATA / "perovskite_stack_dataset.csv"
ABSORBER_OUT = DATA / "perovskite_absorber_library.csv"
PILANIA_RAW = RAW / "pilania_double_perovskites_gap.csv"
META_OUT = DATA / "perovskite_dataset_build_meta.json"

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


def build_stacks() -> tuple[list[dict], list[dict]]:
    """Expand stacks: each perovskite absorber × pooled ETL × pooled HTL.

    χ is used only to *label* Type/CBO/VBO (physics). The expanded set trains
    Type-ML on material names + Eg so inference can skip χ for combos in-set.
    """
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
    # Contact pool: perovskite papers + BeSiP2 contacts only (not BeSiP2 absorber)
    contact_files = [
        RAW / "paper4_scaps_materials.csv",
        RAW / "paper_cs_pb_scaps_materials.csv",
        RAW / "paper_cs3sb2br9_scaps_materials.csv",
        RAW / "paper_besip2_scaps_materials.csv",
    ]

    absorbers: list[tuple[Layer, str, str, str]] = []
    for path, label, family, expect_doi in absorber_specs:
        abs_l, _, _, doi = load_scaps(path)
        assert abs_l is not None, path
        assert doi == expect_doi, (doi, expect_doi)
        absorbers.append((abs_l, label, family, doi))

    etls: dict[str, Layer] = {}
    htls: dict[str, Layer] = {}
    for path in contact_files:
        _, e, h, _ = load_scaps(path)
        _merge_layers(etls, e)
        _merge_layers(htls, h)

    rows: list[dict] = []
    failures: list[dict] = []
    for absorber, label, family, doi in absorbers:
        meta = {
            "gap_method": "SCAPS_literature_table",
            "functional": "literature_chi_Eg",
            "gap_type": "device_simulation_parameter",
            "material_class": "bulk_thin_film_device",
            "perovskite_family": family,
            "record_type": "full_stack",
            "combo_mode": "pooled_etl_htl",
        }
        for etl in etls.values():
            for htl in htls.values():
                row = stack_row(absorber, etl, htl, doi, label, **meta)
                msgs = verify_stack(absorber, etl, htl, row, tol=0.06)
                if msgs:
                    failures.append({"row": row, "msgs": msgs})
                rows.append(row)
    return rows, failures


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
    from matminer.datasets import load_dataset

    df = load_dataset("double_perovskites_gap")
    # persist local copy
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
    stacks, failures = build_stacks()
    paper5 = build_paper5_absorbers()
    pilania = build_pilania_absorbers()
    k2gei6 = build_k2gei6_absorber()
    verified_lead = build_verified_lead_halide_absorbers()
    absorbers = merge_absorber_libraries(paper5, pilania, k2gei6, verified_lead)

    write_csv(STACK_OUT, stacks, STACK_COLUMNS)
    write_csv(ABSORBER_OUT, absorbers, ABSORBER_COLUMNS)

    meta = {
        "note": "NEW perovskite-only dataset; does not modify opto_literature_dataset.csv",
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
        "stack_internal_verify_failures": len(failures),
        "stack_all_rows_verified": len(failures) == 0,
        "stack_expansion": {
            "mode": "pooled_etl_htl",
            "note": (
                "Each perovskite absorber crossed with all ETL×HTL that have Eg+χ. "
                "χ used only to label Type; ML can then predict Type from names+Eg."
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
            "pratyush.pdf CdTe (not perovskite)",
        ],
    }
    META_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    if failures:
        print("STACK VERIFY FAILURES:", failures[:3])


if __name__ == "__main__":
    main()
