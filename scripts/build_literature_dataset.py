"""Build verified literature optoelectronic stack dataset from all curated sources."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from literature_bands import (
    Layer,
    cbo_absorber_etl,
    junction_type,
    stack_row,
    vbo_absorber_htl,
    verify_stack,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "opto_literature_dataset.csv"
REPORT = ROOT / "data" / "dataset_verification_report.json"

COLUMNS = [
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
    "source_doi",
    "source_paper",
]

# Provenance metadata per source — addresses heterogeneous-data review (PBE vs HSE, expt vs DFT)
SOURCE_META: dict[str, dict[str, str]] = {
    "10.1038/s41598-025-98351-y": {
        "gap_method": "SCAPS_literature_table",
        "functional": "literature_chi_Eg",
        "gap_type": "device_simulation_parameter",
        "material_class": "bulk_thin_film_device",
    },
    "10.1038/s41598-024-81797-x": {
        "gap_method": "SCAPS_literature_table",
        "functional": "literature_chi_Eg",
        "gap_type": "device_simulation_parameter",
        "material_class": "bulk_thin_film_device",
    },
    "10.1007/s11082-026-08738-y": {
        "gap_method": "SCAPS_literature_table",
        "functional": "literature_chi_Eg",
        "gap_type": "device_simulation_parameter",
        "material_class": "bulk_thin_film_device",
    },
    "10.1038/s41699-021-00200-9": {
        "gap_method": "DFT_HSE06_monolayer",
        "functional": "HSE06",
        "gap_type": "fundamental_quasiparticle",
        "material_class": "2D_monolayer_vdW",
    },
    "10.1103/PhysRevB.94.035125": {
        "gap_method": "DFT_HSE06_monolayer",
        "functional": "HSE06",
        "gap_type": "fundamental_quasiparticle",
        "material_class": "2D_monolayer_vdW",
    },
    "10.1038/s41699-025-00578-w": {
        "gap_method": "experimental_UPS_absorption",
        "functional": "experiment",
        "gap_type": "optical_absorption_edge",
        "material_class": "2D_exfoliated_MPS3",
    },
}

# 8 monolayers present in BOTH Paper1 and Özçelik with slightly different HSE06 values
OVERLAP_HSE_MONOLAYERS = {
    "1T-HfS2": (1.99, 2.04),
    "1T-HfSe2": (1.02, 1.1),
    "2H-MoS2": (2.15, 2.22),
    "2H-MoSe2": (1.87, 1.95),
    "2H-MoTe2": (1.47, 1.5),
    "2H-WS2": (2.09, 2.2),
    "2H-WSe2": (1.75, 1.88),
    "2H-WTe2": (1.21, 1.3),
}


def meta_for(doi: str) -> dict[str, str]:
    return dict(SOURCE_META[doi])


@dataclass
class SourceBundle:
    name: str
    doi: str
    label: str
    rows: list[dict[str, str | float]]


def load_material_csv(path: Path) -> tuple[Layer, dict[str, Layer], dict[str, Layer]]:
    absorber: Layer | None = None
    etls: dict[str, Layer] = {}
    htls: dict[str, Layer] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer = Layer(row["material"], float(row["Eg_eV"]), float(row["chi_eV"]))
            role = row["layer_role"]
            if role == "absorber":
                absorber = layer
            elif role == "etl":
                etls[layer.name] = layer
            elif role == "htl":
                htls[layer.name] = layer
    if absorber is None:
        raise ValueError(f"No absorber in {path}")
    return absorber, etls, htls


def scaps_stacks(path: Path, doi: str, label: str) -> list[dict[str, str | float]]:
    absorber, etls, htls = load_material_csv(path)
    m = meta_for(doi)
    rows: list[dict[str, str | float]] = []
    for etl in etls.values():
        for htl in htls.values():
            rows.append(stack_row(absorber, etl, htl, doi, label, **m))
    return rows


def paper1_stacks() -> list[dict[str, str | float]]:
    doi = "10.1038/s41699-021-00200-9"
    label = "Paper1 vdW heterobilayer (Supp Table 3 + Table 2 HSE06)"
    m = meta_for(doi)
    mono: dict[str, Layer] = {}
    with (RAW / "paper1_table2_monolayers.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mono[row["material"]] = Layer.from_vbm_eg(
                row["material"], float(row["ev_vbm_eV"]), float(row["eg_hse06_eV"])
            )

    rows: list[dict[str, str | float]] = []
    with (RAW / "paper1_table3_type2.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mla, mlb = row["ml_a"], row["ml_b"]
            if mlb in ("", "-") or mla not in mono or mlb not in mono:
                continue
            a, b = mono[mla], mono[mlb]
            if a.eg < 0.3 or b.eg < 0.3:
                continue  # drop near-metallic monolayers (e.g. PdTe2 Eg~0.09)
            # Bilayer convention: absorber=ML-A, ETL=ML-B (electron sink), HTL=ML-A (hole sink)
            rows.append(stack_row(a, b, a, doi, label, **m))
    return rows


def paper2_stacks() -> list[dict[str, str | float]]:
    doi = "10.1038/s41699-025-00578-w"
    label = "Paper2 MPS3 experimental UPS/XPS (Fig 2d)"
    m = meta_for(doi)
    layers: dict[str, Layer] = {}
    with (RAW / "paper2_mps3_experimental.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layers[row["material"]] = Layer.from_ionization_potential(
                row["material"], float(row["ionization_potential_eV"]), float(row["band_gap_eV"])
            )
    rows: list[dict[str, str | float]] = []
    names = list(layers.keys())
    for a_name in names:
        for b_name in names:
            if a_name == b_name:
                continue
            a, b = layers[a_name], layers[b_name]
            if junction_type(a, b) not in ("Type I", "Type II"):
                continue
            rows.append(stack_row(a, b, a, doi, label, **m))
    return rows


def ozcelik_stacks() -> list[dict[str, str | float]]:
    """Tier B: Özçelik PRB 2016 HSE06 VBM/CBM → Type I/II bilayer stacks."""
    doi = "10.1103/PhysRevB.94.035125"
    label = "Ozcelik PRB 2016 HSE06 monolayers (Table I)"
    m = meta_for(doi)
    mono: dict[str, Layer] = {}
    with (RAW / "ozcelik_prb2016_monolayers.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eg = float(row["eg_hse_eV"])
            if eg < 0.4:  # skip near-metallic / ambiguous gaps
                continue
            mono[row["material"]] = Layer.from_vbm_eg(
                row["material"], float(row["ev_vbm_hse_eV"]), eg
            )
    rows: list[dict[str, str | float]] = []
    names = list(mono.keys())
    for a_name in names:
        for b_name in names:
            if a_name == b_name:
                continue
            a, b = mono[a_name], mono[b_name]
            if junction_type(a, b) not in ("Type I", "Type II"):
                continue
            rows.append(stack_row(a, b, a, doi, label, **m))
    return rows


def write_heterogeneous_audit(all_rows: list[dict]) -> Path:
    """Document ML-relevant heterogeneity issues raised in external review."""
    audit_path = ROOT / "data" / "heterogeneous_data_audit.json"

    # name collisions across material_class
    cross_class: list[dict] = [
        {
            "names": ["MoS2", "2H-MoS2"],
            "issue": "bulk-like SCAPS MoS2 (Eg=1.29, CsPb) vs HSE monolayer 2H-MoS2 (Eg~2.15–2.22)",
            "action": "Different material_class — do not merge; filter by material_class for ML",
        },
        {
            "names": ["WS2", "2H-WS2"],
            "issue": "BeSiP2 SCAPS WS2 (Eg=1.8) vs HSE monolayer 2H-WS2 (Eg~2.09–2.2)",
            "action": "Different material_class — keep separate",
        },
        {
            "names": ["GaAs", "GaAs_b"],
            "issue": "bulk GaAs HTL (1.42 eV) vs buckled monolayer GaAs_b (1.88 eV)",
            "action": "Already distinct names; GaAs_b is 2D buckled phase not bulk",
        },
    ]

    overlap_detail = [
        {
            "material": mat,
            "ozcelik_HSE_eV": vals[0],
            "paper1_HSE_eV": vals[1],
            "delta_eV": round(abs(vals[0] - vals[1]), 3),
            "note": "Same functional (HSE06) but different DFT papers — rows stay source-separated",
        }
        for mat, vals in OVERLAP_HSE_MONOLAYERS.items()
    ]

    by_class = {}
    for r in all_rows:
        by_class.setdefault(r["material_class"], 0)
        by_class[r["material_class"]] += 1

    audit = {
        "review_response": "Addresses heterogeneous-data concerns before LightGBM training",
        "recommendations": [
            "Use gap_method / functional / gap_type / material_class columns to filter homogeneous subsets",
            "Do NOT train on full CSV mixing SCAPS device params + HSE monolayers + experimental MPS3 without encoding",
            "For HSE-only ML use data/opto_literature_dataset_hse06_only.csv (1675 rows)",
            "For device-stack ML use data/opto_literature_dataset_scaps_only.csv (96 rows)",
            "MoTe2: HSE fundamental ~1.47 eV is NOT the same as ARPES optical gap ~0.92 eV at K — documented, not swapped",
        ],
        "rows_by_material_class": by_class,
        "cross_class_name_collisions": cross_class,
        "paper1_ozcelik_hse_overlap": overlap_detail,
        "gap_type_mixing": {
            "device_simulation_parameter": by_class.get("bulk_thin_film_device", 0),
            "fundamental_quasiparticle": sum(
                1 for r in all_rows if r["gap_type"] == "fundamental_quasiparticle"
            ),
            "optical_absorption_edge": sum(
                1 for r in all_rows if r["gap_type"] == "optical_absorption_edge"
            ),
        },
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit_path


def write_homogeneous_subsets(all_rows: list[dict]) -> dict[str, str]:
    """Write ML-ready homogeneous subsets."""
    paths: dict[str, str] = {}
    subsets = {
        "opto_literature_dataset_scaps_only.csv": lambda r: r["material_class"] == "bulk_thin_film_device",
        "opto_literature_dataset_hse06_only.csv": lambda r: r["functional"] == "HSE06",
        "opto_literature_dataset_experimental_only.csv": lambda r: r["functional"] == "experiment",
    }
    for fname, pred in subsets.items():
        sub = [r for r in all_rows if pred(r)]
        p = ROOT / "data" / fname
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(sub)
        paths[fname] = str(p)
        paths[f"{fname}_rows"] = str(len(sub))
    return paths


def dedupe_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    seen: set[tuple] = set()
    out: list[dict[str, str | float]] = []
    for r in rows:
        key = (
            r["material_absorber"],
            r["material_etl"],
            r["material_htl"],
            r["source_doi"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_paper5_absorber_library() -> Path:
    """Export verified absorber-only rows (no ETL/HTL) from Paper 5 supplementary."""
    src = RAW / "paper5_double_perovskites.csv"
    out = ROOT / "data" / "paper5_absorber_library.csv"
    doi = "10.1038/s41524-019-0177-0"
    label = "Li et al. npj Comput Mater 5, 83 (2019) — DFT indirect gap"
    cols = ["material_absorber", "absorber_band_gap_eV", "phase", "heat_of_formation_eV_per_atom", "source_doi", "source_paper"]
    rows: list[dict[str, str | float]] = []
    with src.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gap = float(row["ind_gap"])
            if gap <= 0:
                continue
            parts = row["composition"].split(".")
            if len(parts) != 4:
                continue
            a, b1, b2, x = parts
            formula = f"{a}2{b1}{b2}{x}6"
            phase = "cubic" if row["cubic"] == "1" else "orthorhombic"
            rows.append(
                {
                    "material_absorber": f"{formula} ({phase})",
                    "absorber_band_gap_eV": round(gap, 4),
                    "phase": phase,
                    "heat_of_formation_eV_per_atom": round(float(row["heat_of_formation"]), 6),
                    "source_doi": doi,
                    "source_paper": label,
                }
            )
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out


def verify_row_from_sources(row: dict[str, str | float]) -> list[str]:
    doi = str(row["source_doi"])
    if doi == "10.1038/s41598-025-98351-y":
        path = RAW / "paper4_scaps_materials.csv"
        abs_l, etls, htls = load_material_csv(path)
        return verify_stack(abs_l, etls[str(row["material_etl"])], htls[str(row["material_htl"])], row)
    if doi == "10.1038/s41598-024-81797-x":
        path = RAW / "paper_cs_pb_scaps_materials.csv"
        abs_l, etls, htls = load_material_csv(path)
        return verify_stack(abs_l, etls[str(row["material_etl"])], htls[str(row["material_htl"])], row)
    if doi == "10.1007/s11082-026-08738-y":
        path = RAW / "paper_besip2_scaps_materials.csv"
        abs_l, etls, htls = load_material_csv(path)
        return verify_stack(abs_l, etls[str(row["material_etl"])], htls[str(row["material_htl"])], row)
    if doi == "10.1038/s41699-021-00200-9":
        mono: dict[str, Layer] = {}
        with (RAW / "paper1_table2_monolayers.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                mono[r["material"]] = Layer.from_vbm_eg(
                    r["material"], float(r["ev_vbm_eV"]), float(r["eg_hse06_eV"])
                )
        a, b, h = mono[str(row["material_absorber"])], mono[str(row["material_etl"])], mono[str(row["material_htl"])]
        return verify_stack(a, b, h, row)
    if doi == "10.1038/s41699-025-00578-w":
        layers: dict[str, Layer] = {}
        with (RAW / "paper2_mps3_experimental.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                layers[r["material"]] = Layer.from_ionization_potential(
                    r["material"], float(r["ionization_potential_eV"]), float(r["band_gap_eV"])
                )
        a, b, h = layers[str(row["material_absorber"])], layers[str(row["material_etl"])], layers[str(row["material_htl"])]
        return verify_stack(a, b, h, row)
    if doi == "10.1103/PhysRevB.94.035125":
        mono: dict[str, Layer] = {}
        with (RAW / "ozcelik_prb2016_monolayers.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                eg = float(r["eg_hse_eV"])
                if eg < 0.4:
                    continue
                mono[r["material"]] = Layer.from_vbm_eg(
                    r["material"], float(r["ev_vbm_hse_eV"]), eg
                )
        a, b, h = mono[str(row["material_absorber"])], mono[str(row["material_etl"])], mono[str(row["material_htl"])]
        return verify_stack(a, b, h, row)
    return ["unknown source"]


def main() -> None:
    bundles = [
        SourceBundle(
            "paper4_k2tii6",
            "10.1038/s41598-025-98351-y",
            "K2TiI6 SCAPS Tables 1-2",
            scaps_stacks(RAW / "paper4_scaps_materials.csv", "10.1038/s41598-025-98351-y", "K2TiI6 SCAPS Tables 1-2"),
        ),
        SourceBundle(
            "paper_cs_pb",
            "10.1038/s41598-024-81797-x",
            "CsPbZnIBr2 SCAPS Tables 1-2",
            scaps_stacks(
                RAW / "paper_cs_pb_scaps_materials.csv",
                "10.1038/s41598-024-81797-x",
                "CsPbZnIBr2 SCAPS Tables 1-2",
            ),
        ),
        SourceBundle(
            "tierA_besip2",
            "10.1007/s11082-026-08738-y",
            "BeSiP2 SCAPS Tables 1-2 (Tier A)",
            scaps_stacks(
                RAW / "paper_besip2_scaps_materials.csv",
                "10.1007/s11082-026-08738-y",
                "BeSiP2 SCAPS Tables 1-2 (Tier A)",
            ),
        ),
        SourceBundle(
            "paper1_vdw",
            "10.1038/s41699-021-00200-9",
            "vdW type-II bilayers",
            paper1_stacks(),
        ),
        SourceBundle(
            "paper2_mps3",
            "10.1038/s41699-025-00578-w",
            "MPS3 experimental heterojunctions",
            paper2_stacks(),
        ),
        SourceBundle(
            "tierB_ozcelik",
            "10.1103/PhysRevB.94.035125",
            "Ozcelik PRB 2016 HSE06 Type I/II",
            ozcelik_stacks(),
        ),
    ]

    all_rows = dedupe_rows([r for b in bundles for r in b.rows])

    failures: list[dict] = []
    for i, row in enumerate(all_rows):
        errs = verify_row_from_sources(row)
        if errs:
            failures.append(
                {
                    "row_index": i,
                    "combo": f"{row['material_absorber']}/{row['material_etl']}/{row['material_htl']}",
                    "doi": row["source_doi"],
                    "errors": errs,
                }
            )

    paper5_path = build_paper5_absorber_library()
    audit_path = write_heterogeneous_audit(all_rows)
    subset_paths = write_homogeneous_subsets(all_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    summary = {
        "output": str(OUT),
        "total_rows": len(all_rows),
        "columns": COLUMNS,
        "by_source": {b.name: len(b.rows) for b in bundles},
        "verification_failures": failures,
        "all_rows_verified": len(failures) == 0,
        "heterogeneous_data_audit": str(audit_path),
        "homogeneous_subsets": subset_paths,
        "paper5_absorber_library": str(paper5_path),
        "paper5_absorber_count": sum(1 for _ in open(paper5_path, encoding="utf-8")) - 1,
    }
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
