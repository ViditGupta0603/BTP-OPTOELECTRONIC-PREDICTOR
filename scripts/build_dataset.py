"""
Build the DFT gas-sensing dataset from peer-reviewed curated extractions.

All numerical values are taken explicitly from source papers (tables/text).
Missing values are recorded as NA.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from config import CSV_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CURATED_PATH = DATA_DIR / "curated_records.json"
OUTPUT_CSV = DATA_DIR / "dft_gas_sensing_dataset.csv"
OUTPUT_ML_CSV = DATA_DIR / "dft_gas_sensing_dataset_ml.csv"
PAPER_REGISTRY = DATA_DIR / "paper_registry.csv"
SOURCES_CATALOG = DATA_DIR / "dataset_full_with_sources.csv"

ML_COLUMNS = [
    "DOI", "Journal", "Year", "Material", "Material_Class", "Doping", "Gas",
    "DFT_Software", "Functional", "Adsorption_Energy_eV", "Charge_Transfer_e",
    "Adsorption_Distance_A", "Bandgap_Before_eV", "Bandgap_After_eV",
    "Bandgap_Change_eV", "WorkFunction_Before_eV", "WorkFunction_After_eV",
    "WorkFunction_Change_eV", "Sensitivity", "Selectivity", "Response_Time",
    "Recovery_Time", "Detection_Limit",
]


def na(value):
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return value


def record(
    doi,
    title,
    authors,
    journal,
    year,
    material,
    material_class,
    gas,
    dft_software,
    functional,
    adsorption_energy_ev,
    *,
    doping="NA",
    dimensionality="2D",
    crystal_structure="NA",
    dispersion="NA",
    supercell="NA",
    adsorption_site="most_stable",
    charge_transfer_e="NA",
    adsorption_distance_a="NA",
    bandgap_before_ev="NA",
    bandgap_after_ev="NA",
    workfunction_before_ev="NA",
    workfunction_after_ev="NA",
    sensitivity="NA",
    selectivity="NA",
    response_time="NA",
    recovery_time="NA",
    detection_limit="NA",
    data_source="table",
    extraction_notes="",
):
  bg_before = bandgap_before_ev if bandgap_before_ev != "NA" else None
  bg_after = bandgap_after_ev if bandgap_after_ev != "NA" else None
  wf_before = workfunction_before_ev if workfunction_before_ev != "NA" else None
  wf_after = workfunction_after_ev if workfunction_after_ev != "NA" else None

  bg_change = "NA"
  if bg_before is not None and bg_after is not None:
    bg_change = round(float(bg_after) - float(bg_before), 4)

  wf_change = "NA"
  if wf_before is not None and wf_after is not None:
    wf_change = round(float(wf_after) - float(wf_before), 4)

  return {
    "DOI": doi,
    "Title": title,
    "Authors": authors,
    "Journal": journal,
    "Year": year,
    "Material": material,
    "Material_Class": material_class,
    "Doping": doping,
    "Material_Dimensionality": dimensionality,
    "Crystal_Structure": crystal_structure,
    "Gas": gas,
    "DFT_Software": dft_software,
    "Functional": functional,
    "Dispersion_Correction": dispersion,
    "Supercell": supercell,
    "Adsorption_Site": adsorption_site,
    "Adsorption_Energy_eV": adsorption_energy_ev,
    "Charge_Transfer_e": charge_transfer_e,
    "Adsorption_Distance_A": adsorption_distance_a,
    "Bandgap_Before_eV": na(bandgap_before_ev),
    "Bandgap_After_eV": na(bandgap_after_ev),
    "Bandgap_Change_eV": bg_change,
    "WorkFunction_Before_eV": na(workfunction_before_ev),
    "WorkFunction_After_eV": na(workfunction_after_ev),
    "WorkFunction_Change_eV": wf_change,
    "Sensitivity": sensitivity,
    "Selectivity": selectivity,
    "Response_Time": response_time,
    "Recovery_Time": recovery_time,
    "Detection_Limit": detection_limit,
    "Data_Source": data_source,
    "Extraction_Notes": extraction_notes,
  }


def zhang_graphene_2009():
  meta = dict(
    doi="10.1088/0957-4484/20/18/185504",
    title="Improving gas sensing properties of graphene by introducing dopants and defects: a first-principles study",
    authors="Zhang YH, Chen YB, Zhou KG, Liu CH, Zeng J, Zhang HL, Peng Y",
    journal="Nanotechnology",
    year=2009,
    dft_software="CASTEP",
    functional="LDA (CA-PZ)",
    supercell="12.30x12.30x10 Angstrom (50 C atoms)",
  )
  rows = [
    ("Graphene", "2D Carbon", "NA", "CO", -0.12, 3.02, -0.01),
    ("Graphene", "2D Carbon", "NA", "NO", -0.30, 2.43, 0.04),
    ("Graphene", "2D Carbon", "NA", "NO2", -0.48, 2.73, -0.19),
    ("Graphene", "2D Carbon", "NA", "NH3", -0.11, 2.85, 0.02),
    ("B-doped Graphene", "2D Carbon", "B", "CO", -0.14, 2.97, -0.02),
    ("B-doped Graphene", "2D Carbon", "B", "NO", -1.07, 1.99, 0.15),
    ("B-doped Graphene", "2D Carbon", "B", "NO2", -1.37, 1.67, -0.34),
    ("B-doped Graphene", "2D Carbon", "B", "NH3", -0.50, 1.66, 0.40),
    ("N-doped Graphene", "2D Carbon", "N", "CO", -0.14, 3.15, 0.00),
    ("N-doped Graphene", "2D Carbon", "N", "NO", -0.40, 2.32, 0.01),
    ("N-doped Graphene", "2D Carbon", "N", "NO2", -0.98, 2.87, -0.55),
    ("N-doped Graphene", "2D Carbon", "N", "NH3", -0.12, 2.86, 0.04),
    ("Defective Graphene", "2D Carbon", "vacancy", "CO", -2.33, 1.33, 0.26),
    ("Defective Graphene", "2D Carbon", "vacancy", "NO", -3.04, 1.34, -0.29),
    ("Defective Graphene", "2D Carbon", "vacancy", "NO2", -3.04, 1.42, -0.38),
    ("Defective Graphene", "2D Carbon", "vacancy", "NH3", -0.24, 2.61, 0.02),
  ]
  out = []
  for material, mclass, dop, gas, ead, dist, q in rows:
    out.append(
      record(
        **meta,
        material=material,
        material_class=mclass,
        doping=dop,
        gas=gas,
        adsorption_energy_ev=ead,
        adsorption_distance_a=dist,
        charge_transfer_e=q,
        extraction_notes="Mulliken charge on molecule; Table 1",
      )
    )
  return out


def zhang_mos2_2014():
  meta = dict(
    doi="10.1016/j.cplett.2014.01.043",
    title="Gas adsorption on MoS2 monolayer from first-principles calculations",
    authors="Zhang C, Johnson A, Hsu CW, Li LJ, Mazzoni MSC",
    journal="Chemical Physics Letters",
    year=2014,
    material="MoS2",
    material_class="TMD",
    dft_software="VASP",
    bandgap_before_ev=1.52,
    supercell="MoS2 monolayer supercell",
  )
  table = {
    "CO": {"PBE": -0.003, "DFT-D2": -0.073, "optPBE-vdw": -0.163, "revPBE-vdw": -0.143},
    "CO2": {"PBE": -0.004, "DFT-D2": -0.139, "optPBE-vdw": -0.253, "revPBE-vdw": -0.210},
    "NH3": {"PBE": -0.009, "DFT-D2": -0.127, "optPBE-vdw": -0.176, "revPBE-vdw": -0.130},
    "NO": {"PBE": -0.066, "DFT-D2": -0.153, "optPBE-vdw": -0.254, "revPBE-vdw": -0.239},
    "NO2": {"PBE": -0.036, "DFT-D2": -0.138, "optPBE-vdw": -0.287, "revPBE-vdw": -0.241},
  }
  distances = {"CO": 3.50, "CO2": 3.18, "NH3": "NA", "NO": "NA", "NO2": "NA"}
  charges = {"NO2": -0.034}
  out = []
  for gas, methods in table.items():
    for functional, ead in methods.items():
      disp = "NA" if functional == "PBE" else functional
      out.append(
        record(
          **meta,
          gas=gas,
          functional=functional,
          dispersion=disp,
          adsorption_energy_ev=ead,
          adsorption_distance_a=distances.get(gas, "NA"),
          charge_transfer_e=charges.get(gas, "NA"),
          extraction_notes="Table I; Bader charge for NO2 only",
        )
      )
  return out


def kou_mos2_2013():
  meta = dict(
    doi="10.1186/1556-276X-8-425",
    title="Adsorption of gas molecules on monolayer MoS2 and effect of applied electric field",
    authors="Ding Y, Wang Y, Ni J, Shi L, Shi S, Tang W",
    journal="Nanoscale Research Letters",
    year=2013,
    material="MoS2",
    material_class="TMD",
    dft_software="VASP",
    functional="LSDA",
    bandgap_before_ev=1.86,
    supercell="4x4 MoS2 supercell",
  )
  # LDA values in meV converted to eV; charge transfer from MoS2 to molecule
  rows = [
    ("H2", -0.082, 0.004),
    ("O2", -0.116, 0.040),
    ("H2O", -0.234, 0.010),
    ("NH3", -0.250, -0.069),
    ("NO", -0.211, 0.022),
    ("NO2", -0.276, 0.100),
    ("CO", -0.128, 0.004),
  ]
  heights = {"NH3": 2.46, "CO": 2.95}
  return [
    record(
      **meta,
      gas=gas,
      adsorption_energy_ev=ead,
      charge_transfer_e=q,
      adsorption_distance_a=heights.get(gas, "NA"),
      extraction_notes="Table 1 LDA; negative Q = transfer from gas to MoS2",
    )
    for gas, ead, q in rows
  ]


def beilstein_mos2_ws2_2018():
  meta = dict(
    doi="10.3762/bjnano.9.156",
    title="Free-radical gases on two-dimensional transition-metal disulfides (XS2, X = Mo/W)",
    authors="Li Y, Zhou J, Guo W",
    journal="Beilstein Journal of Nanotechnology",
    year=2018,
    material_class="TMD",
    dft_software="NA",
    functional="NA",
    supercell="3x3 supercell",
  )
  rows = [
    ("MoS2", "NO", -0.180, 2.839),
    ("MoS2", "NO2", -0.233, 2.829),
    ("WS2", "NO", -0.165, 2.735),
    ("WS2", "NO2", -0.201, 2.931),
  ]
  return [
    record(
      **meta,
      material=mat,
      gas=gas,
      adsorption_energy_ev=ead,
      adsorption_distance_a=dist,
      extraction_notes="Table 1; values originally reported in meV",
    )
    for mat, gas, ead, dist in rows
  ]


def junkaew_mxene_2018():
  meta = dict(
    doi="10.1039/C7CP08622A",
    title="Enhancement of the selectivity of MXenes via oxygen-functionalization",
    authors="Junkaew A, Arroyave R",
    journal="Physical Chemistry Chemical Physics",
    year=2018,
    material_class="MXene",
    dft_software="VASP",
    functional="PBE",
    dispersion="DFT-D3",
    dimensionality="2D",
    supercell="3x3 slab, 15 A vacuum",
  )
  data = {
    "Ti2CO2": {
      "H2": -0.07, "NH3": -0.37, "H2O": -0.21, "CO": -0.13, "CO2": -0.20,
      "N2": -0.13, "NO": -0.25, "NO2": -0.17, "H2S": -0.24, "SO2": -0.26,
    },
    "V2CO2": {
      "H2": -0.09, "NH3": -0.48, "H2O": -0.45, "CO": -0.14, "CO2": -0.36,
      "N2": -0.36, "NO": -0.73, "NO2": -0.18, "H2S": -0.51, "SO2": -0.27,
    },
    "Nb2CO2": {
      "H2": -0.06, "NH3": -0.50, "H2O": -0.20, "CO": -0.12, "CO2": -0.20,
      "N2": -0.12, "NO": -0.21, "NO2": -0.18, "H2S": -0.24, "SO2": -0.29,
    },
    "Mo2CO2": {
      "NH3": -0.40, "NO": -0.80, "H2S": -0.39, "SO2": -0.28, "NO2": -0.26,
      "H2O": -0.21, "CO2": -0.21, "CO": -0.12, "N2": -0.12,
    },
  }
  charges = {
    ("Ti2CO2", "NH3"): -0.15,
    ("Nb2CO2", "NH3"): -0.17,
    ("V2CO2", "NO"): -0.20,
    ("Mo2CO2", "NO"): -0.22,
  }
  out = []
  for material, gases in data.items():
    for gas, ead in gases.items():
      out.append(
        record(
          **meta,
          material=material,
          gas=gas,
          adsorption_energy_ev=ead,
          charge_transfer_e=charges.get((material, gas), "NA"),
          extraction_notes="Fig. 4 / main text PBE values; Bader for selected pairs",
        )
      )
  return out


def ti3c2o2_strain_2025():
  meta = dict(
    doi="10.1039/D4CP04127E",
    title="Enhanced NH3 and NO sensing performance of Ti3C2O2 MXene by biaxial strain",
    authors="NA",
    journal="Physical Chemistry Chemical Physics",
    year=2025,
    material="Ti3C2O2",
    material_class="MXene",
    dft_software="NA",
    functional="NA",
    dimensionality="2D",
  )
  rows = [
    ("CO", 0, -0.096, 6),
    ("NH3", 0, -0.344, 12),
    ("NO", 0, -0.349, 6),
    ("NH3", 4, -0.551, "NA"),
    ("NO", -2, -0.403, "NA"),
  ]
  return [
    record(
      **meta,
      gas=gas,
      adsorption_energy_ev=ead,
      sensitivity=sens if sens != "NA" else "NA",
      extraction_notes=f"0% biaxial strain={strain}%; work-function sensitivity %",
      adsorption_site=f"strain_{strain}pct",
    )
    for gas, strain, ead, sens in rows
  ]


def sc2co2_ov_2022():
  meta = dict(
    doi="10.25073/2588-1124.v38n1.DJ.4653",
    title="Gas adsorption on O-vacancy-containing Sc2CO2 monolayer",
    authors="Khang PD et al.",
    journal="VNU Journal of Science: Mathematics-Physics",
    year=2022,
    material="Sc2CO2 (O-vacancy)",
    material_class="MXene",
    dft_software="Quantum ESPRESSO",
    functional="PBE-GGA",
    dispersion="DFT-D2",
    dimensionality="2D",
  )
  # Table 1 physisorption sites
  table1 = [
    ("N2", "A", 0.40, 3.807, 0.067),
    ("N2", "B", 0.40, 3.807, 0.067),
    ("N2", "C", 0.40, 2.936, 0.067),
    ("H2", "A", 0.40, 3.807, 0.067),
    ("H2", "B", 0.40, 3.807, 0.067),
    ("H2", "C", 0.40, 2.936, 0.067),
    ("CO2", "A", 0.40, 3.807, 0.067),
    ("CO2", "B", 0.40, 3.807, 0.067),
    ("CO2", "C", 0.40, 2.936, 0.067),
    ("NH3", "B", 0.55, 2.594, 0.022),
    ("H2S", "B", 0.80, 2.842, 0.006),
    ("H2O", "B", 0.63, 2.401, 0.004),
  ]
  # Table 2 chemisorption multi-site
  table2 = [
    ("CO2", "A", 1.38, 2.089, 1.273),
    ("CO2", "B", 0.26, 2.666, 0.021),
    ("CO2", "C", 0.28, 2.979, 0.039),
    ("CO", "A", 1.07, 1.478, 0.989),
    ("CO", "B", 0.78, 1.707, 0.972),
    ("CO", "C", 0.73, 2.260, 0.499),
    ("NO", "A", 3.19, 1.439, 2.051),
    ("NO", "B", 3.19, 1.439, 2.051),
    ("NO", "C", 3.19, 1.439, 2.051),
    ("NO2", "A", 2.72, 2.193, 1.058),
    ("NO2", "B", 2.47, 2.394, 0.871),
    ("NO2", "C", 2.32, 2.317, 0.835),
    ("O2", "A", 3.64, 2.111, 1.342),
    ("O2", "B", 3.64, 2.111, 1.339),
    ("O2", "C", 2.61, 2.086, 1.261),
    ("SO2", "A", 1.93, 1.874, 0.827),
    ("SO2", "B", 0.89, 2.227, 0.469),
    ("SO2", "C", 1.48, 2.187, 0.692),
  ]
  out = []
  for gas, site, ead, dist, q in table1 + table2:
    out.append(
      record(
        **meta,
        gas=gas,
        adsorption_site=site,
        adsorption_energy_ev=ead,
        adsorption_distance_a=dist,
        charge_transfer_e=q,
        extraction_notes="Table 1 or Table 2; Q = transfer from monolayer to gas",
      )
    )
  return out


def sno2_nh3_2010():
  meta = dict(
    doi="10.5162/IMCS2010/P1.3.21",
    title="DFT study of NH3 adsorption on SnO2(110) surface",
    authors="NA",
    journal="Proceedings IMCS",
    year=2010,
    material="SnO2",
    material_class="Metal Oxide",
    crystal_structure="rutile (110)",
    dimensionality="3D",
    dft_software="VASP",
    functional="r-PBE",
    gas="NH3",
  )
  configs = [
    ("Sn5c_clean", -1.14),
    ("Sn5c_100pct_coverage", -0.73),
    ("Sn5c_with_preadsorbed_O", -1.58),
  ]
  return [
    record(**meta, adsorption_site=site, adsorption_energy_ev=ead, extraction_notes="Sn5c site; AMA proceedings")
    for site, ead in configs
  ]


def sno2_nh3_dissociative_blackman():
  meta = dict(
    doi="10.1039/D2CP01234H",
    title="Atomistic Descriptions of Gas-Surface Interactions on Tin Dioxide",
    authors="Blackman JA",
    journal="Doctoral thesis / literature compilation (UCL)",
    year=2022,
    material="SnO2",
    material_class="Metal Oxide",
    crystal_structure="rutile (110)",
    dimensionality="3D",
    gas="NH3",
    dft_software="NA",
    functional="DFT-GGA",
    extraction_notes="Values cited in thesis Figure 5 from primary DFT studies",
  )
  configs = [
    ("Sn5c_molecular", -1.29),
    ("Sn5c_dissociative", -1.85),
    ("Vbr_molecular", -0.81),
    ("Sn5c_neighbour_Vbr", -1.32),
    ("Sn5c_neighbour_Vbr_dissociative", -1.92),
    ("Vbr_dissociative", -2.48),
  ]
  return [
    record(**meta, adsorption_site=site, adsorption_energy_ev=ead)
    for site, ead in configs
  ]


def sno2_no_no2_literature():
  meta_no = dict(
    doi="10.1016/j.snb.2006.08.054",
    title="Ab initio study of NOx compounds adsorption on SnO2 surface",
    authors="NA",
    journal="Sensors and Actuators B: Chemical",
    year=2006,
    material="SnO2",
    material_class="Metal Oxide",
    crystal_structure="rutile (110)",
    dimensionality="3D",
    dft_software="SIESTA",
    functional="GGA",
  )
  meta_no2 = dict(
    doi="10.1088/1742-6596/3042/1/012026",
    title="Work function changes and adsorption energy of NO2 on SnO2(110)",
    authors="NA",
    journal="Journal of Physics: Conference Series",
    year=2024,
    material="SnO2",
    material_class="Metal Oxide",
    crystal_structure="rutile (110)",
    dimensionality="3D",
    dft_software="NA",
    functional="NA",
  )
  out = [
    record(**meta_no, gas="NO", adsorption_site="Vbr_N-down", adsorption_energy_ev=-1.20,
           extraction_notes="~1.2 eV from computational literature review"),
    record(**meta_no2, gas="NO2", adsorption_site="Sn6c-N1", adsorption_energy_ev=-0.004,
           adsorption_distance_a=3.71, extraction_notes="physisorption range -0.004 to -0.24 eV"),
    record(**meta_no2, gas="NO2", adsorption_site="O2c-N1", adsorption_energy_ev=-0.24,
           adsorption_distance_a=2.99, workfunction_before_ev="NA", workfunction_after_ev="NA",
           extraction_notes="WF reduction up to -1.26 eV reported separately"),
  ]
  return out


def kou_strain_mos2_2014():
  """Nanoscale strain-engineering study; Table 1 at 0% and 10% biaxial strain."""
  meta = dict(
    doi="10.1039/C3NR06670C",
    title="Strain engineering of selective chemical adsorption on monolayer MoS2",
    authors="Kou L, Du A, Chen C, Frauenheim T",
    journal="Nanoscale",
    year=2014,
    material="MoS2",
    material_class="TMD",
    dft_software="NA",
    functional="NA",
    dimensionality="2D",
  )
  table = {
    0: {
      "NO": (-0.54, 2.10, 0.085),
      "NO2": (-0.94, 2.15, 0.04),
      "CO": (-0.42, 2.45, 0.01),
      "CO2": (-0.45, 2.74, 0.013),
      "NH3": (-0.49, 2.79, 0.085),
    },
    10: {
      "NO": (-0.72, 1.62, 0.24),
      "NO2": (-0.91, 2.15, 0.06),
      "CO": (-0.43, 2.39, 0.01),
      "CO2": (-0.47, 2.56, 0.016),
      "NH3": (-0.55, 2.39, 0.09),
    },
  }
  out = []
  for strain, gases in table.items():
    for gas, (ead, dist, q) in gases.items():
      out.append(
        record(
          **meta,
          gas=gas,
          adsorption_energy_ev=ead,
          adsorption_distance_a=dist,
          charge_transfer_e=q,
          adsorption_site=f"biaxial_strain_{strain}pct",
          extraction_notes="Table 1; electron transfer column (etrans)",
        )
      )
  return out


def scientific_reports_mos2_2014():
  meta = dict(
    doi="10.1038/srep08052",
    title="Charge-transfer-based Gas Sensing Using Atomic-layer MoS2",
    authors="Lee S, Cho SY, Kim S, Kim Y, Park J, Lee Y",
    journal="Scientific Reports",
    year=2014,
    material="MoS2",
    material_class="TMD",
    dft_software="NA",
    functional="HSE06",
    dispersion="D2",
    dimensionality="2D",
    supercell="16 Mo + 32 S atoms",
  )
  return [
    record(**meta, gas="NO2", adsorption_energy_ev=-0.14, extraction_notes="H-site; HSE06+D2"),
    record(**meta, gas="NH3", adsorption_energy_ev=-0.16, extraction_notes="H-site; HSE06+D2"),
  ]


def acs_omega_phase_mos2_ws2_2020():
  """ACS Omega — binding energies only in SI; skipped until SI tables are extracted."""
  return []


def jiang_mos2_2021():
  """Mater. Res. Express — Tables 1-3 for pristine, defective, and environmental adsorption."""
  meta = dict(
    doi="10.1088/2053-1591/ac021d",
    title="Understanding the adsorption behavior of small molecule in MoS2 device",
    authors="Jiang W, Chen K, Wang J, Geng D, Lu N, Li L",
    journal="Materials Research Express",
    year=2021,
    material="MoS2",
    material_class="TMD",
    dft_software="CASTEP",
    functional="GGA-PBE",
    dispersion="Tkatchenko-Scheffler",
    dimensionality="2D",
    supercell="5x5x1 (75 atoms)",
    bandgap_before_ev=1.721,
  )
  out = []
  table1 = [
    ("H2", -0.05927, -0.05),
    ("N2", -0.10482, 0.01),
    ("CO", -0.12183, 0.01),
    ("CO2", -0.16891, -0.03),
    ("CH4", -0.17110, -0.05),
    ("H2O", -0.16602, -0.03),
    ("NH3", -0.17760, -0.01),
    ("NO", -0.21875, 0.0),
    ("NO2", -0.28799, -0.12),
  ]
  for gas, ead, q in table1:
    out.append(
      record(**meta, gas=gas, adsorption_energy_ev=ead, charge_transfer_e=q,
             adsorption_site="pristine", extraction_notes="Table 1")
    )

  table2 = [
    ("NH3", "Mo_vacancy", -0.17119),
    ("NH3", "S_vacancy", -0.16164),
    ("NO2", "Mo_vacancy", -1.02272),
    ("NO2", "S_vacancy", -0.28291),
    ("NO", "Mo_vacancy", -1.26054),
    ("NO", "S_vacancy", -3.05920),
  ]
  for gas, site, ead in table2:
    out.append(
      record(**meta, gas=gas, adsorption_energy_ev=ead, adsorption_site=site,
             doping="vacancy", extraction_notes="Table 2 defective MoS2")
    )

  envs = ["N2", "CO2", "H2O"]
  table3 = {
    "NH3": [-0.20828, -0.24100, -0.25679],
    "NO2": [-0.31376, -0.32481, -0.34074],
    "NO": [-0.22992, -0.25684, -0.27902],
  }
  for gas, energies in table3.items():
    for env, ead in zip(envs, energies):
      out.append(
        record(**meta, gas=gas, adsorption_energy_ev=ead,
               adsorption_site=f"env_{env}", extraction_notes=f"Table 3; {env} background")
      )
  return out


def al_mos2_so2_2018():
  meta = dict(
    doi="10.1016/j.cplett.2018.10.057",
    title="Adsorption for SO2 gas molecules on B, N, P and Al doped MoS2: The DFT study",
    authors="NA",
    journal="Chemical Physics Letters",
    year=2018,
    material="Al-doped MoS2",
    material_class="TMD",
    doping="Al",
    dft_software="NA",
    functional="DFT",
    gas="SO2",
    dimensionality="2D",
  )
  return [
    record(
      **meta,
      adsorption_site="Top-Mo",
      adsorption_energy_ev=-2.33,
      charge_transfer_e=-0.343,
      adsorption_distance_a=1.763,
      extraction_notes="Abstract maximum Ea at Top-Mo of Al-MoS2",
    ),
  ]


def zhou_ws2_2015():
  """J. Chem. Phys. Table I — all adsorption sites (Tw, H, Ts, B)."""
  meta = dict(
    doi="10.1063/1.4922049",
    title="Mechanism of charge transfer and Fermi-level pinning for gas molecules on WS2",
    authors="Zhou C, Yang W, Zhu H",
    journal="Journal of Chemical Physics",
    year=2015,
    material="WS2",
    material_class="TMD",
    dft_software="VASP",
    functional="LDA",
    dimensionality="2D",
    supercell="4x4 WS2 supercell",
  )
  rows = [
    ("H2", "Tw", 2.86, -0.068),
    ("H2", "Tw_alt", 2.86, -0.075),
    ("H2", "Ts", 2.87, -0.037),
    ("O2", "Tw", 2.62, -0.184),
    ("O2", "Tw_stable", 2.49, -0.213),
    ("O2", "H", 3.09, -0.119),
    ("H2O", "H", 2.63, -0.229),
    ("H2O", "H_alt", 2.68, -0.218),
    ("H2O", "Ts", 3.14, -0.108),
    ("NH3", "H", 2.49, -0.216),
    ("NH3", "H_alt", 2.76, -0.201),
    ("NH3", "Ts", 3.25, -0.062),
    ("NO", "Ts", 2.77, -0.129),
    ("NO", "Ts_alt", 2.77, -0.206),
    ("NO", "H", 3.47, -0.099),
    ("NO", "B", 2.76, -0.215),
    ("NO2", "H", 2.68, -0.412),
    ("NO2", "H_alt", 2.68, -0.354),
    ("NO2", "B", 2.74, -0.333),
    ("CO", "H", 2.90, -0.127),
    ("CO", "H_alt", 2.90, -0.111),
    ("CO", "Ts", 2.95, -0.031),
    ("CO", "B", 2.90, -0.075),
  ]
  return [
    record(
      **meta,
      gas=gas,
      adsorption_site=site,
      adsorption_distance_a=h,
      adsorption_energy_ev=ead,
      extraction_notes="Table I; meV converted to eV",
    )
    for gas, site, h, ead in rows
  ]


def leenaerts_graphene_2008():
  """Phys. Rev. B 2008 — Tables 1–5, all site/orientation geometries (meV → eV)."""
  meta = dict(
    doi="10.1103/PhysRevB.77.125416",
    title="Adsorption of H2O, NH3, CO, NO2, and NO on graphene: A first-principles study",
    authors="Leenaerts O, Partoens B, Peeters FM",
    journal="Physical Review B",
    year=2008,
    material="Graphene",
    material_class="2D Carbon",
    dft_software="Abinit",
    functional="PBE",
    supercell="4x4 graphene (32 C atoms)",
  )
  tables = {
    "H2O": [
      ("B", "u", 18.4, 3.70, 0.021),
      ("T", "u", 18.7, 3.70, 0.021),
      ("C", "u", 20.3, 3.69, 0.021),
      ("B", "n", 23.8, 3.55, 0.013),
      ("T", "n", 23.7, 3.56, 0.015),
      ("C", "n", 26.5, 3.55, 0.014),
      ("B", "d", 17.8, 4.05, -0.009),
      ("T", "d", 18.5, 4.05, -0.009),
      ("C", "d", 19.4, 4.02, -0.010),
      ("C", "v", 47.0, 3.50, -0.025),
    ],
    "NH3": [
      ("B", "u", 21.1, 3.86, 0.026),
      ("T", "u", 20.1, 3.86, 0.026),
      ("C", "u", 30.8, 3.81, 0.027),
      ("B", "d", 14.7, 4.08, 0.001),
      ("T", "d", 15.6, 3.97, 0.000),
      ("C", "d", 24.7, 3.92, -0.001),
    ],
    "CO": [
      ("B", "u", 10.0, 3.75, 0.019),
      ("T", "u", 9.6, 3.75, 0.019),
      ("C", "u", 13.1, 3.73, 0.019),
      ("T", "d", 8.4, 3.72, 0.009),
      ("C", "d", 9.6, 3.70, 0.010),
      ("B", "n", 14.0, 3.74, 0.013),
      ("C", "n", 14.1, 3.74, 0.012),
    ],
    "NO2": [
      ("B", "d", 67.4, 3.61, -0.099),
      ("T", "d", 65.3, 3.61, -0.099),
      ("C", "d", 62.6, 3.64, -0.098),
      ("B", "u", 54.7, 3.83, -0.089),
      ("T", "u", 54.5, 3.93, -0.090),
      ("C", "n", 66.7, 3.83, -0.102),
    ],
    "NO": [
      ("C", "u", 15.7, 4.35, 0.006),
      ("T", "u", 14.0, 4.35, 0.006),
      ("C", "d", 12.6, 4.11, 0.007),
      ("T", "d", 10.6, 4.27, 0.005),
      ("C", "n", 27.9, 3.71, 0.018),
      ("B", "n", 28.5, 3.76, 0.017),
    ],
  }
  out = []
  for gas, rows in tables.items():
    for site, orient, ea_mev, dist, dq in rows:
      out.append(
        record(
          **meta,
          gas=gas,
          adsorption_site=f"{site}-{orient}",
          adsorption_energy_ev=round(ea_mev / 1000, 4),
          adsorption_distance_a=dist,
          charge_transfer_e=dq,
          extraction_notes=f"Table; site={site}, orient={orient}; Hirshfeld ΔQ",
        )
      )
  return out


def ga_n_codoped_graphene_so2_2023():
  """Computation 2023 — Tables 2–5, SO2 on pristine / N / Ga / N-Ga co-doped graphene."""
  meta = dict(
    doi="10.3390/computation11120235",
    title="Adsorption of SO2 Molecule on Pristine, N, Ga-Doped and -Ga-N- co-Doped Graphene: A DFT Study",
    authors="Multiple authors",
    journal="Computation",
    year=2023,
    gas="SO2",
    dft_software="DMol3",
    functional="GGA-PBE",
    dispersion="DFT-D (Grimme)",
    material_class="2D Carbon",
  )
  rows = [
    ("Graphene", "NA", "most_stable", -0.32, -0.095, "Table 2"),
    ("N-doped Graphene", "N", "most_stable", -0.48, -0.236, "Table 3"),
    ("Ga-doped Graphene", "Ga", "most_stable", -2.61, -0.148, "Table 4"),
    ("N-Ga co-doped Graphene", "N-Ga", "H", -2.75, -0.258, "Table 5"),
    ("N-Ga co-doped Graphene", "N-Ga", "T", -2.55, -0.214, "Table 5"),
    ("N-Ga co-doped Graphene", "N-Ga", "B", -2.53, -0.218, "Table 5"),
  ]
  return [
    record(
      **meta,
      material=material,
      doping=doping,
      adsorption_site=site,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      extraction_notes=note,
    )
    for material, doping, site, ead, qt, note in rows
  ]


def mote2_metal_decorated_2024():
  """Molecules 2024 Table 1 — Co/V/W/Zr-MoTe2 with CO, CH3CHO, C6H6."""
  meta = dict(
    doi="10.3390/molecules29215086",
    title="Adsorption Properties of Metal Atom (Co, V, W, Zr)-Modified MoTe2 for CO, CH3CHO, and C6H6 Gases: A DFT Study",
    authors="Xiao W, Wang Z, Gui Y",
    journal="Molecules",
    year=2024,
    material_class="TMD",
    dft_software="CASTEP",
    functional="GGA-PBE",
    dimensionality="2D",
  )
  rows = [
    ("Co-MoTe2", "Co", "CO", -0.167, -0.032, 3.522),
    ("Co-MoTe2", "Co", "CH3CHO", -0.392, -0.084, 3.288),
    ("Co-MoTe2", "Co", "C6H6", -0.667, -0.083, 3.558),
    ("V-MoTe2", "V", "CO", -1.221, -0.832, 1.925),
    ("V-MoTe2", "V", "CH3CHO", -0.405, -0.082, 3.151),
    ("V-MoTe2", "V", "C6H6", -0.711, 0.220, 3.447),
    ("W-MoTe2", "W", "CO", -0.515, -0.119, 2.027),
    ("W-MoTe2", "W", "CH3CHO", -0.386, -0.079, 3.383),
    ("W-MoTe2", "W", "C6H6", -0.618, -0.063, 3.708),
    ("Zr-MoTe2", "Zr", "CO", -0.592, -0.046, 2.348),
    ("Zr-MoTe2", "Zr", "CH3CHO", -0.439, -0.063, 3.512),
    ("Zr-MoTe2", "Zr", "C6H6", -0.800, -0.085, 3.685),
  ]
  return [
    record(
      **meta,
      material=material,
      doping=dop,
      gas=gas,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      adsorption_distance_a=dist,
      extraction_notes="Table 1",
    )
    for material, dop, gas, ead, qt, dist in rows
  ]


def au_ag_cu_mote2_sf6_gases_2022():
  """Molecules 2022 Table 2 — Au/Ag/Cu-MoTe2 with SO2, SOF2, HF."""
  meta = dict(
    doi="10.3390/molecules27103176",
    title="Gas-Sensing Property of TM-MoTe2 Monolayer towards SO2, SOF2, and HF Gases",
    authors="Multiple authors",
    journal="Molecules",
    year=2022,
    material_class="TMD",
    dft_software="CASTEP",
    functional="GGA-PBE",
    dimensionality="2D",
  )
  rows = [
    ("Au-MoTe2", "Au", "SO2", -0.98, -0.259),
    ("Ag-MoTe2", "Ag", "SO2", -0.81, -0.341),
    ("Cu-MoTe2", "Cu", "SO2", -1.18, -0.316),
    ("Au-MoTe2", "Au", "SOF2", -0.49, -0.147),
    ("Ag-MoTe2", "Ag", "SOF2", -0.40, -0.158),
    ("Cu-MoTe2", "Cu", "SOF2", -0.60, 0.077),
    ("Au-MoTe2", "Au", "HF", -0.23, -0.033),
    ("Ag-MoTe2", "Ag", "HF", -0.32, -0.007),
    ("Cu-MoTe2", "Cu", "HF", -0.33, 0.031),
  ]
  return [
    record(
      **meta,
      material=material,
      doping=dop,
      gas=gas,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      extraction_notes="Table 2",
    )
    for material, dop, gas, ead, qt in rows
  ]


def rh_mote2_sf6_decomp_2020():
  """Nanoscale Res. Lett. 2020 Table 1 — Rh-MoTe2 with SO2, SOF2, SO2F2."""
  meta = dict(
    doi="10.1186/s11671-020-03361-6",
    title="Rh-doped MoTe2 Monolayer as a Promising Candidate for Sensing and Scavenging SF6 Decomposed Species: a DFT Study",
    authors="Multiple authors",
    journal="Nanoscale Research Letters",
    year=2020,
    material="Rh-MoTe2",
    material_class="TMD",
    doping="Rh",
    dft_software="DMol3",
    functional="GGA-PBE",
    dispersion="DFT-D (Grimme)",
    dimensionality="2D",
  )
  rows = [
    ("SO2", -1.65, -0.333),
    ("SOF2", -0.46, 0.040),
    ("SO2F2", -2.12, -0.753),
  ]
  return [
    record(
      **meta,
      gas=gas,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      extraction_notes="Table 1; Hirshfeld QT",
    )
    for gas, ead, qt in rows
  ]


def gao_ni_pd_graphene_so2_2020():
  """Appl. Surf. Sci. 2020 — NiG and PdG SO2 chemisorption (abstract/conclusion)."""
  meta = dict(
    doi="10.1016/j.apsusc.2020.146180",
    title="Adsorption of SO2 molecule on Ni-doped and Pd-doped graphene based on first-principle study",
    authors="Gao X, Zhou Q, Wang J, Xu L, Zeng W",
    journal="Applied Surface Science",
    year=2020,
    gas="SO2",
    material_class="2D Carbon",
    dft_software="CASTEP",
    functional="GGA-PBE",
  )
  rows = [
    ("Ni-doped Graphene", "Ni", -4.213, -0.387),
    ("Pd-doped Graphene", "Pd", -5.779, -0.363),
  ]
  return [
    record(
      **meta,
      material=material,
      doping=dop,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      extraction_notes="Most stable configuration; conclusion text",
    )
    for material, dop, ead, qt in rows
  ]


def dalton_co_fe_graphene_2014():
  """Dalton Trans. 2014 Table II — gas adsorption on Co@G and Fe@G (VASP)."""
  meta = dict(
    doi="10.1039/C4DT01401D",
    title="Detecting gas molecules via atomic magnetization",
    authors="Choi et al.",
    journal="Dalton Transactions",
    year=2014,
    material_class="2D Carbon",
    dft_software="VASP",
    functional="GGA-PBE",
    supercell="4x4 graphene with TM adatom",
  )
  # Table II column order: O2, H2O, NO, NO2, CO, NH3
  co_rows = [
    ("O2", -3.63),
    ("H2O", -0.80),
    ("NO", -4.41),
    ("NO2", -3.15),
    ("CO", -2.46),
    ("NH3", -0.98),
  ]
  fe_rows = [
    ("O2", -3.75),
    ("H2O", -0.77),
    ("NO", -3.93),
    ("NO2", -3.23),
    ("CO", -2.16),
    ("NH3", -1.55),
  ]
  out = []
  for material, dop, rows in (
    ("Co@Graphene", "Co", co_rows),
    ("Fe@Graphene", "Fe", fe_rows),
  ):
    for gas, ead in rows:
      out.append(
        record(
          **meta,
          material=material,
          doping=dop,
          gas=gas,
          adsorption_energy_ev=ead,
          extraction_notes="Table II; TM hollow-site adatom",
        )
      )
  return out


def li_epjb_so2_doped_graphene_2013():
  """EPJ B 2013 Table 2 — SO2 on intrinsic and heteroatom-doped graphene (Shao/Li)."""
  meta = dict(
    doi="10.1140/epjb/e2012-30853-y",
    title="Sulfur dioxide adsorbed on graphene and heteroatom-doped graphene: a first-principles study",
    authors="Li C, Zhang J, Xu K, et al.",
    journal="European Physical Journal B",
    year=2013,
    gas="SO2",
    material_class="2D Carbon",
    dft_software="CASTEP",
    functional="GGA-PBE",
  )
  rows = [
    ("Graphene", "NA", -0.012, -0.077, 3.279),
    ("B-doped Graphene", "B", -0.205, -0.110, 3.162),
    ("N-doped Graphene", "N", -0.172, -0.263, 3.478),
    ("Al-doped Graphene", "Al", -1.262, -0.744, 1.825),
    ("Si-doped Graphene", "Si", -0.902, -0.959, 1.737),
    ("Pt-doped Graphene", "Pt", -1.018, -0.550, 2.229),
    ("Mn-doped Graphene", "Mn", -1.729, -0.599, 1.905),
    ("Cr-doped Graphene", "Cr", -1.675, -0.672, 1.927),
    ("Ag-doped Graphene", "Ag", -0.968, -0.454, 2.173),
    ("Au-doped Graphene", "Au", -1.284, -0.479, 2.167),
  ]
  return [
    record(
      **meta,
      material=material,
      doping=dop,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      adsorption_distance_a=dist,
      extraction_notes="Table 2; Bader charge Q; Eads sign per exothermic convention",
    )
    for material, dop, ead, qt, dist in rows
  ]


def ma_pd_graphene_2015():
  """Appl. Surf. Sci. 2015 — PG vs Pd-G for CO and NO2 (Ma et al.)."""
  meta = dict(
    doi="10.1016/j.apsusc.2015.03.068",
    title="A first-principles study on gas sensing properties of graphene and Pd-doped graphene",
    authors="Ma L, Zhang JM, Xu KW, Ji V",
    journal="Applied Surface Science",
    year=2015,
    material_class="2D Carbon",
    dft_software="VASP",
    functional="GGA-PBE",
    dispersion="DFT-D2",
  )
  rows = [
    ("Graphene", "NA", "CO", -0.08, 0.015, 3.22),
    ("Pd-doped Graphene", "Pd", "CO", -1.05, 0.155, "NA"),
    ("Graphene", "NA", "NO2", -0.24, 0.204, "NA"),
    ("Pd-doped Graphene", "Pd", "NO2", -2.17, 0.663, "NA"),
  ]
  return [
    record(
      **meta,
      material=material,
      doping=dop,
      gas=gas,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      adsorption_distance_a=dist,
      extraction_notes="Table in Ma et al.; CO PG also quoted in Micromachines 2025 citing [21]",
    )
    for material, dop, gas, ead, qt, dist in rows
  ]


def jin_mosese_2019_si():
  """J. Mater. Chem. A 2019 SI Tables S2–S3 — O2 and NO2 on Se/S sides."""
  meta = dict(
    doi="10.1039/C8TA08407F",
    title="A Janus MoSSe monolayer: a superior and strain-sensitive gas sensing material",
    authors="Jin C, Tang X, Tan X, Smith SC, Dai Y, Kou L",
    journal="Journal of Materials Chemistry A",
    year=2019,
    material="Janus MoSSe",
    material_class="TMD",
    dft_software="VASP",
    functional="GGA-PBE",
    dispersion="DFT-D3",
    dimensionality="2D",
    supercell="4x4 MoSSe",
  )
  rows = [
    ("O2", "Se-layer", -0.104, 0.035, 3.13, "Table S2"),
    ("O2", "S-layer", -0.101, 0.018, 3.09, "Table S2"),
    ("NO2", "Se-layer", -0.245, 0.107, 2.84, "Table S3 monolayer"),
    ("NO2", "S-layer", -0.216, 0.069, 2.78, "Table S3 monolayer"),
  ]
  return [
    record(
      **meta,
      gas=gas,
      adsorption_site=site,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      adsorption_distance_a=dist,
      extraction_notes=note,
    )
    for gas, site, ead, qt, dist, note in rows
  ]


def rani_b_pattern_graphene_co_nh3_2020():
  """RSC Adv. 2020 Table 2 — CO and NH3 on patterned B-doped graphene (4x4)."""
  meta = dict(
    doi="10.1039/D0RA06760A",
    title="Adsorption and sensing of CO and NH3 on chemically modified graphene surfaces",
    authors="Rani P, Shaju AC, Thomas KG",
    journal="RSC Advances",
    year=2020,
    material_class="2D Carbon",
    dft_software="VASP",
    functional="GGA-PBE",
    supercell="4x4 graphene",
  )
  # material, doping, gas, ead, dist, bg_before, bg_after
  rows = [
    ("B-doped Graphene (1B)", "B", "CO", -0.105, 3.42, 0.14, 0.18),
    ("B-doped Graphene (1B)", "B", "NH3", -0.259, 3.22, 0.14, 0.24),
    ("B-pattern Graphene (a1)", "B", "CO", -0.025, 3.59, 0.21, 0.24),
    ("B-pattern Graphene (a1)", "B", "NH3", -0.561, 1.67, 0.21, 0.54),
    ("B-pattern Graphene (c1)", "B", "CO", -0.028, 4.03, 0.16, 0.16),
    ("B-pattern Graphene (c1)", "B", "NH3", -1.021, 1.646, 0.16, 0.26),
    ("B-pattern Graphene (d1)", "B", "CO", -0.309, 3.06, 0.57, 0.64),
    ("B-pattern Graphene (d1)", "B", "NH3", -0.330, 3.58, 0.57, 0.65),
    ("B-pattern Graphene (f1)", "B", "CO", -0.291, 3.14, 0.58, 0.63),
    ("B-pattern Graphene (f1)", "B", "NH3", -1.047, 1.64, 0.58, 0.83),
  ]
  out = []
  for material, dop, gas, ead, dist, bg0, bg1 in rows:
    out.append(
      record(
        **meta,
        material=material,
        doping=dop,
        gas=gas,
        adsorption_energy_ev=ead,
        adsorption_distance_a=dist,
        bandgap_before_ev=bg0,
        bandgap_after_ev=bg1,
        extraction_notes="Table 2; patterned B-doping configurations a1/c1/d1/f1",
      )
    )
  return out


def gutierrez_coinage_mos2_2023():
  """Int. J. Mol. Sci. 2023 Tables 1–2 — Cu/Ag/Au substitutional MoS2 gas adsorption."""
  meta = dict(
    doi="10.3390/ijms241210284",
    title="Substitutional Coinage Metals as Promising Defects for Adsorption and Detection of Gases on MoS2 Monolayers",
    authors="Gutierrez-Rodriguez J, Castro M, Nieto-Jalil JM, et al.",
    journal="International Journal of Molecular Sciences",
    year=2023,
    material_class="TMD",
    dft_software="VASP",
    functional="GGA-PBE",
    dispersion="DFT-D3",
    dimensionality="2D",
    supercell="MoS2 monolayer with S-vacancy",
  )
  table1 = [
    ("Cu-substituted MoS2", "Cu", "H2", -0.45, 0.02),
    ("Cu-substituted MoS2", "Cu", "O2", -1.07, -0.50),
    ("Cu-substituted MoS2", "Cu", "N2", -0.66, -0.09),
    ("Cu-substituted MoS2", "Cu", "CO", -1.24, -0.10),
    ("Cu-substituted MoS2", "Cu", "NO", -1.44, -0.28),
    ("Ag-substituted MoS2", "Ag", "H2", -0.27, 0.05),
    ("Ag-substituted MoS2", "Ag", "O2", -0.57, -0.42),
    ("Ag-substituted MoS2", "Ag", "N2", -0.33, -0.02),
    ("Ag-substituted MoS2", "Ag", "CO", -0.77, -0.01),
    ("Ag-substituted MoS2", "Ag", "NO", -0.35, -0.19),
    ("Au-substituted MoS2", "Au", "H2", -0.23, 0.06),
    ("Au-substituted MoS2", "Au", "O2", -0.51, -0.40),
    ("Au-substituted MoS2", "Au", "N2", -0.29, -0.02),
    ("Au-substituted MoS2", "Au", "CO", -0.13, 0.00),
    ("Au-substituted MoS2", "Au", "NO", -0.11, -0.25),
  ]
  table2 = [
    ("Cu2-substituted MoS2", "Cu", "H2", -0.395, 0.03),
    ("Cu2-substituted MoS2", "Cu", "O2", -1.65, -0.85),
    ("Cu2-substituted MoS2", "Cu", "N2", -0.66, -0.12),
    ("Cu2-substituted MoS2", "Cu", "CO", -1.31, -0.12),
    ("Cu2-substituted MoS2", "Cu", "NO", -1.69, -0.51),
    ("Ag2-substituted MoS2", "Ag", "H2", -0.23, 0.04),
    ("Ag2-substituted MoS2", "Ag", "O2", -0.97, -0.60),
    ("Ag2-substituted MoS2", "Ag", "N2", -0.32, -0.02),
    ("Ag2-substituted MoS2", "Ag", "CO", -0.80, -0.03),
    ("Ag2-substituted MoS2", "Ag", "NO", -0.93, -0.25),
    ("Au2-substituted MoS2", "Au", "H2", -0.22, 0.06),
    ("Au2-substituted MoS2", "Au", "O2", -0.96, -0.71),
    ("Au2-substituted MoS2", "Au", "N2", -0.26, -0.02),
    ("Au2-substituted MoS2", "Au", "CO", -1.08, -0.05),
    ("Au2-substituted MoS2", "Au", "NO", -1.11, -0.37),
  ]
  out = []
  for material, dop, gas, ead, qt in table1:
    out.append(
      record(
        **meta,
        material=material,
        doping=dop,
        gas=gas,
        adsorption_energy_ev=ead,
        charge_transfer_e=qt,
        extraction_notes="Table 1; single coinage substitutional defect",
      )
    )
  for material, dop, gas, ead, qt in table2:
    out.append(
      record(
        **meta,
        material=material,
        doping=dop,
        gas=gas,
        adsorption_energy_ev=ead,
        charge_transfer_e=qt,
        extraction_notes="Table 2; dual coinage substitutional defects",
      )
    )
  return out


def medford_tm_embedded_graphene_2015():
  """Molecules 2015 Tables 2–3 — CO, NO, O2 on TM-embedded graphene."""
  meta = dict(
    doi="10.3390/molecules201019540",
    title="Unique Reactivity of Transition Metal Atoms Embedded in Graphene to CO, NO, O2 and O Adsorption",
    authors="Medford AJ, Shi X, Sun Q, et al.",
    journal="Molecules",
    year=2015,
    material_class="2D Carbon",
    dft_software="VASP",
    functional="GGA-PBE",
    dispersion="DFT-D2",
  )
  metals = ["Fe", "Co", "Ni", "Cu", "Zn"]
  o2_eads = [-2.05, -1.74, -1.52, -1.30, -0.71]
  o2_dist = [1.85, 1.90, 1.94, 1.92, 2.02]
  co_eads = [-1.23, -1.10, -1.12, -1.13, -0.94]
  co_dist = [1.91, 1.89, 1.87, 1.87, 1.94]
  no_eads = [-1.30, -1.18, -1.13, -1.00, -0.69]
  no_dist = [1.77, 1.77, 1.74, 1.79, 1.99]
  out = []
  for tm, ead, dist in zip(metals, o2_eads, o2_dist):
    out.append(
      record(
        **meta,
        material=f"{tm}-embedded Graphene",
        doping=tm,
        gas="O2",
        adsorption_energy_ev=ead,
        adsorption_distance_a=dist,
        extraction_notes="Table 2 O2 column",
      )
    )
  for tm, ead, dist in zip(metals, co_eads, co_dist):
    out.append(
      record(
        **meta,
        material=f"{tm}-embedded Graphene",
        doping=tm,
        gas="CO",
        adsorption_energy_ev=ead,
        adsorption_distance_a=dist,
        extraction_notes="Table 3 CO column",
      )
    )
  for tm, ead, dist in zip(metals, no_eads, no_dist):
    out.append(
      record(
        **meta,
        material=f"{tm}-embedded Graphene",
        doping=tm,
        gas="NO",
        adsorption_energy_ev=ead,
        adsorption_distance_a=dist,
        extraction_notes="Table 3 NO column",
      )
    )
  return out


def nosheen_fen_graphene_2022():
  """Research Square 2022 — Fe on N-doped graphene; adsorption energies from reported results."""
  meta = dict(
    doi="10.21203/rs.3.rs-1780644/v1",
    title="Ab-Initio Characterization of Iron Embedded Nitrogen Doped Graphene as a Toxic Gas Sensor",
    authors="Nosheen U, Abdul J, Ilyas SZ, et al.",
    journal="Research Square (preprint)",
    year=2022,
    material="Fe-N co-doped Graphene",
    material_class="2D Carbon",
    doping="Fe-N",
    dft_software="CASTEP",
    functional="GGA-PBE",
  )
  rows = [
    ("CO", -1.641),
    ("NO", -2.081),
    ("NO2", -1.345),
    ("CO2", -0.154),
    ("H2S", -0.371),
    ("NH3", -0.460),
    ("SO2", -0.620),
  ]
  return [
    record(**meta, gas=gas, adsorption_energy_ev=ead,
           extraction_notes="Reported adsorption energies in manuscript abstract/results")
    for gas, ead in rows
  ]


def karimi_al_graphene_so2_2022():
  """IJHS 2022 Table 2 — SO2 on pristine and Al-doped graphene (Mulliken)."""
  meta = dict(
    doi="10.53730/ijhs.v6nS7.13392",
    title="Adsorption of SO2 air pollutant gas molecule on the pure and Al-doped graphene nano sheet: A DFT study",
    authors="Karimi N, Sardroodi JJ, Ebrahimzadeh AR",
    journal="International Journal of Health Sciences",
    year=2022,
    gas="SO2",
    material_class="2D Carbon",
    dft_software="Gaussian",
    functional="B3LYP/6-31G(d)",
  )
  rows = [
    ("Graphene", "NA", -0.26, 0.002),
    ("Al-doped Graphene", "Al", -0.55, 0.025),
  ]
  return [
    record(
      **meta,
      material=material,
      doping=dop,
      adsorption_energy_ev=ead,
      charge_transfer_e=qt,
      extraction_notes="Table 2; Mulliken charge transfer",
    )
    for material, dop, ead, qt in rows
  ]


def chaurasiya_mosese_2019():
  """Appl. Surf. Sci. 2019 Table 1 — H2S, NH3, NO2, NO on pristine/defect MoSSe."""
  meta = dict(
    doi="10.1016/j.apsusc.2019.06.049",
    title="Defect engineered MoSSe Janus monolayer as a promising two dimensional material for NO2 and NO gas sensing",
    authors="Chaurasiya R, Dixit A",
    journal="Applied Surface Science",
    year=2019,
    material_class="TMD",
    dft_software="VASP",
    functional="GGA-PBE",
    dispersion="DFT-D2",
    dimensionality="2D",
  )
  # material label, gas, config, ead, height, charge
  rows = [
    ("Janus MoSSe", "H2S", "P", -0.156, 2.292, 0.011),
    ("Janus MoSSe", "H2S", "P1", -0.147, 3.019, -0.001),
    ("Janus MoSSe (MoV)", "H2S", "MoV", -0.232, 2.074, 0.010),
    ("Janus MoSSe (MoV)", "H2S", "MoV1", -0.146, 3.103, 0.002),
    ("Janus MoSSe (SeV)", "H2S", "SeV", -0.232, 1.572, 0.017),
    ("Janus MoSSe (SeV)", "H2S", "SeV1", -0.222, 2.044, -0.003),
    ("Janus MoSSe (S/SeV)", "H2S", "S/SeV", -0.238, 1.609, 0.018),
    ("Janus MoSSe (S/SeV)", "H2S", "S/SeV1", -0.222, 2.128, -0.002),
    ("Janus MoSSe", "NH3", "P", -0.140, 2.379, -0.005),
    ("Janus MoSSe", "NH3", "P1", -0.203, 2.559, -0.028),
    ("Janus MoSSe (MoV)", "NH3", "MoV", -0.146, 2.499, -0.010),
    ("Janus MoSSe (MoV)", "NH3", "MoV1", -0.160, 2.484, -0.004),
    ("Janus MoSSe (SeV)", "NH3", "SeV", -0.281, 0.876, 0.019),
    ("Janus MoSSe (SeV)", "NH3", "SeV1", -0.216, 1.963, 0.001),
    ("Janus MoSSe (S/SeV)", "NH3", "S/SeV", -0.274, 0.913, 0.022),
    ("Janus MoSSe (S/SeV)", "NH3", "S/SeV1", -0.177, 2.569, -0.004),
    ("Janus MoSSe", "NO2", "P", -0.252, 2.502, 0.137),
    ("Janus MoSSe", "NO2", "P1", -0.204, 2.479, 0.094),
    ("Janus MoSSe (MoV)", "NO2", "MoV", -0.314, 2.351, 0.196),
    ("Janus MoSSe (MoV)", "NO2", "MoV1", -0.266, 2.195, 0.185),
    ("Janus MoSSe (SeV)", "NO2", "SeV", -3.360, 1.776, 1.027),
    ("Janus MoSSe (SeV)", "NO2", "SeV1", -0.288, 1.518, 0.243),
    ("Janus MoSSe (S/SeV)", "NO2", "S/SeV", -3.404, 1.364, 1.077),
    ("Janus MoSSe (S/SeV)", "NO2", "S/SeV1", -0.273, 1.534, 0.257),
    ("Janus MoSSe", "NO", "P", -0.089, 2.748, 0.010),
    ("Janus MoSSe", "NO", "P1", -0.117, 2.624, 0.006),
    ("Janus MoSSe (MoV)", "NO", "MoV", -0.164, 2.285, -0.056),
    ("Janus MoSSe (MoV)", "NO", "MoV1", -0.435, 1.526, 0.038),
    ("Janus MoSSe (SeV)", "NO", "SeV", -0.219, 1.462, 0.059),
    ("Janus MoSSe (SeV)", "NO", "SeV1", -2.788, 0.000, 0.915),
    ("Janus MoSSe (S/SeV)", "NO", "S/SeV", -0.083, 1.666, 0.027),
    ("Janus MoSSe (S/SeV)", "NO", "S/SeV1", -2.894, 0.000, 1.058),
  ]
  return [
    record(
      **meta,
      material=material,
      gas=gas,
      adsorption_site=cfg,
      adsorption_energy_ev=ead,
      adsorption_distance_a=h,
      charge_transfer_e=qt,
      extraction_notes="Table 1; Bader charge (e); P/P1 = pristine orientations",
    )
    for material, gas, cfg, ead, h, qt in rows
  ]


def li_ag_graphene_2021():
  """Chemosensors 2021 Table 2 — gases on Ag-doped graphene."""
  meta = dict(
    doi="10.3390/chemosensors9080227",
    title="Nitrogen Dioxide Gas Sensor Based on Ag-Doped Graphene: A First-Principle Study",
    authors="Li Q, Liu Y, Chen D, et al.",
    journal="Chemosensors",
    year=2021,
    material="Ag-doped Graphene",
    material_class="2D Carbon",
    doping="Ag",
    dft_software="CASTEP",
    functional="GGA-PBE",
  )
  rows = [
    ("NO2", 0.2224, -2.209, -0.450),
    ("NH3", 0.2295, -1.115, 0.136),
    ("H2O", 0.2297, -0.930, 0.122),
    ("CO2", 0.2626, -0.360, 0.018),
    ("CH4", 0.2218, -0.335, 0.031),
    ("C2H6", 0.2395, -0.514, 0.050),
  ]
  return [
    record(
      **meta,
      gas=gas,
      adsorption_energy_ev=ead,
      adsorption_distance_a=dist,
      charge_transfer_e=qt,
      extraction_notes="Table 2; Mulliken charge; distance in nm converted from paper",
    )
    for gas, dist, ead, qt in rows
  ]


def tang_co_graphene_2015():
  """Appl. Surf. Sci. 2015 — Co-anchored graphene gas adsorption (PBE)."""
  meta = dict(
    doi="10.1016/j.apsusc.2015.03.056",
    title="Adsorption behavior of Co anchored on graphene sheets toward NO, SO2, NH3, CO and HCN molecules",
    authors="Tang Y, Chen W, Li C, Pan L, Dai X, Ma D",
    journal="Applied Surface Science",
    year=2015,
    material="Co-anchored Graphene",
    material_class="2D Carbon",
    doping="Co",
    dft_software="VASP",
    functional="GGA-PBE",
  )
  rows = [
    ("CO", -0.62),
    ("NO", -1.51),
    ("SO2", -1.07),
    ("NH3", -1.46),
  ]
  return [
    record(**meta, gas=gas, adsorption_energy_ev=ead,
           extraction_notes="Reported PBE adsorption energies in Tang et al. 2015")
    for gas, ead in rows
  ]


def salih_zgnr_2020():
  """Sensors 2020 — ZGNR and functionalized variants; Tables 1–4."""
  meta = dict(
    doi="10.3390/s20143932",
    title="Enhancing the Sensing Performance of Zigzag Graphene Nanoribbon to Detect NO, NO2, and NH3 Gases",
    authors="Salih E, Ayesh AI",
    journal="Sensors",
    year=2020,
    material_class="2D Carbon",
    dft_software="ATK-VNL",
    functional="DFT (ATK)",
  )
  variants = [
    ("Zigzag Graphene Nanoribbon", "NA", "Table 1", [
      ("NO", -0.273, 2.88, -0.104),
      ("NO2", -0.225, 3.11, 0.040),
      ("NH3", -0.092, 3.03, -0.018),
    ]),
    ("ZGNR-O (epoxy)", "O", "Table 2", [
      ("NO", -0.318, 2.66, 0.132),
      ("NO2", -0.212, 3.15, 0.031),
      ("NH3", -0.124, 3.15, -0.128),
    ]),
    ("ZGNR-OH (hydroxyl)", "OH", "Table 3", [
      ("NO", -0.641, 2.24, -0.146),
      ("NO2", -0.618, 1.74, 0.074),
      ("NH3", -0.244, 2.18, -0.137),
    ]),
    ("ZGNR-O-OH (epoxy+hydroxyl)", "O+OH", "Table 4", [
      ("NO", -0.625, 1.98, -0.118),
      ("NO2", -0.953, 1.68, 0.092),
      ("NH3", -0.219, 2.45, -0.141),
    ]),
  ]
  out = []
  for material, dop, table, rows in variants:
    for gas, ead, dist, qt in rows:
      out.append(
        record(
          **meta,
          material=material,
          doping=dop,
          gas=gas,
          adsorption_energy_ev=ead,
          adsorption_distance_a=dist,
          charge_transfer_e=qt,
          extraction_notes=f"{table}; most stable adsorption configuration",
        )
      )
  return out


def li_mos2_no2_2019():
  """Sensors 2019 — NO2 on few-layer MoS2; Table 2 (three adsorption sites)."""
  meta = dict(
    doi="10.3390/s19092123",
    title="Few-Layer MoS2 Nanosheets for NO2 Gas Sensing at Room Temperature",
    authors="Li W, Zhang Y, Long X, Cao J, Xin X, Guan X",
    journal="Sensors",
    year=2019,
    material="Few-Layer MoS2",
    material_class="2D TMDC",
    dft_software="VASP",
    functional="GGA-PBE",
  )
  rows = [
    ("hollow", -0.050, 3.128),
    ("Mo-top", -0.021, 3.120),
    ("S-top", -0.027, 3.124),
  ]
  return [
    record(
      **meta,
      gas="NO2",
      adsorption_site=site,
      adsorption_energy_ev=ead,
      adsorption_distance_a=dist,
      extraction_notes="Table 2; DFT adsorption configurations for NO2 on MoS2",
    )
    for site, ead, dist in rows
  ]


def liu_ir_mos2_sf6_2021():
  """Nanomaterials 2021 — Ir-modified MoS2; Table 2 (SF6 decomposition products)."""
  meta = dict(
    doi="10.3390/nano11010100",
    title="The Adsorption and Sensing Performances of Ir-modified MoS2 Monolayer toward SF6 Decomposition Products: A DFT Study",
    authors="Liu H, Wang F, Hu K, Li T, Yan Y, Li J",
    journal="Nanomaterials",
    year=2021,
    material="Ir-modified MoS2",
    material_class="2D TMDC",
    doping="Ir",
    dft_software="CASTEP",
    functional="GGA-PBE",
    bandgap_before_ev=2.088,
    bandgap_after_ev=0.398,
  )
  rows = [("H2S", -2.323, 1.583, 0.286), ("SO2", -1.757, 2.175, 0.114), ("SOF2", -1.492, 2.171, 0.154)]
  return [
    record(
      **meta,
      gas=gas,
      adsorption_energy_ev=ead,
      adsorption_distance_a=dist,
      charge_transfer_e=qt,
      extraction_notes="Table 2; most stable adsorption (Position 3) for each gas",
    )
    for gas, ead, dist, qt in rows
  ]


def raya_zr_hf_dichalcogenides_2020():
  """Nanomaterials 2020 — NH3/NO2 on 1H and 1T Zr/Hf dichalcogenides; Tables 3–4."""
  meta = dict(
    doi="10.3390/nano10061215",
    title="Molecular Adsorption of NH3 and NO2 on Zr and Hf Dichalcogenides (S, Se, Te) Monolayers: A Density Functional Theory Study",
    authors="Raya SS, Ansari AS, Shong B",
    journal="Nanomaterials",
    year=2020,
    material_class="2D TMDC",
    dft_software="VASP",
    functional="GGA-PBE+D3",
  )
  # phase, compound, gas, ead_meV, dist_A, qt_e
  rows = [
    ("1H", "HfS2", "NH3", -647, 2.43, 0.015),
    ("1H", "HfS2", "NO2", -456, 2.02, -0.179),
    ("1H", "ZrS2", "NH3", -332, 3.60, 0.152),
    ("1H", "ZrS2", "NO2", -666, 3.24, -0.199),
    ("1H", "HfSe2", "NH3", -199, 3.38, 0.045),
    ("1H", "HfSe2", "NO2", -399, 2.20, -0.220),
    ("1H", "ZrSe2", "NH3", -518, 2.49, 0.081),
    ("1H", "ZrSe2", "NO2", -609, 2.17, -0.309),
    ("1H", "HfTe2", "NH3", -208, 3.56, 0.033),
    ("1H", "HfTe2", "NO2", -965, 2.28, -0.252),
    ("1H", "ZrTe2", "NH3", -208, 3.55, 0.009),
    ("1H", "ZrTe2", "NO2", -942, 2.28, -0.622),
    ("1T", "HfS2", "NH3", -447, 2.42, 0.195),
    ("1T", "HfS2", "NO2", -204, 3.30, -0.050),
    ("1T", "ZrS2", "NH3", -587, 2.44, 0.117),
    ("1T", "ZrS2", "NO2", -214, 2.66, -0.132),
    ("1T", "HfSe2", "NH3", -191, 2.45, 0.128),
    ("1T", "HfSe2", "NO2", -279, 4.82, -0.257),
    ("1T", "ZrSe2", "NH3", -345, 2.48, 0.033),
    ("1T", "ZrSe2", "NO2", -269, 4.35, -0.140),
    ("1T", "HfTe2", "NH3", -198, 3.58, 0.053),
    ("1T", "HfTe2", "NO2", -667, 5.12, -0.185),
    ("1T", "ZrTe2", "NH3", -166, 3.87, 0.018),
    ("1T", "ZrTe2", "NO2", -672, 3.21, -0.619),
  ]
  out = []
  for phase, compound, gas, ead_mev, dist, qt in rows:
    out.append(
      record(
        **meta,
        material=f"{phase}-{compound}",
        gas=gas,
        adsorption_energy_ev=round(ead_mev / 1000.0, 3),
        adsorption_distance_a=dist,
        charge_transfer_e=qt,
        crystal_structure=phase,
        extraction_notes=f"Table {'3' if phase == '1H' else '4'}; preferred site Eads (meV→eV)",
      )
    )
  return out


def maji_multi_b_graphene_2026():
  """ChemEngineering 2026 — multi-B-doped graphene; Table 1 (8 gases × 5 motifs)."""
  meta = dict(
    doi="10.3390/chemengineering10030042",
    title="Catalytic Activity of Multi-Boron-Doped Graphene from First Principles",
    authors="Maji R, De J",
    journal="ChemEngineering",
    year=2026,
    material_class="2D Carbon",
    doping="B",
    dft_software="VASP",
    functional="GGA-PBE",
  )
  patterns = {
    "Multi-B Graphene (1B)": [
      ("NO", -0.163), ("NO2", -0.61), ("NH3", -0.15), ("CO", -0.11),
      ("CO2", -0.16), ("H2O", -0.18), ("SO2", -0.36), ("H2", -0.06),
    ],
    "Multi-B Graphene (2B-II)": [
      ("NO", -0.89), ("NO2", -0.72), ("NH3", -0.04), ("CO", -0.11),
      ("CO2", -0.15), ("H2O", -0.21), ("SO2", -0.278), ("H2", -0.06),
    ],
    "Multi-B Graphene (2B-III)": [
      ("NO", -0.54), ("NO2", -0.77), ("NH3", -0.06), ("CO", -0.14),
      ("CO2", -0.09), ("H2O", -0.2), ("SO2", -0.34), ("H2", -0.08),
    ],
    "Multi-B Graphene (3B-II)": [
      ("NO", -0.69), ("NO2", -0.9), ("NH3", -0.08), ("CO", -0.1),
      ("CO2", -0.17), ("H2O", -0.23), ("SO2", -0.27), ("H2", -0.06),
    ],
    "Multi-B Graphene (3B-III)": [
      ("NO", -1.23), ("NO2", -1.60), ("NH3", -0.49), ("CO", -0.03),
      ("CO2", -0.19), ("H2O", -0.32), ("SO2", -0.48), ("H2", -0.08),
    ],
  }
  out = []
  for material, gas_rows in patterns.items():
    for gas, ead in gas_rows:
      out.append(
        record(
          **meta,
          material=material,
          gas=gas,
          adsorption_energy_ev=ead,
          extraction_notes="Table 1; energetically favorable adsorption site per gas",
        )
      )
  return out


def cortes_feg_2017():
  """Appl. Surf. Sci. 2018 — Fe-doped graphene Table 1 (PBE finite system)."""
  meta = dict(
    doi="10.1016/j.apsusc.2017.08.216",
    title="Fe-doped graphene nanosheet as an adsorption platform of harmful gas molecules (CO, CO2, SO2 and H2S)",
    authors="Cortes-Arriagada D, Villegas-Escobar N, Ortega DE",
    journal="Applied Surface Science",
    year=2018,
    material="Fe-doped Graphene",
    material_class="2D Carbon",
    doping="Fe",
    dft_software="Gaussian",
    functional="PBE/Def2-SVP",
  )
  rows = [("CO", -1.60), ("SO2", -1.80)]
  return [
    record(**meta, gas=gas, adsorption_energy_ev=ead,
           extraction_notes="Table 1; PBE finite-system values from Cortes-Arriagada et al.")
    for gas, ead in rows
  ]


def build_all_curated_records():
  builders = [
    zhang_graphene_2009,
    leenaerts_graphene_2008,
    zhang_mos2_2014,
    kou_mos2_2013,
    beilstein_mos2_ws2_2018,
    junkaew_mxene_2018,
    ti3c2o2_strain_2025,
    sc2co2_ov_2022,
    sno2_nh3_2010,
    sno2_nh3_dissociative_blackman,
    sno2_no_no2_literature,
    kou_strain_mos2_2014,
    scientific_reports_mos2_2014,
    jiang_mos2_2021,
    al_mos2_so2_2018,
    zhou_ws2_2015,
    ga_n_codoped_graphene_so2_2023,
    mote2_metal_decorated_2024,
    au_ag_cu_mote2_sf6_gases_2022,
    rh_mote2_sf6_decomp_2020,
    gao_ni_pd_graphene_so2_2020,
    dalton_co_fe_graphene_2014,
    li_epjb_so2_doped_graphene_2013,
    ma_pd_graphene_2015,
    jin_mosese_2019_si,
    rani_b_pattern_graphene_co_nh3_2020,
    gutierrez_coinage_mos2_2023,
    medford_tm_embedded_graphene_2015,
    nosheen_fen_graphene_2022,
    karimi_al_graphene_so2_2022,
    chaurasiya_mosese_2019,
    li_ag_graphene_2021,
    tang_co_graphene_2015,
    cortes_feg_2017,
    acs_omega_phase_mos2_ws2_2020,
    salih_zgnr_2020,
    li_mos2_no2_2019,
    liu_ir_mos2_sf6_2021,
    raya_zr_hf_dichalcogenides_2020,
    maji_multi_b_graphene_2026,
  ]
  records = []
  for builder in builders:
    records.extend(builder())
  return records


def save_curated_json(records):
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  with CURATED_PATH.open("w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)


def records_to_dataframe(records):
  df = pd.DataFrame(records)
  for col in CSV_COLUMNS:
    if col not in df.columns:
      df[col] = "NA"
  return df[CSV_COLUMNS]


def build_paper_registry(df):
  registry = (
    df.groupby(["DOI", "Title", "Authors", "Journal", "Year"], dropna=False)
    .size()
    .reset_index(name="record_count")
    .sort_values("record_count", ascending=False)
  )
  return registry


def build_sources_catalog(df):
  catalog = df[
    [
      "DOI", "Title", "Authors", "Journal", "Year",
      "Material", "Doping", "Gas", "Adsorption_Energy_eV",
      "Charge_Transfer_e", "Adsorption_Distance_A",
      "DFT_Software", "Functional", "Data_Source", "Extraction_Notes",
    ]
  ].copy()
  catalog.insert(0, "Record_ID", range(1, len(catalog) + 1))
  catalog["Source_URL"] = catalog["DOI"].apply(
    lambda d: f"https://doi.org/{d}" if pd.notna(d) and d != "NA" else "NA"
  )
  return catalog


def main():
  records = build_all_curated_records()
  save_curated_json(records)
  df = records_to_dataframe(records)

  # Load supplemental verified batch if present
  supplemental = DATA_DIR / "supplemental_records.json"
  if supplemental.exists():
    with supplemental.open(encoding="utf-8") as f:
      extra = json.load(f)
    df = pd.concat([df, records_to_dataframe(extra)], ignore_index=True)

  df.to_csv(OUTPUT_CSV, index=False)
  df[ML_COLUMNS].to_csv(OUTPUT_ML_CSV, index=False)
  registry = build_paper_registry(df)
  registry.to_csv(PAPER_REGISTRY, index=False)
  catalog = build_sources_catalog(df)
  catalog.to_csv(SOURCES_CATALOG, index=False)

  print(f"Wrote {len(df)} material-gas records to {OUTPUT_CSV}")
  print(f"Source catalog: {SOURCES_CATALOG}")
  print(f"ML-ready subset: {OUTPUT_ML_CSV}")
  print(f"Unique papers: {df['DOI'].nunique()}")
  print(f"Materials: {sorted(df['Material'].unique())}")
  print(f"Gases: {sorted(df['Gas'].unique())}")
  print(f"Records with adsorption energy: {(df['Adsorption_Energy_eV'] != 'NA').sum()}")


if __name__ == "__main__":
  main()
