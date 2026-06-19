"""Project configuration for DFT gas-sensing dataset collection."""

TARGET_JOURNALS = [
    "Sensors and Actuators B: Chemical",
    "Applied Surface Science",
    "ACS Sensors",
    "ACS Applied Electronic Materials",
    "Journal of Physical Chemistry C",
    "Computational Materials Science",
    "Journal of Physics and Chemistry of Solids",
    "Sensors and Actuators B",
    "Physical Chemistry Chemical Physics",
    "Nanotechnology",
    "Chemical Physics Letters",
    "Nanoscale",
    "ACS Omega",
    "Journal of Materials Chemistry A",
    "Beilstein Journal of Nanotechnology",
]

TARGET_MATERIALS = [
    "ZnO", "SnO2", "TiO2", "WO3", "In2O3", "CuO", "Fe2O3",
    "Graphene", "Graphene Oxide", "Reduced Graphene Oxide",
    "MoS2", "WS2", "MoSe2", "WSe2", "Black Phosphorus",
    "Ti3C2Tx", "Nb2CTx", "Ti2CO2", "V2CO2", "Nb2CO2", "Mo2CO2",
]

TARGET_GASES = [
    "NH3", "NO2", "CO", "CO2", "SO2", "H2S", "CH4", "HCN", "NO",
    "Formaldehyde", "Acetone", "Ethanol", "Toluene", "Benzene",
    "H2", "H2O", "N2", "O2", "SOF2",
]

CSV_COLUMNS = [
    "DOI",
    "Title",
    "Authors",
    "Journal",
    "Year",
    "Material",
    "Material_Class",
    "Doping",
    "Material_Dimensionality",
    "Crystal_Structure",
    "Gas",
    "DFT_Software",
    "Functional",
    "Dispersion_Correction",
    "Supercell",
    "Adsorption_Site",
    "Adsorption_Energy_eV",
    "Charge_Transfer_e",
    "Adsorption_Distance_A",
    "Bandgap_Before_eV",
    "Bandgap_After_eV",
    "Bandgap_Change_eV",
    "WorkFunction_Before_eV",
    "WorkFunction_After_eV",
    "WorkFunction_Change_eV",
    "Sensitivity",
    "Selectivity",
    "Response_Time",
    "Recovery_Time",
    "Detection_Limit",
    "Data_Source",
    "Extraction_Notes",
]
