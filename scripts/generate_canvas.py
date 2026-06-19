"""Generate the dataset catalog canvas from built CSV data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
payload = json.loads((ROOT / "data" / "canvas_payload.json").read_text(encoding="utf-8"))
records_json = json.dumps(payload["records"], ensure_ascii=False)
papers_json = json.dumps(payload["papers"], ensure_ascii=False)
stats = payload["stats"]

canvas = f'''import {{ useMemo, useState }} from "react";
import {{
  Card, CardBody, CardHeader, Grid, H1, Stack, Stat, Table, Text, TextInput, Select, Link, Row,
}} from "cursor/canvas";

const RECORDS = {records_json} as const;
const PAPERS = {papers_json} as const;

type RecordRow = (typeof RECORDS)[number];

export default function DftDatasetCatalog() {{
  const [query, setQuery] = useState("");
  const [gasFilter, setGasFilter] = useState("all");
  const gases = useMemo(() => ["all", ...Array.from(new Set(RECORDS.map((r) => r.Gas))).sort()], []);

  const filtered = useMemo(() => {{
    const q = query.trim().toLowerCase();
    return RECORDS.filter((r) => {{
      if (gasFilter !== "all" && r.Gas !== gasFilter) return false;
      if (!q) return true;
      const hay = `${{r.Material}} ${{r.Gas}} ${{r.DOI}} ${{r.Authors}} ${{r.Journal}}`.toLowerCase();
      return hay.includes(q);
    }});
  }}, [query, gasFilter]);

  const recordColumns = [
    {{ key: "Record_ID", header: "#", width: 56 }},
    {{ key: "Material", header: "Material" }},
    {{ key: "Gas", header: "Gas", width: 72 }},
    {{ key: "Adsorption_Energy_eV", header: "E_ads (eV)", width: 96, align: "right" as const }},
    {{ key: "DOI", header: "DOI", render: (r: RecordRow) => <Link href={{`https://doi.org/${{r.DOI}}`}}>{{r.DOI}}</Link> }},
    {{ key: "Year", header: "Year", width: 64, align: "right" as const }},
    {{ key: "Extraction_Notes", header: "Source note" }},
  ];

  const paperColumns = [
    {{ key: "record_count", header: "Rows", width: 64, align: "right" as const }},
    {{ key: "Year", header: "Year", width: 64, align: "right" as const }},
    {{ key: "DOI", header: "DOI", render: (p: (typeof PAPERS)[number]) => <Link href={{`https://doi.org/${{p.DOI}}`}}>{{p.DOI}}</Link> }},
    {{ key: "Title", header: "Title" }},
    {{ key: "Authors", header: "Authors" }},
    {{ key: "Journal", header: "Journal" }},
  ];

  return (
    <Stack gap={{24}}>
      <Stack gap={{8}}>
        <H1>DFT Gas-Sensing Dataset — Full Catalog</H1>
        <Text tone="secondary">Curated literature extractions only; values taken from published tables/text.</Text>
        <Text tone="secondary" size="sm">Files: data/dft_gas_sensing_dataset.csv · data/dataset_full_with_sources.csv</Text>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat label="Total records" value={{String({stats["total"]})}} />
        <Stat label="Source papers" value={{String({stats["papers"]})}} />
        <Stat label="Materials" value={{String({stats["materials"]})}} />
        <Stat label="Gases" value={{String({stats["gases"]})}} />
      </Grid>

      <Card>
        <CardHeader title="Literature sources ({stats['papers']} papers)" subtitle="One row per peer-reviewed DOI" />
        <CardBody padding={{0}}>
          <Table columns={{paperColumns}} data={{PAPERS}} />
        </CardBody>
      </Card>

      <Card>
        <CardHeader
          title="All {stats['total']} material–gas records"
          subtitle={{`Showing ${{filtered.length}} of ${{RECORDS.length}} after filters`}}
          trailing={{
            <Row gap={{8}}>
              <Select value={{gasFilter}} onChange={{setGasFilter}} options={{gases.map((g) => ({{ label: g === "all" ? "All gases" : g, value: g }}))}} />
              <TextInput value={{query}} onChange={{setQuery}} placeholder="Search material, gas, DOI, author…" />
            </Row>
          }}
        />
        <CardBody padding={{0}}>
          <Table columns={{recordColumns}} data={{filtered}} />
        </CardBody>
      </Card>
    </Stack>
  );
}}
'''

out_path = Path(r"C:\Users\Admin\.cursor\projects\c-Users-vidit-Desktop-btp\canvases\dft-dataset-catalog.canvas.tsx")
out_path.write_text(canvas, encoding="utf-8")
print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")
