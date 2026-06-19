"""
Search peer-reviewed literature for DFT gas-sensing papers in target journals.

Uses OpenAlex (no API key). Outputs a candidate paper list for manual/semi-automated extraction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT = DATA_DIR / "literature_candidates.json"
OUTPUT_CSV = DATA_DIR / "literature_candidates.csv"

TARGET_JOURNAL_SUBSTRINGS = [
    "sensors and actuators b",
    "applied surface science",
    "acs sensors",
    "acs applied electronic materials",
    "journal of physical chemistry c",
    "computational materials science",
    "journal of physics and chemistry of solids",
    "physical chemistry chemical physics",
    "chemical physics letters",
    "nanoscale",
    "journal of materials chemistry",
    "beilstein journal of nanotechnology",
    "sensors",
]

SEARCH_QUERIES = [
    "DFT adsorption energy gas sensing MoS2",
    "DFT adsorption energy gas sensing SnO2",
    "DFT adsorption energy gas sensing ZnO",
    "DFT adsorption energy gas sensing graphene",
    "DFT adsorption energy gas sensing MXene",
    "DFT adsorption energy gas sensing TiO2",
    "DFT adsorption energy gas sensing WO3",
    "first principles NH3 NO2 gas sensor metal oxide",
    "CASTEP adsorption energy gas sensing",
    "DMol3 adsorption energy gas sensor",
]


def openalex_search(query: str, per_page: int = 50) -> list[dict]:
    url = (
        "https://api.openalex.org/works"
        f"?search={quote(query)}"
        "&filter=type:article,from_publication_date:2010-01-01"
        f"&per_page={per_page}"
        "&select=id,doi,title,publication_year,authorships,primary_location,cited_by_count"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])


def journal_name(work: dict) -> str:
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    return source.get("display_name") or ""


def author_string(work: dict, max_authors: int = 3) -> str:
    authorships = work.get("authorships") or []
    names = []
    for a in authorships[:max_authors]:
        author = a.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
    if len(authorships) > max_authors:
        names.append("et al.")
    return ", ".join(names) if names else "NA"


def is_target_journal(journal: str) -> bool:
    j = journal.lower()
    return any(s in j for s in TARGET_JOURNAL_SUBSTRINGS)


def normalize_work(work: dict, query: str) -> dict:
    doi = work.get("doi") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    return {
        "openalex_id": work.get("id"),
        "doi": doi or "NA",
        "title": work.get("title") or "NA",
        "authors": author_string(work),
        "journal": journal_name(work),
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "search_query": query,
        "target_journal_match": is_target_journal(journal_name(work)),
        "extraction_status": "pending",
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    candidates = []

    for query in SEARCH_QUERIES:
        print(f"Searching: {query}")
        try:
            results = openalex_search(query)
        except requests.RequestException as exc:
            print(f"  failed: {exc}")
            continue
        for work in results:
            key = work.get("doi") or work.get("id")
            if key in seen:
                continue
            seen.add(key)
            candidates.append(normalize_work(work, query))
        time.sleep(0.3)

    candidates.sort(
        key=lambda x: (
            not x["target_journal_match"],
            -(x["cited_by_count"] or 0),
        )
    )

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    import pandas as pd

    pd.DataFrame(candidates).to_csv(OUTPUT_CSV, index=False)
    target_hits = sum(1 for c in candidates if c["target_journal_match"])
    print(f"Saved {len(candidates)} unique candidates ({target_hits} target-journal matches)")
    print(f"  JSON: {OUTPUT}")
    print(f"  CSV:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
