"""LLM: predict Eg/χ for a material (Azure OpenAI / OpenAI / Anthropic).

Uses domain knowledge + chemical formula reasoning only — no web/deep search.

Preferred (Azure) — put in `.env`:
  AZURE_OPENAI_API_KEY=...
  AZURE_OPENAI_ENDPOINT=https://shopsifu.cognitiveservices.azure.com/
  AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
  AZURE_OPENAI_API_VERSION=2025-01-01-preview
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


DOMAIN_CONTEXT = """
You are a DETERMINISTIC formula calculator for OptoStack (no creativity).

HARD RULES
- NO web search, browsing, deep research, tool calls, or recalling papers.
- NO inventing literature values or DOIs.
- ONLY: parse the chemical formula into elemental counts, then apply the FIXED RULES below.
- Same formula must always yield the same Eg and χ for perovskite absorber families.

STEP 1 — Parse formula into elemental counts (e.g. Cs2AgBiBr6 → Cs:2, Ag:1, Bi:1, Br:6).
  Expand FA/MA organic cations (FA=CH5N2, MA=CH3NH3).

STEP 2 — Classify perovskite family, then apply family priors (not generic role-only):
  ABX3 halide (A=Cs/Rb/K/MA/FA; B=Pb/Sn/Ge; X=I/Br/Cl):
    Vegard from end-members; I < Br < Cl Eg. Mixed A: sum f_A * Eg(ABX3).
  Halide double A2B'B''X6: wider/indirect; Br~2.0–2.3, Cl~2.7–3.3 eV; χ~3.9–4.0.
  Vacancy-ordered A2BX6: iodides ~1.3–1.8 eV; Br/Cl much wider (Cs2SnBr6~3.2, Cs2SnCl6~4.9).
  A3B2X9 (0D): Sb/Bi iodides ~2.0 eV; bromides ~2.6–2.9 eV.
  Oxide perovskite: wide Eg (≥2 eV); χ~4.1 — not primary halide screening.
  2D RP/DJ / monolayers / contacts (ZnO,TiO2,SnO2,MoO3,NiO,organics) / GaAs,CdTe,BeSiP2:
    NOT absorbers — do not invent absorber Eg for screening.

STEP 3 — Role priors only for true ETL/HTL contacts (not perovskite absorbers)
  etl: Eg0=3.20, χ0=4.00
  htl: Eg0=2.80, χ0=2.40
  absorber halide χ prior ~3.7–4.2

STEP 4 — Halide / anion Eg shift when no Vegard end-member
  F:+0.45  Cl:+0.30  Br:+0.10  I:+0.00  O:+0.55 (atom-fraction weighted)

STEP 5 — Clamp Eg to family range (ABX3 ~[1.0,3.2]; A2BX6 ~[1.0,5.2]; double ~[1.5,3.8]).

OUTPUT
- method must be "formula_rules"
- confidence: high if Vegard end-members clear; medium if family prior; low if OOD
- Put the arithmetic steps briefly in candidates[0].notes
- warnings: list any assumptions (e.g. assumed bulk 3D)
"""

PROMPT = """Compute Eg (eV) and χ (eV) from formula rules ONLY (no literature recall):

Material: {material}
Role: {role}

Steps required in notes: (1) elemental counts (2) role priors (3) halide shift (4) final Eg/χ.

Return ONLY valid JSON:
{{
  "material": "{material}",
  "prediction": {{
    "Eg_eV": 0.0,
    "chi_eV": 0.0,
    "method": "formula_rules",
    "confidence": "medium"
  }},
  "candidates": [
    {{
      "Eg_eV": 0.0,
      "chi_eV": 0.0,
      "method": "formula_rules",
      "doi": "",
      "citation": "",
      "notes": "counts=...; Eg0=...; shift=...; Eg=...; chi=..."
    }}
  ],
  "warnings": []
}}
"""

SYSTEM = (
    DOMAIN_CONTEXT
    + "\nYou output ONLY JSON. No web search. No literature recall. "
    "Apply the fixed formula rules only."
)


def azure_api_key() -> str | None:
    return os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")


def resolve_provider(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if azure_api_key() and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "No LLM credentials. For Azure set in .env:\n"
        "  AZURE_OPENAI_API_KEY=...\n"
        "  AZURE_OPENAI_ENDPOINT=https://shopsifu.cognitiveservices.azure.com/\n"
        "  AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini\n"
        "  AZURE_OPENAI_API_VERSION=2025-01-01-preview"
    )


def call_azure(prompt: str) -> str:
    from openai import AzureOpenAI

    endpoint = os.environ.get(
        "AZURE_OPENAI_ENDPOINT", "https://shopsifu.cognitiveservices.azure.com/"
    ).rstrip("/")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    key = azure_api_key()
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY missing")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=key,
    )
    # Disable tools / force pure generation — no browsing
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        seed=42,
    )
    return resp.choices[0].message.content or "{}"


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        seed=42,
    )
    return resp.choices[0].message.content or "{}"


def call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=1500,
        temperature=0.0,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def cache_path(material: str, role: str) -> Path:
    safe = material.replace("/", "_").replace("\\", "_").strip()
    return ROOT / "data" / "llm_predictions" / f"{role}__{safe}.json"


def call_llm(prompt: str, provider: str = "auto") -> tuple[str, str]:
    provider = resolve_provider(provider)
    if provider == "azure":
        return call_azure(prompt), provider
    if provider == "openai":
        return call_openai(prompt), provider
    if provider == "anthropic":
        return call_anthropic(prompt), provider
    raise ValueError(f"Unknown provider: {provider}")


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def normalize_prediction(data: dict, material: str) -> dict:
    pred = data.get("prediction")
    if not isinstance(pred, dict) or pred.get("Eg_eV") is None:
        cands = data.get("candidates") or []
        if cands:
            top = cands[0]
            pred = {
                "Eg_eV": top.get("Eg_eV"),
                "chi_eV": top.get("chi_eV"),
                "method": top.get("method", "estimate"),
                "confidence": "medium",
            }
        else:
            pred = {"Eg_eV": None, "chi_eV": None, "method": "unavailable", "confidence": "low"}
    data["material"] = data.get("material") or material
    data["prediction"] = pred
    data["_meta"] = {
        "source": "llm",
        "output": "predicted_values",
        "mode": "formula_reasoning_no_web",
    }
    return data


def predict_material(
    material: str,
    role: str = "absorber",
    provider: str = "auto",
    *,
    force: bool = False,
) -> dict:
    """Predict Eg/χ. Reuses on-disk cache so repeated runs stay stable."""
    _load_dotenv()
    out = cache_path(material, role)
    if out.exists() and not force:
        data = json.loads(out.read_text(encoding="utf-8"))
        data.setdefault("_meta", {})
        data["_meta"]["cache_hit"] = True
        data["_meta"]["saved"] = str(out)
        return data

    prompt = PROMPT.format(material=material, role=role)
    raw, used = call_llm(prompt, provider)
    data = normalize_prediction(extract_json(raw), material)
    data["_meta"]["provider"] = used
    data["_meta"]["cache_hit"] = False
    if used == "azure":
        data["_meta"]["deployment"] = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["_meta"]["saved"] = str(out)
    return data


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Predict Eg/χ with Azure/OpenAI/Anthropic")
    ap.add_argument("--material", required=True)
    ap.add_argument("--role", default="absorber", choices=["absorber", "etl", "htl"])
    ap.add_argument(
        "--provider",
        choices=["azure", "openai", "anthropic", "auto"],
        default="auto",
    )
    args = ap.parse_args()

    try:
        data = predict_material(args.material, args.role, args.provider)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": "LLM did not return JSON", "detail": str(exc)}, indent=2))
        return
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return

    summary = {
        "material": data["material"],
        "predicted_Eg_eV": data["prediction"].get("Eg_eV"),
        "predicted_chi_eV": data["prediction"].get("chi_eV"),
        "method": data["prediction"].get("method"),
        "confidence": data["prediction"].get("confidence"),
        "provider": data["_meta"].get("provider"),
        "mode": data["_meta"].get("mode"),
        "saved": data["_meta"].get("saved"),
    }
    print(json.dumps(summary, indent=2))
    print("\nFull JSON:")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
