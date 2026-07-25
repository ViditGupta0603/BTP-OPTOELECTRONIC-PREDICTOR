"""Minimal OptoStack web UI (Python / Flask only).

Run:
  pip install -r requirements.txt
  python app.py
Then open http://127.0.0.1:7860

LLM is automatic: only used when a layer is new / missing χ.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import traceback
from pathlib import Path

from flask import Flask, render_template_string, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from predict_stack import EG_MODEL, TYPE_MODEL, load_layer_lookup, predict_stack, train_eg_model, train_type_models  # noqa: E402

app = Flask(__name__)

# Notes that mention χ / electron affinity or "lookup" should not surface in the UI.
_CHI_NOTE_RE = re.compile(
    r"(?:\bchi\b|\bχ\b|electron\s*affinity)",
    re.IGNORECASE,
)
_LOOKUP_WORD_RE = re.compile(r"\blookup\b", re.IGNORECASE)


def _ui_notes(notes: list | None) -> list[str]:
    """Filter/sanitize notes for user-facing display (no χ, no 'lookup')."""
    out: list[str] = []
    for raw in notes or []:
        text = str(raw).strip()
        if not text:
            continue
        if _CHI_NOTE_RE.search(text):
            # Drop chi-only notes; strip chi clauses from mixed notes.
            if re.search(r"(?i)^\s*estimated\s+absorber\s+chi\b", text):
                continue
            text = re.sub(r"(?i),\s*chi\s*=\s*[-+]?\d+(?:\.\d+)?", "", text)
            text = re.sub(r"(?i)\s*/\s*χ\b", "", text)
            text = re.sub(r"(?i)\bEg/χ\b", "Eg", text)
            text = re.sub(r"(?i)\bχ\b", "", text)
            text = re.sub(r"(?i)\bchi\b", "", text)
            text = re.sub(r"\s{2,}", " ", text).strip(" ·,;")
            if not text or _CHI_NOTE_RE.search(text):
                continue
        text = _LOOKUP_WORD_RE.sub("library", text)
        out.append(text)
    return out


_CHI_JSON_KEYS = re.compile(r"(?i)(^|_)(chi|electron_affinity)(_|$)")


def _ui_result_json(result: dict | None) -> str:
    """JSON dump for the UI details panel — strip χ fields and lookup labels."""
    if not result:
        return ""

    def scrub(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if _CHI_JSON_KEYS.search(str(k)):
                    continue
                if k == "field_labels" and isinstance(v, dict):
                    out[k] = {
                        fk: fv
                        for fk, fv in v.items()
                        if fv == "predicted" and not _CHI_JSON_KEYS.search(str(fk))
                    }
                    continue
                if k == "notes" and isinstance(v, list):
                    out[k] = _ui_notes(v)
                    continue
                if k == "label" and v in ("lookup", "literature"):
                    continue
                if k == "method" and isinstance(v, str) and "chi" in v.lower():
                    out[k] = "physics"
                    continue
                out[k] = scrub(v)
            return out
        if isinstance(obj, list):
            return [scrub(x) for x in obj]
        if isinstance(obj, str):
            return _LOOKUP_WORD_RE.sub("library", obj)
        return obj

    return json.dumps(scrub(result), indent=2)

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>OptoStack</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --ink: #1a1a1a;
      --muted: #5c5c5c;
      --line: #d4cfc4;
      --card: #fffdf8;
      --accent: #2f5d50;
      --accent2: #8b4513;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }
    main {
      max-width: 760px;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }
    h1 {
      font-size: 1.75rem;
      margin: 0 0 0.35rem;
      letter-spacing: -0.02em;
    }
    .sub { color: var(--muted); margin: 0 0 1.5rem; font-size: 0.95rem; }
    form, .out {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 1.25rem;
      margin-bottom: 1rem;
    }
    label { display: block; font-size: 0.85rem; font-weight: 600; margin: 0.75rem 0 0.3rem; }
    input[type=text], select {
      width: 100%;
      padding: 0.55rem 0.65rem;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      font-size: 0.95rem;
    }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
    @media (max-width: 600px) { .row { grid-template-columns: 1fr; } }
    button {
      margin-top: 1rem;
      background: var(--accent);
      color: #fff;
      border: 0;
      border-radius: 6px;
      padding: 0.65rem 1.1rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { filter: brightness(1.08); }
    .hint { color: var(--muted); font-size: 0.8rem; margin-top: 0.35rem; }
    .pill {
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      background: #e7efe9;
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 600;
      margin-right: 0.35rem;
      margin-bottom: 0.25rem;
    }
    .pill.warn { background: #f3e6d8; color: var(--accent2); }
    .pill.yes { background: #d8efe0; color: #1e5c3a; }
    .pill.marginal { background: #f5e8c8; color: #7a5a12; }
    .pill.no { background: #f0d8d8; color: #7a1e1e; }
    .pill.blocked { background: #f0d8d8; color: #7a1e1e; }
    .pill.predicted { background: #f3e6d8; color: var(--accent2); }
    .pill.user { background: #ebe8f5; color: #4a3a7a; }
    .verdict {
      margin-top: 1rem;
      padding: 0.85rem 1rem;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .verdict p { margin: 0.4rem 0 0; color: var(--muted); font-size: 0.9rem; }
    .src { font-size: 0.72rem; font-weight: 600; margin-left: 0.35rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem; }
    @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.75rem;
      background: #fff;
    }
    .stat b { display: block; font-size: 1.05rem; margin-top: 0.2rem; }
    .stat span { color: var(--muted); font-size: 0.78rem; }
    pre {
      background: #1e1e1e;
      color: #e8e8e8;
      padding: 0.85rem;
      border-radius: 8px;
      overflow: auto;
      font-size: 0.78rem;
      margin-top: 0.75rem;
    }
    .err { color: #8b1e1e; background: #f8e8e8; border: 1px solid #e0b4b4;
           padding: 0.75rem; border-radius: 8px; }
    details { margin-top: 0.75rem; }
  </style>
</head>
<body>
<main>
  <h1>OptoStack</h1>
  <p class="sub">Enter a <strong>perovskite absorber</strong> + ETL + HTL. Get junction Type (I / II / III) and optoelectronic suitability.
     Screening is perovskite-only: non-perovskite absorbers (CZTS, CIGS, CdTe, GaAs, Si, …) are blocked.
     Known materials use library values; unknowns use a deterministic <strong>ML/formula estimator</strong> (LLM off by default).</p>

  <form method="post">
    <label>Absorber</label>
    <input type="text" name="absorber" list="absorbers" required
           value="{{ absorber }}" placeholder="e.g. K2TiI6 or Cs2AgBiBr6"/>
    <datalist id="absorbers">
      {% for a in absorbers %}<option value="{{ a }}"></option>{% endfor %}
    </datalist>

    <div class="row">
      <div>
        <label>ETL</label>
        <input type="text" name="etl" list="etls" required
               value="{{ etl }}" placeholder="e.g. TiO2, SnO2, ZnO…"/>
        <datalist id="etls">
          {% for e in etls %}<option value="{{ e }}"></option>{% endfor %}
        </datalist>
      </div>
      <div>
        <label>HTL</label>
        <input type="text" name="htl" list="htls" required
               value="{{ htl }}" placeholder="e.g. MoO3, NiO, CuI…"/>
        <datalist id="htls">
          {% for h in htls %}<option value="{{ h }}"></option>{% endfor %}
        </datalist>
      </div>
    </div>
    <p class="hint">Perovskite formulas only (ABX₃, A₂BB′X₆, A₂BX₆, A₃B₂X₉, …). Non-perovskites are rejected. Predicted values are tagged when ML/estimate is used.</p>

    <button type="submit">Predict Type &amp; suitability</button>
  </form>

  {% if error %}
  <div class="out err"><strong>Error:</strong> {{ error }}</div>
  {% endif %}

  {% if result %}
  {% set L = result.field_labels or {} %}
  <div class="out">
    {% if result.not_perovskite %}
    <div class="verdict" style="margin-top:0;margin-bottom:1rem;border-color:#e0b4b4;background:#f8ecec">
      <span>Perovskite screening</span>
      <span class="pill blocked">not perovskite</span>
      {% if result.blocked %}<span class="pill blocked">blocked</span>{% endif %}
      <p><strong>{{ result.message }}</strong></p>
      {% if result.hint %}
      <p class="hint">{{ result.hint }}</p>
      {% endif %}
      {% if result.literature_reference %}
      <p class="hint">Literature reference: {{ result.literature_reference.source_paper }} ({{ result.literature_reference.material_class }})</p>
      {% endif %}
    </div>
    {% endif %}

    {% if not result.blocked %}
    <div>
      {% if result.method == 'compute_from_Eg_chi' %}
      <span class="pill">physics</span>
      {% elif result.method == 'literature_stack_row' %}
      <span class="pill">known stack</span>
      {% elif result.method == 'ml_type_from_names_and_Eg' %}
      <span class="pill warn">ML Type</span>
      {% elif result.method %}
      <span class="pill">{{ result.method }}</span>
      {% endif %}
      {% if result.sources and result.sources.llm_used %}
      <span class="pill warn">LLM</span>
      {% else %}
      <span class="pill">no LLM</span>
      {% endif %}
      {% if result.perovskite_family %}
      <span class="pill">{{ result.perovskite_family }}</span>
      {% endif %}
      {% if result.confidence %}
      <span class="pill {% if result.caution or result.confidence == 'low' %}warn{% endif %}">confidence {{ result.confidence }}</span>
      {% endif %}
      {% if result.caution %}
      <span class="pill warn">OOD caution</span>
      {% endif %}
    </div>

    <div class="grid">
      <div class="stat">
        <span>Absorber–ETL</span>
        <b>{{ result.absorber_etl_type or result.predicted_absorber_etl_type or '—' }}
          {% if L.get('absorber_etl_type') == 'predicted' %}<span class="pill src predicted">predicted</span>{% endif %}
        </b>
      </div>
      <div class="stat">
        <span>Absorber–HTL</span>
        <b>{{ result.absorber_htl_type or result.predicted_absorber_htl_type or '—' }}
          {% if L.get('absorber_htl_type') == 'predicted' %}<span class="pill src predicted">predicted</span>{% endif %}
        </b>
      </div>
      <div class="stat">
        <span>Absorber Eg (eV)</span>
        <b>{{ '%.3f'|format(result.absorber_band_gap_eV) if result.absorber_band_gap_eV is not none else '—' }}
          {% if L.get('absorber_Eg') == 'predicted' %}<span class="pill src predicted">predicted</span>{% endif %}
        </b>
      </div>
      <div class="stat">
        <span>ETL Eg (eV)</span>
        <b>{% if result.etl_band_gap_eV is not none %}{{ '%.2f'|format(result.etl_band_gap_eV) }}{% else %}—{% endif %}
          {% if L.get('etl_Eg') == 'predicted' %}<span class="pill src predicted">predicted</span>{% endif %}
        </b>
      </div>
      <div class="stat">
        <span>HTL Eg (eV)</span>
        <b>{% if result.htl_band_gap_eV is not none %}{{ '%.2f'|format(result.htl_band_gap_eV) }}{% else %}—{% endif %}
          {% if L.get('htl_Eg') == 'predicted' %}<span class="pill src predicted">predicted</span>{% endif %}
        </b>
      </div>
    </div>

    {% if result.optoelectronic %}
    <div class="verdict">
      <span>Optoelectronic device use?</span>
      {% set v = result.optoelectronic.verdict %}
      <span class="pill {{ 'yes' if v == 'YES' else ('marginal' if v == 'MARGINAL' else ('blocked' if v == 'BLOCKED' else 'no')) }}">{{ v }}</span>
      {% set opto_label = result.optoelectronic.label or L.get('optoelectronic') %}
      {% if opto_label == 'predicted' %}<span class="pill src predicted">predicted</span>{% endif %}
      <p>{{ result.optoelectronic.reason }}</p>
      <p class="hint">Rule: Type I/II OK · Type III usually not preferred · both I/II → YES · one III → MARGINAL · both III → NO</p>
    </div>
    {% endif %}

    {% if ui_notes and not result.blocked %}
    <p class="hint" style="margin-top:0.9rem">{{ ui_notes | join(' · ') }}</p>
    {% endif %}
    {% endif %}

    <details>
      <summary>Full JSON</summary>
      <pre>{{ result_json }}</pre>
    </details>
  </div>
  {% endif %}
</main>
</body>
</html>
"""


def _contact_lists() -> tuple[list[str], list[str], list[str]]:
    """Suggestion lists from material libraries (fallback to raw SCAPS)."""
    etls: set[str] = set()
    htls: set[str] = set()
    abs_names: set[str] = set()

    abs_lib = ROOT / "data" / "perovskite_absorber_library.csv"
    etl_lib = ROOT / "data" / "etl_material_library.csv"
    htl_lib = ROOT / "data" / "htl_material_library.csv"

    if abs_lib.exists():
        with abs_lib.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("material_absorber") or "").strip()
                if not name:
                    continue
                # Prefer base formula without phase for UI suggestions
                base = re.sub(r"\s*\(.*\)\s*$", "", name)
                abs_names.add(base)

    if etl_lib.exists():
        with etl_lib.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("material"):
                    etls.add(row["material"].strip())
    if htl_lib.exists():
        with htl_lib.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("material"):
                    htls.add(row["material"].strip())

    if not etls or not htls:
        raw = ROOT / "data" / "raw"
        for fname in (
            "paper4_scaps_materials.csv",
            "paper_cs_pb_scaps_materials.csv",
            "paper_cs3sb2br9_scaps_materials.csv",
            "paper_besip2_scaps_materials.csv",
        ):
            path = raw / fname
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    role, name = row.get("layer_role"), row.get("material")
                    if not name:
                        continue
                    if role == "etl":
                        etls.add(name)
                    elif role == "htl":
                        htls.add(name)
                    elif role == "absorber" and name != "BeSiP2":
                        abs_names.add(name)

    preferred = ["K2TiI6", "CsPb0.625Zn0.375IBr2", "Cs3Sb2Br9", "K2GeI6", "Cs2AgBiBr6"]
    absorbers = [a for a in preferred if a in abs_names]
    absorbers += sorted(abs_names - set(absorbers))[:80]  # keep datalist manageable
    return absorbers, sorted(etls), sorted(htls)


@app.route("/", methods=["GET", "POST"])
def index():
    absorbers, etls, htls = _contact_lists()
    absorber = "K2TiI6"
    etl = "TiO2" if "TiO2" in etls else (etls[0] if etls else "")
    htl = "MoO3" if "MoO3" in htls else (htls[0] if htls else "")
    result = None
    error = None

    if request.method == "POST":
        absorber = (request.form.get("absorber") or "").strip()
        etl = (request.form.get("etl") or "").strip()
        htl = (request.form.get("htl") or "").strip()
        try:
            if not EG_MODEL.exists() or not TYPE_MODEL.exists():
                load_layer_lookup()
                train_eg_model()
                train_type_models()
            # Default: library values + ML formula estimator (no LLM)
            result = predict_stack(absorber, etl, htl, use_llm=False)
        except Exception as exc:
            error = str(exc)
            traceback.print_exc()

    return render_template_string(
        PAGE,
        absorbers=absorbers,
        etls=etls,
        htls=htls,
        absorber=absorber,
        etl=etl,
        htl=htl,
        result=result,
        ui_notes=_ui_notes(result.get("notes") if isinstance(result, dict) else None),
        result_json=_ui_result_json(result if isinstance(result, dict) else None),
        error=error,
    )


def main() -> None:
    if not EG_MODEL.exists() or not TYPE_MODEL.exists():
        print("Training models (first run)...")
        load_layer_lookup()
        print(train_eg_model())
        print(train_type_models())
    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0"
    print(f"Serving on http://{host}:{port} (local: http://127.0.0.1:{port})")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
