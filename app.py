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
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from formula_estimator import EG_CHI_MODEL, train_estimators  # noqa: E402
from formula_parse import normalize_formula_text  # noqa: E402
from predict_stack import (  # noqa: E402
    EG_MODEL,
    TYPE_MODEL,
    load_layer_lookup,
    predict_multilayer,
    predict_stack,
    train_eg_model,
    train_type_models,
)

app = Flask(__name__)


@app.after_request
def _no_cache_html(response):
    if response.content_type and "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-store"
    return response


_MODELS_LOCK = threading.Lock()
_MODELS_READY = threading.Event()
# How long a prediction request waits on a cold-start training run before it
# gives up and asks the user to retry, instead of hanging past the host's
# request timeout.
_WARMUP_WAIT_S = 90.0

# Notes that mention χ / electron affinity or "lookup" should not surface in the UI.
_CHI_NOTE_RE = re.compile(
    r"(?:\bchi\b|\bχ\b|electron\s*affinity)",
    re.IGNORECASE,
)
_LOOKUP_WORD_RE = re.compile(r"\blookup\b", re.IGNORECASE)

# Notes that expose how a value was produced (estimator/LLM plumbing). The
# per-field "predicted" badge already carries that information for the user.
_METHOD_NOTE_RE = re.compile(
    r"(?i)(\bllm\b|\bml\b|\bestimator\b|^\s*estimated\s+\w+\s+Eg\s*=)"
)
# "OOD" is modelling jargon; keep the caveat, drop the acronym.
_OOD_PHRASE_RE = re.compile(r"(?i)\bOOD\s*/\s*unusual\s+family\b")


def _ui_notes(notes: list | None) -> list[str]:
    """Filter/sanitize notes for user-facing display (no χ, no method internals)."""
    out: list[str] = []
    for raw in notes or []:
        text = str(raw).strip()
        if not text:
            continue
        if _METHOD_NOTE_RE.search(text):
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
        text = _OOD_PHRASE_RE.sub("unusual material family", text)
        # Suitability screening is no longer user-facing.
        if re.search(r"(?i)\bsuitabilit", text):
            text = re.sub(
                r"(?i);\s*suitability\s+YES\s+should\s+be\s+read\s+with\s+an\s+optical-absorption\s+caveat\.?",
                "; optical absorption may be weaker than for a direct-gap material.",
                text,
            )
            text = re.sub(r"(?i)\bsuitabilit\w*\b", "", text)
            text = re.sub(r"\s{2,}", " ", text).strip(" ·,;")
            if not text:
                continue
        out.append(text)
    return out


_CHI_JSON_KEYS = re.compile(r"(?i)(^|_)(chi|electron_affinity)(_|$)")


def _ui_result_json(result: dict | None) -> str:
    """JSON dump for the UI details panel — strip χ and suitability."""
    if not result:
        return ""

    def scrub(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in ("optoelectronic", "suitability"):
                    continue
                if _CHI_JSON_KEYS.search(str(k)):
                    continue
                if k == "field_labels" and isinstance(v, dict):
                    out[k] = {
                        fk: fv
                        for fk, fv in v.items()
                        if fk != "optoelectronic"
                        and fv == "predicted"
                        and not _CHI_JSON_KEYS.search(str(fk))
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
  <title>OptoStack — 2–7 layer stack</title>
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
    .grid.first { margin-top: 0; }
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
    .layer-row {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 0.65rem;
      align-items: center;
      margin-top: 0.55rem;
    }
    .layer-row input[type=checkbox] {
      width: 1.1rem;
      height: 1.1rem;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .layer-row span.idx { color: var(--muted); font-size: 0.82rem; font-weight: 600; min-width: 3rem; }
    button:disabled { opacity: 0.55; cursor: not-allowed; filter: none; }
    .junction-list { margin-top: 0.75rem; }
    .junction-item {
      border: 1px solid var(--line); border-radius: 8px; padding: 0.65rem 0.75rem;
      background: #fff; margin-bottom: 0.5rem;
    }
    .junction-item b { font-size: 1rem; }
    details { margin-top: 0.75rem; }
  </style>
</head>
<body>
<main>
  <h1>OptoStack</h1>
  <p class="sub">Enter an ordered perovskite stack of <strong>2–7 layers</strong> (top → bottom). Flag one or more as <strong>Absorber</strong>.
     A classic 3-layer device is <em>3 layers</em> with ETL, absorber, and HTL in order.
     Non-perovskite absorbers are blocked. Estimated values are tagged <strong>predicted</strong>.</p>

  <form method="post" id="stack-form">
    <label for="layer_count">Number of layers (physical order, top → bottom)</label>
    <select name="layer_count" id="layer_count">
      {% for n in range(2, 8) %}
      <option value="{{ n }}" {% if layer_count == n %}selected{% endif %}>{{ n }} layers</option>
      {% endfor %}
    </select>
    <p class="hint">Enter materials in stack order. Flag one or more layers as <strong>Absorber</strong> (e.g. tandem perovskite).</p>
    <div id="layer-list">
      {% for i in range(layer_count) %}
      <div class="layer-row">
        <span class="idx">Layer {{ i + 1 }}</span>
        <input type="text" name="layer_name_{{ i }}" list="materials"
               value="{{ custom_layers[i] if custom_layers and i < custom_layers|length else '' }}"
               placeholder="e.g. TiO2, MAPbI3, Spiro-OMeTAD"/>
        <label style="margin:0;font-weight:600;font-size:0.82rem;white-space:nowrap;">
          <input type="checkbox" name="absorber_flag_{{ i }}" value="1"
                 {% if custom_absorber_flags and i < custom_absorber_flags|length and custom_absorber_flags[i] %}checked{% endif %}/> Absorber
        </label>
      </div>
      {% endfor %}
    </div>
    <datalist id="materials">
      {% for a in absorbers %}<option value="{{ a }}"></option>{% endfor %}
      {% for e in etls %}<option value="{{ e }}"></option>{% endfor %}
      {% for h in htls %}<option value="{{ h }}"></option>{% endfor %}
    </datalist>
    <p class="hint">Transport layers use library Eg only (no ML fallback). Contact–contact junctions without library data return <strong>UNKNOWN</strong>.</p>

    <button type="submit" id="submit-btn">Predict Type</button>
  </form>

  {% if error %}
  <div class="out err"><strong>Error:</strong> {{ error }}</div>
  {% endif %}

  {% if result %}
  <div class="out">
    {% if result.blocked %}
    <div class="verdict" style="margin-top:0;margin-bottom:1rem;border-color:#e0b4b4;background:#f8ecec">
      <span>{% if result.not_perovskite %}Perovskite screening{% else %}Contact screening{% endif %}</span>
      {% if result.not_perovskite %}
      <span class="pill blocked">not perovskite</span>
      {% elif result.invalid_contact %}
      <span class="pill blocked">invalid contact</span>
      {% endif %}
      <span class="pill blocked">blocked</span>
      <p><strong>{{ result.message }}</strong></p>
      {% if result.hint %}
      <p class="hint">{{ result.hint }}</p>
      {% endif %}
    </div>
    {% endif %}

    {% if not result.blocked %}
    <div class="grid first">
      {% for layer in result.layers %}
      <div class="stat">
        <span>Layer {{ layer.index + 1 }} — {{ layer.name }} ({{ layer.role }})</span>
        <b>{% if layer.eg_ev is not none %}{{ '%.3f'|format(layer.eg_ev) }}{% else %}—{% endif %} eV
          {% if layer.eg_method == 'predicted' %}<span class="pill src predicted">predicted</span>{% elif layer.eg_method == 'missing' %}<span class="pill warn">missing</span>{% endif %}
        </b>
      </div>
      {% endfor %}
    </div>
    <div class="junction-list">
      <p class="hint" style="margin-top:0.9rem"><strong>Adjacent junctions</strong></p>
      {% for j in result.junctions %}
      <div class="junction-item">
        <span>{{ j.layer_a }} – {{ j.layer_b }}</span>
        <b>{{ j.type }}
          {% if j.method == 'ML/predicted' %}<span class="pill src predicted">predicted</span>{% elif j.method == 'UNKNOWN' %}<span class="pill warn">unknown</span>{% endif %}
        </b>
        {% if j.note %}<p class="hint">{{ j.note }}</p>{% endif %}
      </div>
      {% endfor %}
    </div>
    {% endif %}

    {% if ui_notes and not result.blocked %}
    <p class="hint" style="margin-top:0.9rem">{{ ui_notes | join(' · ') }}</p>
    {% endif %}

    <details>
      <summary>Full JSON</summary>
      <pre>{{ result_json }}</pre>
    </details>
  </div>
  {% endif %}
</main>
<script>
(function () {
  const layerCount = document.getElementById('layer_count');
  const layerList = document.getElementById('layer-list');
  const submitBtn = document.getElementById('submit-btn');

  function currentLayerValues() {
    const vals = [];
    layerList.querySelectorAll('input[type=text]').forEach((el) => vals.push(el.value));
    return vals;
  }
  function currentAbsorberFlags() {
    const flags = [];
    layerList.querySelectorAll('.layer-row').forEach((row) => {
      const cb = row.querySelector('input[type=checkbox]');
      flags.push(cb ? cb.checked : false);
    });
    return flags;
  }
  function renderLayers(n, preserve) {
    const oldVals = preserve ? currentLayerValues() : [];
    const oldFlags = preserve ? currentAbsorberFlags() : [false, true, false];
    layerList.innerHTML = '';
    for (let i = 0; i < n; i++) {
      const row = document.createElement('div');
      row.className = 'layer-row';
      const checked = oldFlags[i] ? ' checked' : '';
      row.innerHTML =
        '<span class="idx">Layer ' + (i + 1) + '</span>' +
        '<input type="text" name="layer_name_' + i + '" list="materials" value="' +
        (oldVals[i] || '').replace(/"/g, '&quot;') + '" placeholder="e.g. TiO2, MAPbI3, Spiro-OMeTAD"/>' +
        '<label style="margin:0;font-weight:600;font-size:0.82rem;white-space:nowrap;">' +
        '<input type="checkbox" name="absorber_flag_' + i + '" value="1"' + checked + '/> Absorber</label>';
      layerList.appendChild(row);
    }
    updateSubmit();
  }
  function updateSubmit() {
    const boxes = layerList.querySelectorAll('input[type=checkbox]');
    submitBtn.disabled = !Array.from(boxes).some((cb) => cb.checked);
  }
  layerCount.addEventListener('change', () => renderLayers(parseInt(layerCount.value, 10), true));
  layerList.addEventListener('change', updateSubmit);
  updateSubmit();
})();
</script>
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
                name = normalize_formula_text(row.get("material_absorber"))
                if not name:
                    continue
                # Prefer base formula without phase for UI suggestions
                base = re.sub(r"\s*\(.*\)\s*$", "", name)
                abs_names.add(base)

    if etl_lib.exists():
        with etl_lib.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_formula_text(row.get("material"))
                if name:
                    etls.add(name)
    if htl_lib.exists():
        with htl_lib.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = normalize_formula_text(row.get("material"))
                if name:
                    htls.add(name)

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
                    role = row.get("layer_role")
                    name = normalize_formula_text(row.get("material"))
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
    layer_count = 3
    custom_layers: list[str] = ["TiO2", "MAPbI3", "MoO3"]
    custom_absorber_flags: list[bool] = [False, True, False]
    result = None
    error = None

    if request.method == "POST":
        try:
            if not ensure_models(timeout=_WARMUP_WAIT_S):
                error = (
                    "The server is still warming up its models after a restart. "
                    "Please try again in a minute."
                )
            else:
                layer_count = int(request.form.get("layer_count") or "3")
                layer_count = max(2, min(7, layer_count))
                custom_layers = [
                    normalize_formula_text(request.form.get(f"layer_name_{i}") or "")
                    for i in range(layer_count)
                ]
                custom_absorber_flags = [
                    request.form.get(f"absorber_flag_{i}") == "1"
                    for i in range(layer_count)
                ]
                specs = [
                    {"name": n, "is_absorber": custom_absorber_flags[i]}
                    for i, n in enumerate(custom_layers)
                ]
                result = predict_multilayer(specs, use_llm=False)
        except Exception as exc:
            error = str(exc)
            traceback.print_exc()

    return render_template_string(
        PAGE,
        absorbers=absorbers,
        etls=etls,
        htls=htls,
        layer_count=layer_count,
        custom_layers=custom_layers,
        custom_absorber_flags=custom_absorber_flags,
        result=result,
        ui_notes=_ui_notes(result.get("notes") if isinstance(result, dict) else None),
        result_json=_ui_result_json(result if isinstance(result, dict) else None),
        error=error,
    )


@app.route("/predict_multilayer", methods=["POST"])
def predict_multilayer_api():
    """JSON API for custom N-layer stacks (FR-11 sketch)."""
    if not ensure_models(timeout=_WARMUP_WAIT_S):
        return jsonify(
            {"error": "Models still warming up — retry shortly.", "models_ready": False}
        ), 503
    payload = request.get_json(silent=True) or {}
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return jsonify({"error": "Request body must include a layers array."}), 400
    specs = [
        {"name": str(item.get("name", "")), "is_absorber": bool(item.get("is_absorber"))}
        for item in layers
    ]
    try:
        result = predict_multilayer(
            specs,
            eg=payload.get("eg"),
            chi=payload.get("chi"),
            use_llm=bool(payload.get("use_llm")),
        )
        status = 400 if result.get("blocked") and not result.get("not_perovskite") else 200
        if result.get("not_perovskite"):
            status = 422
        return jsonify(result), status
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/predict_stack", methods=["POST"])
def predict_stack_api():
    """JSON API — existing 3-field stack (unchanged semantics)."""
    if not ensure_models(timeout=_WARMUP_WAIT_S):
        return jsonify(
            {"error": "Models still warming up — retry shortly.", "models_ready": False}
        ), 503
    payload = request.get_json(silent=True) or {}
    absorber = normalize_formula_text(payload.get("absorber") or "")
    etl = normalize_formula_text(payload.get("etl") or "")
    htl = normalize_formula_text(payload.get("htl") or "")
    if not (absorber and etl and htl):
        return jsonify({"error": "absorber, etl, and htl are required."}), 400
    try:
        result = predict_stack(
            absorber,
            etl,
            htl,
            eg=payload.get("eg"),
            chi=payload.get("chi"),
            use_llm=bool(payload.get("use_llm")),
        )
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/healthz")
def healthz():
    """Liveness probe that answers immediately, even while models are training."""
    return {"status": "ok", "models_ready": _MODELS_READY.is_set()}


def ensure_models(verbose: bool = False, timeout: float | None = None) -> bool:
    """Ensure model artifacts exist and load on this sklearn version.

    Production ``.joblib`` files are committed under ``data/models/``. If they
    were trained on a different scikit-learn build, ``joblib.load`` can fail
    with ``No module named '_loss'`` — in that case we delete the bad
    artifacts and retrain once on the running environment.

    A lock keeps the startup warm-up and a concurrent request from training the
    same artifact twice. ``timeout`` caps how long a caller waits for a warm-up
    that is already running (None waits indefinitely); False means training is
    still in flight, not that it failed.
    """
    if _MODELS_READY.is_set():
        return True
    if not _MODELS_LOCK.acquire(timeout=-1 if timeout is None else timeout):
        return False
    try:
        if _models_loadable():
            _MODELS_READY.set()
            return True
        if verbose:
            print("Training models (missing or incompatible with this sklearn)...", flush=True)
        load_layer_lookup()
        for path, train in (
            (EG_CHI_MODEL, train_estimators),
            (EG_MODEL, train_eg_model),
            (TYPE_MODEL, train_type_models),
        ):
            if path.exists() and not _joblib_loadable(path):
                if verbose:
                    print(f"Removing incompatible artifact: {path.name}", flush=True)
                path.unlink(missing_ok=True)
            if not path.exists():
                out = train()
                if verbose:
                    print(out, flush=True)
        if not _models_loadable():
            if verbose:
                print("Model warm-up failed: artifacts still unloadable.", flush=True)
            return False
        _MODELS_READY.set()
        if verbose:
            print("Models ready.", flush=True)
        return True
    finally:
        _MODELS_LOCK.release()


def _joblib_loadable(path: Path) -> bool:
    try:
        joblib.load(path)
        return True
    except Exception as exc:  # noqa: BLE001 — any unpickle failure means retrain
        print(f"Cannot load {path.name}: {exc}", flush=True)
        return False


def _models_loadable() -> bool:
    return (
        EG_MODEL.exists()
        and TYPE_MODEL.exists()
        and EG_CHI_MODEL.exists()
        and _joblib_loadable(EG_MODEL)
        and _joblib_loadable(TYPE_MODEL)
        and _joblib_loadable(EG_CHI_MODEL)
    )


def main() -> None:
    port = int(os.environ.get("PORT", 7860))
    host = "0.0.0.0"
    # Train in the background so the socket is bound within seconds: hosts such
    # as Render scan for an open port right after start and fail the deploy if
    # a cold-start training run (minutes) holds up app.run().
    threading.Thread(
        target=ensure_models,
        kwargs={"verbose": True},
        name="model-warmup",
        daemon=True,
    ).start()
    print(f"Serving on http://{host}:{port} (local: http://127.0.0.1:{port})", flush=True)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
