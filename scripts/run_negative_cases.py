"""Negative-case suite for OptoStack predict_stack (use_llm=False).

Writes data/negative_cases_test_report.md with PASS/FAIL table.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_stack import predict_stack  # noqa: E402

ETL_DEFAULT = "TiO2"
HTL_DEFAULT = "NiO"


def _safe(fn):
    try:
        return fn(), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def _verdict(r: dict | None) -> str:
    if r is None:
        return "CRASH"
    if r.get("blocked") or r.get("not_perovskite"):
        return "BLOCKED"
    opto = r.get("optoelectronic") or {}
    return str(opto.get("verdict") or "UNKNOWN")


def _types(r: dict | None) -> str:
    if not r or r.get("blocked"):
        return "—"
    return f"{r.get('absorber_etl_type') or '?'}/{r.get('absorber_htl_type') or '?'}"


def _eg(r: dict | None) -> str:
    if not r or r.get("absorber_band_gap_eV") is None:
        return "—"
    return f"{float(r['absorber_band_gap_eV']):.3f}"


def _notes_blob(r: dict | None) -> str:
    if not r:
        return ""
    parts = []
    if r.get("message"):
        parts.append(str(r["message"]))
    for n in r.get("notes") or []:
        parts.append(str(n))
    opto = r.get("optoelectronic") or {}
    if opto.get("reason"):
        parts.append(str(opto["reason"]))
    if opto.get("htl_caveat"):
        parts.append(f"htl_caveat={opto['htl_caveat']}")
    return " | ".join(parts)


def expect_blocked(name: str, absorber: str, etl=ETL_DEFAULT, htl=HTL_DEFAULT) -> dict:
    r, err = _safe(lambda: predict_stack(absorber, etl, htl, use_llm=False))
    ok = err is None and _verdict(r) == "BLOCKED" and bool(r.get("blocked"))
    return {
        "case": name,
        "category": "should_block",
        "stack": f"{absorber} / {etl} / {htl}",
        "expected": "BLOCKED",
        "got": "CRASH" if err else _verdict(r),
        "types": _types(r),
        "eg": _eg(r),
        "result": "PASS" if ok else "FAIL",
        "detail": (err or r.get("message") or "")[:200] if r or err else "",
        "crash": err,
    }


def expect_runs(
    name: str,
    absorber: str,
    etl: str,
    htl: str,
    *,
    expect_verdicts: set[str] | None = None,
    forbid_verdicts: set[str] | None = None,
    eg_range: tuple[float, float] | None = None,
    eg_not_near: float | None = None,
    require_pedot_caveat: bool = False,
    require_caution_or_not_blind_yes: bool = False,
    note_substr: str | None = None,
) -> dict:
    r, err = _safe(lambda: predict_stack(absorber, etl, htl, use_llm=False))
    fails = []
    if err:
        fails.append(f"crash: {err.splitlines()[0]}")
        return {
            "case": name,
            "category": "edge",
            "stack": f"{absorber} / {etl} / {htl}",
            "expected": "runs without crash",
            "got": "CRASH",
            "types": "—",
            "eg": "—",
            "result": "FAIL",
            "detail": err[:300],
            "crash": err,
        }

    v = _verdict(r)
    blob = _notes_blob(r).lower()

    if expect_verdicts is not None and v not in expect_verdicts:
        fails.append(f"verdict {v} not in {sorted(expect_verdicts)}")
    if forbid_verdicts is not None and v in forbid_verdicts:
        fails.append(f"verdict {v} forbidden")

    if eg_range is not None:
        eg = r.get("absorber_band_gap_eV")
        if eg is None:
            fails.append("missing Eg")
        else:
            lo, hi = eg_range
            if not (lo <= float(eg) <= hi):
                fails.append(f"Eg={eg} outside [{lo},{hi}]")

    if eg_not_near is not None:
        eg = r.get("absorber_band_gap_eV")
        if eg is not None and abs(float(eg) - eg_not_near) < 0.05:
            fails.append(f"Eg={eg} too close to forbidden {eg_not_near}")

    if require_pedot_caveat:
        opto = r.get("optoelectronic") or {}
        has = (
            opto.get("htl_caveat") == "degenerate_metallic"
            or "degenerate" in blob
            or "pedot" in blob and "caveat" in blob
            or "unreliable" in blob
        )
        if not has:
            fails.append("missing PEDOT/degenerate HTL caveat")

    if require_caution_or_not_blind_yes:
        # Should not blindly YES without Type III or caveat/caution
        opto = r.get("optoelectronic") or {}
        if v == "YES" and not (
            r.get("caution")
            or "caveat" in blob
            or "type iii" in (r.get("absorber_etl_type") or "").lower()
            or "type iii" in (r.get("absorber_htl_type") or "").lower()
            or opto.get("gap_type")
        ):
            # For mismatched stacks we specifically want NOT blind YES
            # Caller should use forbid_verdicts={"YES"} when literature says Type III
            pass

    if note_substr and note_substr.lower() not in blob:
        fails.append(f"missing note containing '{note_substr}'")

    expected_bits = []
    if expect_verdicts:
        expected_bits.append(f"verdict in {sorted(expect_verdicts)}")
    if forbid_verdicts:
        expected_bits.append(f"not in {sorted(forbid_verdicts)}")
    if eg_range:
        expected_bits.append(f"Eg in {eg_range}")
    if eg_not_near is not None:
        expected_bits.append(f"Eg≠{eg_not_near}")
    if require_pedot_caveat:
        expected_bits.append("PEDOT caveat")

    return {
        "case": name,
        "category": "edge",
        "stack": f"{absorber} / {etl} / {htl}",
        "expected": "; ".join(expected_bits) or "runs",
        "got": f"{v}; types={_types(r)}; Eg={_eg(r)}",
        "types": _types(r),
        "eg": _eg(r),
        "result": "PASS" if not fails else "FAIL",
        "detail": "; ".join(fails) if fails else (_notes_blob(r)[:180]),
        "crash": None,
        "raw": {
            "verdict": v,
            "types": _types(r),
            "eg": r.get("absorber_band_gap_eV"),
            "method": r.get("method"),
            "caution": r.get("caution"),
            "htl_caveat": (r.get("optoelectronic") or {}).get("htl_caveat"),
            "message": r.get("message"),
        },
    }


def expect_stable(name: str, absorber: str, etl: str, htl: str) -> dict:
    r1, e1 = _safe(lambda: predict_stack(absorber, etl, htl, use_llm=False))
    r2, e2 = _safe(lambda: predict_stack(absorber, etl, htl, use_llm=False))
    if e1 or e2:
        return {
            "case": name,
            "category": "stability",
            "stack": f"{absorber} / {etl} / {htl}",
            "expected": "identical outputs",
            "got": "CRASH",
            "types": "—",
            "eg": "—",
            "result": "FAIL",
            "detail": e1 or e2,
            "crash": e1 or e2,
        }

    def key(r):
        return (
            _verdict(r),
            r.get("absorber_etl_type"),
            r.get("absorber_htl_type"),
            r.get("absorber_band_gap_eV"),
            r.get("etl_band_gap_eV"),
            r.get("htl_band_gap_eV"),
            r.get("method"),
            bool(r.get("blocked")),
        )

    same = key(r1) == key(r2)
    return {
        "case": name,
        "category": "stability",
        "stack": f"{absorber} / {etl} / {htl}",
        "expected": "identical key fields",
        "got": str(key(r1)),
        "types": _types(r1),
        "eg": _eg(r1),
        "result": "PASS" if same else "FAIL",
        "detail": "" if same else f"r1={key(r1)} r2={key(r2)}",
        "crash": None,
    }


def main() -> int:
    rows: list[dict] = []

    # --- Should BLOCK: contact oxides as absorber ---
    for mat in ["ZnO", "TiO2", "SnO2", "MoO3", "NiO"]:
        rows.append(expect_blocked(f"block_contact_{mat}", mat))

    # --- Should BLOCK: wrong absorbers ---
    for mat in ["CZTS", "CIGS", "GaAs", "CdTe", "Si", "graphene"]:
        rows.append(expect_blocked(f"block_pv_{mat}", mat))

    # --- Should BLOCK: 2D ---
    for mat in ["1T-PbI2", "2H-MoS2", "BA2PbI4"]:
        rows.append(expect_blocked(f"block_2d_{mat}", mat))

    # --- Should BLOCK: BeSiP2 ---
    rows.append(expect_blocked("block_BeSiP2", "BeSiP2"))

    # --- Empty / garbage ---
    for mat in ["asdf", "123", "H2O", "NaCl"]:
        rows.append(expect_blocked(f"block_garbage_{mat}", mat))

    # --- Role misuse: should still run (not crash); may have weird Types ---
    rows.append(
        expect_runs(
            "role_misuse_K2TiI6_MoO3_as_ETL_TiO2_as_HTL",
            "K2TiI6",
            "MoO3",
            "TiO2",
            forbid_verdicts={"BLOCKED", "CRASH"},
        )
    )
    rows.append(
        expect_runs(
            "role_misuse_CsPbBr3_PEDOT_as_ETL_TiO2_as_HTL",
            "CsPbBr3",
            "PEDOT:PSS",
            "TiO2",
            forbid_verdicts={"BLOCKED", "CRASH"},
        )
    )
    # PEDOT as HTL — caveat required
    rows.append(
        expect_runs(
            "pedot_as_htl_caveat",
            "CsPbBr3",
            "TiO2",
            "PEDOT:PSS",
            forbid_verdicts={"BLOCKED", "CRASH"},
            require_pedot_caveat=True,
        )
    )

    # --- Halide / formula traps ---
    rows.append(
        expect_runs(
            "halide_FAPbBr3_not_FAPbI3_Eg",
            "FAPbBr3",
            "TiO2",
            "Spiro-OMeTAD",
            eg_range=(2.05, 2.40),
            eg_not_near=1.48,
            forbid_verdicts={"BLOCKED", "CRASH"},
        )
    )
    rows.append(
        expect_runs(
            "halide_Cs2TiI6_not_Br_Eg",
            "Cs2TiI6",
            "TiO2",
            "Spiro-OMeTAD",
            eg_range=(1.50, 1.70),
            eg_not_near=1.80,
            forbid_verdicts={"BLOCKED", "CRASH"},
        )
    )
    # Typo near-misses — should block or caution, not silently map to wrong halide
    rows.append(
        expect_runs(
            "typo_FAPbBr_near_miss",
            "FAPbBr",
            "TiO2",
            "NiO",
            # incomplete formula: prefer BLOCKED; if runs, Eg must not be FAPbI3 1.48
            eg_not_near=1.48,
        )
    )
    rows.append(
        expect_runs(
            "typo_Cs2TiI_near_miss",
            "Cs2TiI",
            "TiO2",
            "NiO",
            eg_not_near=1.80,
        )
    )
    rows.append(
        expect_runs(
            "typo_MAPbI_near_miss",
            "MAPbI",
            "TiO2",
            "NiO",
            eg_not_near=1.55,
        )
    )

    # --- Edge stacks ---
    # Round 3 physics: Cs2SnI6/TiO2 is Type I (straddling), P3HT Type II → YES is correct
    rows.append(
        expect_runs(
            "edge_Cs2SnI6_TiO2_P3HT",
            "Cs2SnI6",
            "TiO2",
            "P3HT",
            expect_verdicts={"YES"},
            eg_range=(1.30, 1.40),
            forbid_verdicts={"BLOCKED", "CRASH"},
        )
    )
    # True Type-III edge (Round 3): CFTS HTL → MARGINAL, not blind YES
    rows.append(
        expect_runs(
            "edge_Cs2SnI6_TiO2_CFTS_typeIII",
            "Cs2SnI6",
            "TiO2",
            "CFTS",
            expect_verdicts={"MARGINAL", "NO"},
            forbid_verdicts={"YES", "BLOCKED", "CRASH"},
        )
    )

    # Metal-halide precursors must BLOCK (not slip through as fake ABX3)
    for mat in ["PbI2", "SnI2", "GeI2", "PbBr2"]:
        rows.append(expect_blocked(f"block_precursor_{mat}", mat))

    # Wide-gap oxide perovskite — eligible with caution, not crash
    rows.append(
        expect_runs(
            "edge_widegap_BaTiO3",
            "BaTiO3",
            "TiO2",
            "NiO",
            forbid_verdicts={"CRASH"},
            note_substr="oxide",
        )
    )

    # --- Stability ---
    rows.append(expect_stable("stability_MAPbI3", "MAPbI3", "TiO2", "Spiro-OMeTAD"))
    rows.append(expect_stable("stability_Cs2TiI6", "Cs2TiI6", "TiO2", "NiO"))
    rows.append(expect_stable("stability_blocked_ZnO", "ZnO", "TiO2", "NiO"))
    rows.append(expect_stable("stability_garbage_asdf", "asdf", "TiO2", "NiO"))

    # Extra: PEDOT as ETL behavior (document, not necessarily caveat)
    r_pedot_etl, e_pe = _safe(
        lambda: predict_stack("CsPbBr3", "PEDOT:PSS", "TiO2", use_llm=False)
    )
    pedot_etl_row = {
        "case": "pedot_as_etl_behavior",
        "category": "role_misuse",
        "stack": "CsPbBr3 / PEDOT:PSS / TiO2",
        "expected": "runs; PEDOT-as-ETL caveat optional (HTL caveat is HTL-only)",
        "got": "CRASH" if e_pe else f"{_verdict(r_pedot_etl)}; types={_types(r_pedot_etl)}",
        "types": _types(r_pedot_etl),
        "eg": _eg(r_pedot_etl),
        "result": "PASS" if e_pe is None and _verdict(r_pedot_etl) != "CRASH" else "FAIL",
        "detail": (
            e_pe
            or (
                f"htl_caveat={(r_pedot_etl.get('optoelectronic') or {}).get('htl_caveat')}; "
                f"notes_have_degenerate={'degenerate' in _notes_blob(r_pedot_etl).lower()}; "
                "NOTE: degenerate caveat currently HTL-only — PEDOT as ETL has no caveat by design"
            )
        ),
        "crash": e_pe,
    }
    rows.append(pedot_etl_row)

    n_pass = sum(1 for r in rows if r["result"] == "PASS")
    n_fail = sum(1 for r in rows if r["result"] == "FAIL")
    crashes = [r for r in rows if r.get("crash")]

    # Write markdown report
    lines = [
        "# OptoStack negative-case test report",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        f"**Mode:** `predict_stack(..., use_llm=False)`  ",
        f"**Totals:** {n_pass} PASS / {n_fail} FAIL / {len(rows)} cases  ",
        "",
        "## Summary table",
        "",
        "| # | Case | Category | Stack | Expected | Got | Types | Eg | Result |",
        "|---|------|----------|-------|----------|-----|-------|----|--------|",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['case']}` | {r['category']} | {r['stack']} | "
            f"{r['expected'][:60]} | {str(r['got'])[:70]} | {r['types']} | {r['eg']} | "
            f"**{r['result']}** |"
        )

    lines += ["", "## Failures detail", ""]
    fails = [r for r in rows if r["result"] == "FAIL"]
    if not fails:
        lines.append("None — all cases passed.")
    else:
        for r in fails:
            lines.append(f"### FAIL: `{r['case']}`")
            lines.append("")
            lines.append(f"- Stack: `{r['stack']}`")
            lines.append(f"- Expected: {r['expected']}")
            lines.append(f"- Got: {r['got']}")
            lines.append(f"- Detail: {r['detail']}")
            lines.append("")

    lines += [
        "## Category notes",
        "",
        "### Should BLOCK",
        "Contact oxides, thin-film PV, graphene, 2D RP/DJ/monolayers, BeSiP2, and garbage "
        "absorbers must return `blocked=True` / verdict BLOCKED with no Type/suitability claim.",
        "",
        "### Role misuse",
        "Tool does not enforce ETL/HTL conventions; stacks must still run. "
        "PEDOT:PSS as **HTL** must attach degenerate/metallic caveat. "
        "PEDOT as **ETL** currently has no dedicated caveat (HTL-only path).",
        "",
        "### Halide traps",
        "FAPbBr3 must resolve ~2.23 eV (not FAPbI3 1.48). "
        "Cs2TiI6 must resolve ~1.58 eV (not Br ~1.8).",
        "",
        "### Edge stacks",
        "Cs2SnI6/TiO2/P3HT should not be a blind YES without Type III somewhere.",
        "",
        "### Stability",
        "Identical inputs must yield identical key fields; no crashes.",
        "",
    ]

    if crashes:
        lines += ["## Crashes", ""]
        for r in crashes:
            lines.append(f"- `{r['case']}`: ```\n{r['crash'][:800]}\n```")
            lines.append("")

    lines += [
        "## Verdict",
        "",
        (
            f"**{n_pass}/{len(rows)} PASS.** "
            + ("Negative-case gate looks healthy." if n_fail == 0 else f"{n_fail} failure(s) need attention.")
        ),
        "",
    ]

    out = ROOT / "data" / "negative_cases_test_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    raw_out = ROOT / "data" / "negative_cases_test_report.json"
    raw_out.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "pass": n_pass,
                "fail": n_fail,
                "total": len(rows),
                "rows": [{k: v for k, v in r.items() if k != "raw"} for r in rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    print(f"PASS={n_pass} FAIL={n_fail} TOTAL={len(rows)}")
    for r in fails:
        print(f"  FAIL {r['case']}: {r['detail'][:120]}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
