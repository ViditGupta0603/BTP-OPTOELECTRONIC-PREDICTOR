OptoStack — how to present this package to your professor
==========================================================

Preferred reading order (GitHub / VS Code / zip share)
------------------------------------------------------
1. docs/REPORT_README.txt          ← you are here (presentation guide)
2. docs/OPTOSTACK_FULL_REPORT.md   ← main document to show / print / share
3. docs/report_figures/            ← PNGs embedded by the full report
4. data/cross_validation_report.md ← optional deep-dive numbers (same CV as report §4)
5. data/perovskite_test_set_literature_accuracy_report.md
                                   ← optional literature Eg holdout detail
6. docs/TOOL_FLOWCHART.md          ← optional Mermaid source (also inlined in full report)
7. docs/TOOL_WORKFLOW.md           ← optional operator checklist

What to say in 2 minutes
------------------------
• OptoStack screens perovskite absorber + ETL + HTL for Anderson junction
  Type (I / II / III) and suitability (YES / MARGINAL / NO).
• It does NOT predict PCE, stability, or fabrication yield.
• Cross-validation scorecard (GroupKFold): absorber Eg R² ≈ 0.93;
  Type HTL accuracy ≈ 0.83; Type ETL accuracy ≈ 0.67 (harder leave-absorber-out).
• Suitability is a deterministic rule on the two interface Types — not an ML model.
• Example: MAPbI3 / TiO2 / MoO3 → Type II + Type III → MARGINAL
  (MoO3 Type III can be intentional “broken gap by design”).

How to share offline
--------------------
Zip at least:
  docs/OPTOSTACK_FULL_REPORT.md
  docs/REPORT_README.txt
  docs/report_figures/

GitHub: open OPTOSTACK_FULL_REPORT.md — Mermaid diagrams render on GitHub;
images use relative paths under docs/report_figures/.

Do not commit .env (API keys). Runtime demo:
  python scripts/predict_stack.py --absorber MAPbI3 --etl TiO2 --htl MoO3
  python app.py   → http://127.0.0.1:7860
