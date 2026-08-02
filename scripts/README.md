# Scripts

Project overview: [README.md](../README.md)

| Script | Purpose |
|--------|---------|
| `predict_stack.py` | **Main pipeline** — train + predict Type (LLM only with `--llm`) |
| `enrich_chi_dataset.py` | χ fill + ETL/HTL libraries + layer properties |
| `perovskite_rules.py` | Family taxonomy, Vegard priors, perovskite gate helpers |
| `formula_estimator.py` | Blend family rules + ML for unknown formulas |
| `cross_validate_models.py` | GroupKFold / StratifiedKFold CV + figures → `data/cross_validation_report.md` (+ `docs/report_figures/`) |
| `eval_literature_test_set.py` | Literature test-set Eg accuracy |
| `evaluate_models.py` | Train/test split metrics → `data/model_eval_report.json` |
| `iterative_accuracy_loop.py` | Optional: merge verified data → retrain → score Type/Eg |
| `eval_browser_random_test.py` | Optional: browser random perovskite Eg accuracy |
| `benchmark_predictions.py` | Optional: ML vs literature benchmark dumps |
| `llm_literature_assist.py` | Standalone LLM Eg/χ (optional) |
| `literature_bands.py` | CBO/VBO / Type / suitability from Eg+χ |
| `formula_parse.py` | FA/MA/organic cation + parenthesis formula parsing |
| `build_perovskite_dataset.py` | Build expanded perovskite stacks |
| `verify_perovskite_dataset.py` | Verify perovskite datasets |
| `build_literature_dataset.py` | Full opto literature master (optional) |

See [docs/TOOL_WORKFLOW.md](../docs/TOOL_WORKFLOW.md) (operators) and [docs/TECHNICAL_WORKFLOW.md](../docs/TECHNICAL_WORKFLOW.md) (engineering). Index: [docs/WORKFLOW.md](../docs/WORKFLOW.md).
