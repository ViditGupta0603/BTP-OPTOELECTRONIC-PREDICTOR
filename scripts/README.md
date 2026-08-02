# Scripts

Project overview: [README.md](../README.md)

| Script | Purpose |
|--------|---------|
| `predict_stack.py` | **Main pipeline** — train + predict Type (LLM only with `--llm`) |
| `enrich_chi_dataset.py` | χ fill + ETL/HTL libraries + layer properties |
| `perovskite_rules.py` | Family taxonomy, Vegard priors, perovskite gate helpers |
| `formula_estimator.py` | Blend family rules + ML for unknown formulas |
| `iterative_accuracy_loop.py` | Merge verified data → retrain → score Type/Eg → `data/iterative_accuracy_report.md` |
| `evaluate_models.py` | Train/test split metrics → `data/model_eval_report.json` |
| `cross_validate_models.py` | GroupKFold / StratifiedKFold CV + figures → `data/cross_validation_report.md` |
| `eval_literature_test_set.py` | Literature test-set Eg accuracy |
| `eval_browser_random_test.py` | Browser random perovskite Eg accuracy |
| `benchmark_predictions.py` | ML Eg/χ vs literature → `data/perovskite_prediction_benchmark.md` |
| `llm_literature_assist.py` | Standalone LLM Eg/χ (optional) |
| `literature_bands.py` | CBO/VBO / Type / suitability from Eg+χ |
| `formula_parse.py` | FA/MA/organic cation + parenthesis formula parsing |
| `build_perovskite_dataset.py` | Build expanded perovskite stacks |
| `verify_perovskite_dataset.py` | Verify perovskite datasets |
| `build_literature_dataset.py` | Full opto literature master (optional) |

See [docs/TOOL_WORKFLOW.md](../docs/TOOL_WORKFLOW.md) (operators) and [docs/TECHNICAL_WORKFLOW.md](../docs/TECHNICAL_WORKFLOW.md) (engineering). Index: [docs/WORKFLOW.md](../docs/WORKFLOW.md).
