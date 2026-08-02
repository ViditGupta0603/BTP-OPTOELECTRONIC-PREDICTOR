# Workflow docs (index)

The full operator and engineering workflows now live in dedicated files:

| Doc | Audience | Path |
|-----|----------|------|
| **Full advisor report** | Examiner / demo | [OPTOSTACK_FULL_REPORT.md](OPTOSTACK_FULL_REPORT.md) |
| **How to present** | Sharing / zip | [REPORT_README.txt](REPORT_README.txt) |
| **Tool flowchart** | Runtime + training Mermaid | [TOOL_FLOWCHART.md](TOOL_FLOWCHART.md) |
| **Tool workflow** | Users / operators | [TOOL_WORKFLOW.md](TOOL_WORKFLOW.md) |
| **Technical workflow** | Engineering | [TECHNICAL_WORKFLOW.md](TECHNICAL_WORKFLOW.md) |
| **Development log** | History of what we built | [PROJECT_DEVELOPMENT_LOG.md](PROJECT_DEVELOPMENT_LOG.md) |
| **Datasets** | Provenance & rebuild | [DATASETS.md](DATASETS.md) |

Project overview: [README.md](../README.md)

## Quick commands (unchanged)

```bash
pip install -r requirements.txt
python scripts/enrich_chi_dataset.py
python scripts/predict_stack.py --train
python app.py   # http://127.0.0.1:7860  (binds 0.0.0.0:$PORT)
```

```bash
python scripts/predict_stack.py --absorber K2TiI6 --etl TiO2 --htl MoO3
python scripts/predict_stack.py --list-materials
# optional LLM (CLI only):
python scripts/predict_stack.py --absorber K2GeI6 --etl TiO2 --htl MoO3 --llm
```

For decision order, suitability rules, `predicted` badges, and disclaimers → [TOOL_WORKFLOW.md](TOOL_WORKFLOW.md).  
For modules, Anderson Types, artifacts, deploy → [TECHNICAL_WORKFLOW.md](TECHNICAL_WORKFLOW.md).
