# CertFE: Certified Feature Engineering

CertFE is a **certified program synthesis** framework for tabular feature engineering. Instead of generating features as black-box transformations, CertFE represents features as **programs in a restricted JSON DSL** and subjects every candidate to a **six-certificate verification layer (C1–C6)** before it can be added to the feature set. A **progressive evaluator** and **semantic critic** further gate acceptance, producing a final feature set with audit-ready certificates.


## Quick Start

```python
from certfe.runner.run import run

state = run(
    dataset_name="credit",   # dataset from "PromptFE"
    R_max=50,                # max rounds
    J=5,                     # stop after J rounds without accept
    K_max=30,                # max accepted features
    epsilon_rel=0.01,        # relative improvement threshold
    model_name="RF",         # downstream model
    seed=0,
    batch_size=5,
    output_dir="./output",
)
```

This produces three files in `output_dir/`:
- `features.jsonl` — per-candidate certificate records (hard + soft layers)
- `metrics.json` — aggregate stats (|S|, baseline, final score, stop reason)
- `cpuf.json` — cumulative cost per accepted feature

## Dependencies

- Python ≥ 3.10
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `openai` (for LLM calls; any OpenAI-compatible endpoint)
- `python-dotenv`
- `lightgbm` (optional, for LGB downstream model)
- PromptFE (for dataset loading and evaluation metrics)

## Configuration

Set environment variables or create a `.env` file:

```
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://your-endpoint/v1
```

## License

TBD
