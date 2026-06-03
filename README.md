# CertFE: Certified Feature Engineering

CertFE is a **certified program synthesis** framework for tabular feature engineering. Instead of generating features as black-box transformations, CertFE represents features as **programs in a restricted JSON DSL** and subjects every candidate to a **six-certificate verification layer (C1–C6)** before it can be added to the feature set. A **progressive evaluator** and **semantic critic** further gate acceptance, producing a final feature set with audit-ready certificates.

## Overview

| Component | Role |
|-----------|------|
| **Generator** (LLM) | Proposes batches of feature programs conditioned on a family-aware coverage map |
| **Verifier** (C1–C6) | Rejects programs that violate schema, leak labels, misuse types/units, or break reproducibility |
| **Critic** (LLM) | Two-call blind protocol: interprets the program, then judges alignment with declared intent |
| **Evaluator** | 4-tier progressive pipeline (sanity → small-sample → single-split → 5-fold CV) |
| **Redundancy** | Dual-channel dedup: Spearman rank cosine (value) + canonical hash (structure) |
| **Runner** | Minimal agent loop with yield-aware stopping (R_max, J-consecutive-no-accept, K_max) |

## Certificate Layer (C1–C6)

Every feature program is compiled into a lineage DAG, then checked:

| Cert | Name | What it catches |
|------|------|-----------------|
| C1 | Schema Validity | Unknown columns, arity mismatches |
| C2 | No-Leakage | Target column or forbidden columns in feature computation |
| C3 | Temporal Validity | Future-time leakage (n/a on IID datasets) |
| C4 | Type/Unit Safety | Numeric ops on categoricals, unit mismatches in Add/Subtract |
| C5 | Train-Only Fitting | Fit-ops (ZScore, TargetEncode) seeing val/test data |
| C6 | Lineage/Reproducibility | DAG cycles, dangling references, non-determinism |

## Feature DSL

Features are expressed as JSON programs over a vocabulary of ~30 operators:

- **Leaf**: `Col`, `Const`
- **Unary (numeric)**: `Log1p`, `Sqrt`, `Abs`, `Square`, `Reciprocal`
- **Fit (train-only)**: `ZScore`, `MinMaxScale`, `QuantileBucketize`, `FrequencyEncode`, `TargetEncode`
- **Binary**: `Add`, `Subtract`, `Multiply`, `Divide`, `Ratio`, `Cross`, `IsEqual`, `Coalesce`
- **Variadic**: `RowMean`, `RowMax`, `RowMin`, `RowSum`, `RowStd`
- **Missing**: `IsNull`

Operators are organized into 8 IID-applicable feature families (nonlinear_transform, normalization, binning, arithmetic, rowwise_stats, categorical_encoding, categorical_interaction, missingness).

## Project Structure

```
CertFE/
├── dsl/                  # DSL: AST, grammar, compiler, executor, operators, types, lineage
│   ├── ast.py            # AST node types (ColRef, UnaryOp, BinaryOp, VariadicOp, ...)
│   ├── grammar.py        # JSON ↔ AST parser/serializer
│   ├── compile.py        # Type-checker + lineage graph builder
│   ├── execute.py        # Runtime evaluator with fit-on-train semantics
│   ├── operators.py      # Operator registry with signatures
│   ├── types.py          # DataType, Unit, type derivation rules
│   ├── family.py         # FeatureFamily enum + IID/temporal/relational scopes
│   └── lineage.py        # Lineage DAG for certificate verification
├── generator/            # LLM-driven feature program proposer
│   ├── proposer.py       # Batch proposer with coverage-aware prompting
│   ├── prompt.py         # System/user prompt templates
│   ├── coverage.py       # Family × col-group coverage map
│   └── llm_client.py     # OpenAI-compatible LLM client
├── verifier/             # Six formal certificates (C1–C6)
│   ├── base.py           # CertTrace, CertResult, Verdict, FailureCode
│   ├── policy.py         # CertificatePolicy configuration
│   ├── schema_validity.py  # C1
│   ├── no_leakage.py       # C2
│   ├── temporal_validity.py  # C3
│   ├── type_unit_safety.py   # C4
│   ├── train_only_fitting.py # C5
│   └── lineage_repr.py       # C6
├── critic/               # Semantic critic (two-call blind protocol)
│   └── critic.py
├── evaluator/            # Progressive evaluation pipeline
│   ├── adapter.py        # Dataset loading + downstream model evaluation
│   └── tiers.py          # Tier1→Tier2→Tier3→Tier4
├── redundancy/           # Value + structural redundancy detection
│   └── redundancy.py
├── runner/               # Main agent loop
│   └── run.py
└── selector/             # (deferred to future work)
```

## Quick Start

```python
from certfe.runner.run import run

state = run(
    dataset_name="credit",   # dataset from PromptFE/data/
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

## Citation

```bibtex
@misc{certfe,
  title={CertFE: Certified Feature Engineering via Program Synthesis},
  author={},
  year={2026},
}
```

## License

TBD
