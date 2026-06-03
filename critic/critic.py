"""CertFE Semantic Critic — two-call protocol for true-blind interpretation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from certfe.generator.llm_client import call_llm, CriticConfig

logger = logging.getLogger(__name__)


@dataclass
class SemanticResult:
    """Output of the two-call critic protocol."""
    plausibility: str  # "plausible" | "questionable" | "unclassifiable"
    interpretation: str
    plausibility_witness: str
    intent_alignment: str  # "match" | "partial" | "mismatch"
    alignment_witness: str
    total_tokens: int = 0


CALL1_SYSTEM = """You are a feature engineering reviewer. Given a feature program (JSON DSL) and column descriptions, you must:
1. Write a one-sentence interpretation of what this program computes.
2. Judge its plausibility: is this a meaningful feature or nonsensical?

The author's stated intent is NOT shown. Judge purely from the program structure.

Respond with ONLY a JSON object:
{
  "interpretation": "<one sentence describing what this program computes>",
  "plausibility": "plausible" | "questionable" | "unclassifiable",
  "plausibility_witness": "<brief reason for your plausibility judgment>"
}"""

CALL2_SYSTEM = """You are a feature engineering reviewer. You are shown:
1. An independent interpretation of a feature program (written without seeing the author's intent).
2. The author's declared intent for that program.

Judge whether the independent interpretation and the declared intent refer to the same computation/concept.

Respond with ONLY a JSON object:
{
  "intent_alignment": "match" | "partial" | "mismatch",
  "alignment_witness": "<brief explanation of your judgment>"
}"""


def critique(
    program_json: dict,
    column_descriptions: dict[int, str],
    declared_intent: str,
    config: CriticConfig | None = None,
) -> SemanticResult:
    """Run the two-call critic protocol.

    Call 1: interpretation + plausibility (blind to declared_intent)
    Call 2: intent_alignment (sees interpretation + declared_intent)
    """
    cfg = config or CriticConfig()

    col_desc_text = "\n".join(
        f"  col_index={i}: {desc}" for i, desc in sorted(column_descriptions.items())
    )
    program_text = json.dumps(program_json, indent=2)

    # --- Call 1: interpretation + plausibility (blind) ---
    call1_user = f"""## Column descriptions
{col_desc_text}

## Feature program
{program_text}

Write your interpretation and plausibility judgment. Respond with ONLY JSON."""

    resp1 = call_llm(CALL1_SYSTEM, call1_user, config=cfg)
    interpretation, plausibility, plausibility_witness = _parse_call1(resp1.text)
    tokens = resp1.total_tokens

    # --- Call 2: intent_alignment ---
    call2_user = f"""## Independent interpretation (written without seeing the author's intent)
"{interpretation}"

## Author's declared intent
"{declared_intent}"

Judge alignment. Respond with ONLY JSON."""

    resp2 = call_llm(CALL2_SYSTEM, call2_user, config=cfg)
    intent_alignment, alignment_witness = _parse_call2(resp2.text)
    tokens += resp2.total_tokens

    return SemanticResult(
        plausibility=plausibility,
        interpretation=interpretation,
        plausibility_witness=plausibility_witness,
        intent_alignment=intent_alignment,
        alignment_witness=alignment_witness,
        total_tokens=tokens,
    )


def _parse_call1(text: str) -> tuple[str, str, str]:
    """Parse Call 1 response → (interpretation, plausibility, witness)."""
    try:
        data = json.loads(_strip_fences(text))
        interpretation = data.get("interpretation", "")
        plausibility = data.get("plausibility", "unclassifiable")
        if plausibility not in ("plausible", "questionable", "unclassifiable"):
            plausibility = "unclassifiable"
        witness = data.get("plausibility_witness", "")
        return interpretation, plausibility, witness
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Critic Call 1 parse failed")
        return "", "unclassifiable", "critic parse failure"


def _parse_call2(text: str) -> tuple[str, str]:
    """Parse Call 2 response → (intent_alignment, witness)."""
    try:
        data = json.loads(_strip_fences(text))
        alignment = data.get("intent_alignment", "unclassifiable")
        if alignment not in ("match", "partial", "mismatch"):
            alignment = "unclassifiable"
        witness = data.get("alignment_witness", "")
        return alignment, witness
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Critic Call 2 parse failed")
        return "unclassifiable", "critic parse failure"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text
