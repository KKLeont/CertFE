"""Coverage-aware batch feature program proposer (Plan B)."""
from __future__ import annotations

import json
import logging

from certfe.dsl.grammar import program_from_json, GrammarError
from certfe.dsl.ast import FeatureProgram

from .llm_client import call_llm, GeneratorConfig
from .coverage import CoverageMap
from .prompt import build_system_prompt, build_user_prompt, build_coverage_context

logger = logging.getLogger(__name__)


class ProposalResult:
    """Result of a single propose() call."""

    def __init__(self):
        self.programs: list[FeatureProgram] = []
        self.declared_intents: dict[str, str] = {}
        self.attempted_total: int = 0
        self.parse_failures: int = 0
        self.token_usage: int = 0


class Proposer:
    """Family-conditioned, coverage-aware batch feature program proposer."""

    def __init__(
        self,
        dataset_name: str,
        column_info: dict[int, dict],
        target_col: int,
        task: str,
        llm_config: GeneratorConfig | None = None,
    ):
        self.dataset_name = dataset_name
        self.column_info = column_info
        self.target_col = target_col
        self.task = task
        self.llm_config = llm_config or GeneratorConfig()
        self.system_prompt = build_system_prompt(
            dataset_name=dataset_name,
            column_info=column_info,
            target_col=target_col,
            task=task,
        )
        self._recent_rejections: list[dict] = []

    def add_rejection(self, feature_id: str, reason: str):
        self._recent_rejections.append({"feature_id": feature_id, "reason": reason})
        if len(self._recent_rejections) > 20:
            self._recent_rejections = self._recent_rejections[-20:]

    def propose(
        self,
        coverage_map: CoverageMap,
        accepted_features: list[str] | None = None,
        batch_size: int = 5,
    ) -> ProposalResult:
        """Generate a batch of feature programs."""
        result = ProposalResult()

        all_cols = set(self.column_info.keys())
        coverage_context = build_coverage_context(coverage_map, all_cols)
        user_prompt = build_user_prompt(
            coverage_context=coverage_context,
            current_features=accepted_features,
            recent_rejections=self._recent_rejections[-10:] if self._recent_rejections else None,
            batch_size=batch_size,
        )

        response = call_llm(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            config=self.llm_config,
        )
        result.token_usage = response.total_tokens

        if not response.text:
            logger.warning("Empty LLM response")
            return result

        items = self._parse_json_array(response.text)
        result.attempted_total = len(items)

        for item in items:
            intent = item.get("declared_intent", "")
            try:
                prog = program_from_json(item)
                result.programs.append(prog)
                result.declared_intents[prog.feature_id] = intent
            except GrammarError as e:
                result.parse_failures += 1
                logger.debug("Skipping malformed program: %s", e)

        return result

    def _parse_json_array(self, text: str) -> list[dict]:
        """Parse LLM response into a list of dicts."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            return []

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            logger.warning("LLM response is not a JSON array")
            return []

        return data
