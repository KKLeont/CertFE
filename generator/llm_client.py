"""LLM client for CertFE generator and critic."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(_env_path)

_API_KEY = os.getenv("OPENAI_API_KEY")
if not _API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


@dataclass
class GeneratorConfig:
    model: str = "deepseek-v4-flash"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 60.0


@dataclass
class CriticConfig:
    model: str = "deepseek-v4-flash"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 60.0


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _build_client() -> OpenAI:
    return OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    config: GeneratorConfig | CriticConfig | None = None,
    max_retries: int = 2,
) -> LLMResponse:
    """LLM call with basic retry on failure."""
    cfg = config or GeneratorConfig()
    client = _build_client()

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            usage = response.usage
            return LLMResponse(
                text=response.choices[0].message.content or "",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                import time
                time.sleep(1.0 * (attempt + 1))
                continue
            break

    return LLMResponse(text="", prompt_tokens=0, completion_tokens=0)
