import json
import os
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("llm")


class LLMClient:
    """Talks to GPT-5 through Azure AI Inference. Every intelligence agent uses this."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            token = settings.GITHUB_TOKEN or os.environ.get("GITHUB_TOKEN", "")
            if not token:
                raise RuntimeError(
                    "GITHUB_TOKEN not set — the AI agents need it to think. "
                    "Set it in .env or as an environment variable."
                )

            self._client = ChatCompletionsClient(
                endpoint=settings.LLM_ENDPOINT,
                credential=AzureKeyCredential(token),
            )
            logger.info(f"LLM client ready: {settings.LLM_MODEL}")

        return self._client

    def ask(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 4096) -> str:
        from azure.ai.inference.models import SystemMessage, UserMessage

        logger.debug(f"LLM request: {user_prompt[:100]}...")

        response = self.client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            model=settings.LLM_MODEL,
            temperature=temperature,
            model_extras={"max_completion_tokens": max_tokens},
        )

        result = response.choices[0].message.content.strip()
        logger.debug(f"LLM response: {result[:100]}...")
        return result

    def ask_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 4096) -> dict:
        """Same as ask() but parses the response as JSON. Handles markdown fences."""
        raw = self.ask(system_prompt, user_prompt, temperature, max_tokens)

        # strip markdown code fences if the model wrapped its response
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                cleaned = part.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM JSON: {raw[:500]}")
            raise ValueError(f"LLM returned non-JSON: {raw[:200]}")


_llm_client: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
