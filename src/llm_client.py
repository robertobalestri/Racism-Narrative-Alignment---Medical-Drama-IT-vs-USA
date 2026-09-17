"""Thin Azure OpenAI client for the GPT-4.1 deployment used in the experiments."""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

from openai import AzureOpenAI

from .config import GPT41_CONFIG

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10


def extract_json(text: str) -> Dict[str, Any]:
    """Extracts the first JSON object embedded in an LLM answer."""
    without_fences = re.sub(r"```(?:json|markdown|plaintext)?", "", text).strip()
    for pattern in (r"\{.*?\}", r"\{.*\}"):
        match = re.search(pattern, without_fences, re.DOTALL)
        if not match:
            continue
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("Could not extract valid JSON", text, 0)


class GPT41Client:
    """Calls the configured GPT-4.1 deployment and returns the parsed JSON answer."""

    def __init__(self) -> None:
        missing = [
            key
            for key in ("api_key", "api_endpoint", "api_version", "deployment_name")
            if not GPT41_CONFIG.get(key)
        ]
        if missing:
            raise ValueError(
                f"Missing GPT-4.1 configuration values: {missing}. Check your .env file."
            )

        self._client = AzureOpenAI(
            azure_endpoint=GPT41_CONFIG["api_endpoint"],
            api_key=GPT41_CONFIG["api_key"],
            api_version=GPT41_CONFIG["api_version"],
        )
        self._deployment = GPT41_CONFIG["deployment_name"]
        self._temperature = float(GPT41_CONFIG["temperature"])

    def complete_json(self, prompt: str, label: str = "N/A") -> Optional[Dict[str, Any]]:
        """Returns the JSON answer for a prompt, or None if every attempt failed."""
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                )
                return extract_json(response.choices[0].message.content.strip())
            except Exception as error:
                last_error = error
                logger.warning(f"Attempt {attempt}/{MAX_ATTEMPTS} failed for {label}: {error}")
                if attempt < MAX_ATTEMPTS:
                    time.sleep(RETRY_DELAY_SECONDS)

        logger.error(f"All {MAX_ATTEMPTS} attempts failed for {label}: {last_error}")
        return None


def demo() -> None:
    parsed = extract_json('Sure!\n```json\n{"score": "+1", "explanation": "ok"}\n```')
    assert parsed == {"score": "+1", "explanation": "ok"}, parsed
    assert extract_json('noise {"a": 1} tail')["a"] == 1
    try:
        extract_json("no json here")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("expected JSONDecodeError")
    print("llm_client demo OK")


if __name__ == "__main__":
    demo()
