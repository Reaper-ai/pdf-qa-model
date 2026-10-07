"""OpenAI/HF-InferenceClient-shaped adapter for a local Ollama model.

Exposes `.chat_completion(messages=..., max_tokens=..., temperature=...)` so it
is a drop-in replacement for huggingface_hub.InferenceClient inside
qa_model.py and evaluator.py — no call-site changes required.

Uses Ollama's native /api/chat with think=False so reasoning models
(e.g. gemma4:e2b) return the answer directly instead of spending the
max_tokens budget on hidden reasoning tokens.
"""
import json
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        model: str = "gemma4:e2b",
        host: str = "http://localhost:11434",
        timeout: int = 180,
        max_retries: int = 3,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def chat_completion(
        self,
        messages,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: Optional[float] = None,
        stop=None,
        **kwargs,
    ) -> SimpleNamespace:
        payload = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **({"top_p": top_p} if top_p is not None else {}),
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        data = json.dumps(payload).encode("utf-8")
        last_err = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.load(resp)
                content = body.get("message", {}).get("content", "") or ""
                finish = body.get("done_reason", "stop")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content, role="assistant"),
                            finish_reason=finish,
                        )
                    ],
                    model=self.model,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                last_err = exc
                time.sleep(0.5 * (2 ** attempt))

        raise RuntimeError(f"Ollama call failed after {self.max_retries} attempts: {last_err}")
