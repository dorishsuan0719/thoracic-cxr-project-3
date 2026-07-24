"""Local-only Ollama service for Full-image multilabel research explanations."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from full_image_multilabel_report_prompt import (
    DISCLAIMER,
    NO_POSITIVE_MESSAGE,
    REQUIRED_HEADINGS,
    build_report_messages,
)


class FullImageOllamaError(RuntimeError):
    pass


class FullImageMultilabelOllamaService:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        requested_model: str = "gemma3:4b",
        timeout_seconds: float = 120.0,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url.rstrip("/"))
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Ollama must use a local HTTP endpoint")
        self.base_url = base_url.rstrip("/")
        self.requested_model = requested_model.strip() or "gemma3:4b"
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive")
        self.selected_model: str | None = None
        self.available_models: list[str] = []
        self.health_seconds: float | None = None
        self._last_health_error: str | None = None

    def _request_json(
        self, method: str, endpoint: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                if response.status != 200:
                    raise FullImageOllamaError(f"Ollama returned HTTP {response.status}")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise FullImageOllamaError("Ollama response root is not a JSON object")
                return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise FullImageOllamaError(
                f"Ollama request failed: {type(exc).__name__}: {exc}"
            ) from exc

    def health(self) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            payload = self._request_json("GET", "/api/tags")
            models = payload.get("models")
            if not isinstance(models, list):
                raise FullImageOllamaError("Ollama /api/tags did not return a models list")
            self.available_models = sorted(
                {
                    str(item.get("name") or item.get("model"))
                    for item in models
                    if isinstance(item, dict) and (item.get("name") or item.get("model"))
                }
            )
            self.health_seconds = time.perf_counter() - started
            if self.requested_model not in self.available_models:
                raise FullImageOllamaError(
                    f"Requested local model is not installed: {self.requested_model}; "
                    f"available={self.available_models}"
                )
            self.selected_model = self.requested_model
            self._last_health_error = None
            return {
                "status": "PASS",
                "backend": self.base_url,
                "model": self.selected_model,
                "available_models": self.available_models,
                "health_seconds": self.health_seconds,
            }
        except Exception as exc:
            self.health_seconds = time.perf_counter() - started
            self._last_health_error = f"{type(exc).__name__}: {exc}"
            return {
                "status": "UNAVAILABLE",
                "backend": self.base_url,
                "model": self.requested_model,
                "available_models": self.available_models,
                "health_seconds": self.health_seconds,
                "error": self._last_health_error,
            }

    def require_available(self) -> str:
        health = self.health()
        if health["status"] != "PASS" or not self.selected_model:
            raise FullImageOllamaError(health.get("error", "Ollama is unavailable"))
        return self.selected_model

    @staticmethod
    def validate_response(
        text: str, predicted_class_ids: list[int], predicted_class_names_zh: list[str]
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise FullImageOllamaError("Ollama returned an empty explanation")
        if len(text) > 14000:
            raise FullImageOllamaError("Ollama explanation exceeds the accepted length")
        missing_headings = [heading for heading in REQUIRED_HEADINGS if heading not in text]
        if missing_headings:
            raise FullImageOllamaError(f"Missing required report headings: {missing_headings}")
        forbidden = [
            term
            for term in ("No finding", "Normal", "Background", "Unknown", "Ground Truth", "TP/FP/FN/TN")
            if term in text
        ]
        if forbidden:
            raise FullImageOllamaError(f"Ollama introduced forbidden content: {forbidden}")
        if DISCLAIMER not in text:
            raise FullImageOllamaError("Ollama did not preserve the required disclaimer")
        if predicted_class_ids:
            missing_names = [name for name in predicted_class_names_zh if name not in text]
            if missing_names:
                raise FullImageOllamaError(
                    f"Ollama did not preserve predicted positive class names: {missing_names}"
                )
        elif NO_POSITIVE_MESSAGE not in text:
            raise FullImageOllamaError("Ollama did not preserve the no-positive threshold message")
        return {
            "status": "PASS",
            "required_headings_present": True,
            "forbidden_content": [],
            "predicted_classes_preserved": True,
            "disclaimer_preserved": True,
        }

    def generate(self, structured_result: dict[str, Any], patient_info: dict[str, Any]) -> dict[str, Any]:
        model = self.require_available()
        allowed_keys = {
            "class_probabilities",
            "class_thresholds",
            "predicted_class_ids",
            "predicted_class_names_en",
            "predicted_class_names_zh",
            "preprocessing_description",
            "model_name",
            "disclaimer",
        }
        if set(structured_result) != allowed_keys:
            raise FullImageOllamaError(
                f"Unexpected Ollama structured fields: {sorted(set(structured_result) - allowed_keys)}"
            )
        messages, prompt_hash = build_report_messages(
            patient_info=patient_info,
            **structured_result,
        )
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.1},
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        retry_count = 0
        for attempt in range(2):
            try:
                response = self._request_json("POST", "/api/chat", payload)
                message = response.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise FullImageOllamaError("Ollama response has no message.content")
                text = message["content"].strip()
                validation = self.validate_response(
                    text,
                    structured_result["predicted_class_ids"],
                    structured_result["predicted_class_names_zh"],
                )
                return {
                    "status": "PASS",
                    "backend": self.base_url,
                    "model": model,
                    "prompt_sha256": prompt_hash,
                    "response": text,
                    "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "generation_seconds": time.perf_counter() - started,
                    "retry_count": retry_count,
                    "validation": validation,
                    "image_sent_to_ollama": False,
                    "ground_truth_sent_to_ollama": False,
                }
            except FullImageOllamaError as exc:
                last_error = exc
                if attempt == 0:
                    retry_count = 1
                    time.sleep(0.4)
                    continue
                break
        raise FullImageOllamaError(str(last_error) if last_error else "Unknown Ollama failure")
