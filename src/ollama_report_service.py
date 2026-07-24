"""Local-only Ollama health, model selection, chat, and response validation."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from patch_roi_report_prompt import CLASS_NAME_ZH, REQUIRED_HEADINGS, build_report_messages


class OllamaServiceError(RuntimeError):
    pass


class OllamaReportService:
    def __init__(
        self,
        base_url: str,
        requested_model: str,
        timeout_seconds: float,
        project_root: Path,
    ) -> None:
        if base_url.rstrip("/") != "http://127.0.0.1:11434":
            raise ValueError("Ollama base URL must be exactly http://127.0.0.1:11434")
        self.base_url = base_url.rstrip("/")
        self.requested_model = requested_model.strip() or "auto"
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("Ollama timeout must be positive")
        self.project_root = project_root.expanduser().resolve()
        self.selected_model: str | None = None
        self.available_models: list[str] = []
        self.health_seconds: float | None = None

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
                    raise OllamaServiceError(f"Ollama returned HTTP {response.status}")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise OllamaServiceError("Ollama response root is not a JSON object")
                return parsed
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaServiceError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc

    def get_tags(self) -> dict[str, Any]:
        started = time.perf_counter()
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaServiceError("Ollama /api/tags did not return a models list")
        self.available_models = sorted(
            {
                str(item.get("name") or item.get("model"))
                for item in models
                if isinstance(item, dict) and (item.get("name") or item.get("model"))
            }
        )
        self.health_seconds = time.perf_counter() - started
        return {
            "status": "PASS",
            "base_url": self.base_url,
            "available_models": self.available_models,
            "model_count": len(self.available_models),
            "health_seconds": self.health_seconds,
        }

    def _configured_models(self) -> list[str]:
        patterns = (
            re.compile(r"OLLAMA_MODEL\s*=\s*['\"]?([^'\"\s#]+)", re.IGNORECASE),
            re.compile(r"--ollama-model\s+['\"]([^'\"]+)['\"]", re.IGNORECASE),
        )
        candidates = [
            self.project_root / "config",
            self.project_root / ".env.example",
            self.project_root / "llm_service.py",
            self.project_root / "app.py",
            self.project_root / "README.md",
        ]
        found: set[str] = set()
        files: list[Path] = []
        for candidate in candidates:
            if candidate.is_dir():
                files.extend(path for path in candidate.rglob("*") if path.is_file())
            elif candidate.is_file():
                files.append(candidate)
        for path in files:
            if path.stat().st_size > 1024 * 1024:
                continue
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                continue
            for pattern in patterns:
                found.update(match.group(1).strip() for match in pattern.finditer(text))
        return sorted(found)

    def resolve_model(self) -> str:
        if not self.available_models:
            self.get_tags()
        explicit = self.requested_model
        if explicit.lower() != "auto":
            if explicit not in self.available_models:
                raise OllamaServiceError(
                    f"Requested Ollama model is not installed locally: {explicit}. "
                    f"Available: {self.available_models}"
                )
            self.selected_model = explicit
            return explicit
        environment_model = os.environ.get("OLLAMA_MODEL", "").strip()
        if environment_model:
            if environment_model not in self.available_models:
                raise OllamaServiceError(
                    f"OLLAMA_MODEL is not installed locally: {environment_model}"
                )
            self.selected_model = environment_model
            return environment_model
        configured = [model for model in self._configured_models() if model in self.available_models]
        if len(configured) == 1:
            self.selected_model = configured[0]
            return configured[0]
        if len(configured) > 1:
            raise OllamaServiceError(
                f"Multiple project Ollama models were found; use --ollama-model explicitly: {configured}"
            )
        if len(self.available_models) == 1:
            self.selected_model = self.available_models[0]
            return self.selected_model
        if not self.available_models:
            raise OllamaServiceError(
                "No local Ollama model is installed. Install one manually, then pass --ollama-model."
            )
        raise OllamaServiceError(
            f"Multiple local Ollama models are installed; use --ollama-model explicitly: {self.available_models}"
        )

    @staticmethod
    def validate_response(text: str, prediction: dict[str, Any]) -> dict[str, Any]:
        if not text or not text.strip():
            raise OllamaServiceError("Ollama returned an empty explanation")
        if len(text) > 12000:
            raise OllamaServiceError("Ollama explanation exceeds the maximum accepted length")
        missing_headings = [heading for heading in REQUIRED_HEADINGS if heading not in text]
        if missing_headings:
            raise OllamaServiceError(f"Ollama explanation is missing required headings: {missing_headings}")
        forbidden_classes = [value for value in ("No finding", "Normal", "Background", "Unknown") if value in text]
        if forbidden_classes:
            raise OllamaServiceError(
                f"Ollama explanation introduced forbidden classes: {forbidden_classes}"
            )
        predicted_id = int(prediction["predicted_class_id"])
        expected_names = (prediction["predicted_class_name"], CLASS_NAME_ZH[predicted_id])
        if not all(name in text for name in expected_names):
            raise OllamaServiceError("Ollama explanation did not preserve the predicted class names")
        disclaimer = "本結果僅供研究與系統展示，不是臨床診斷，不能取代合格醫療專業人員的判讀。"
        if disclaimer not in text:
            raise OllamaServiceError("Ollama explanation did not preserve the required disclaimer")
        return {
            "status": "PASS",
            "missing_headings": [],
            "forbidden_classes": [],
            "predicted_class_preserved": True,
            "disclaimer_preserved": True,
        }

    def generate(self, prediction: dict[str, Any]) -> dict[str, Any]:
        model = self.resolve_model()
        messages, prompt_hash = build_report_messages(prediction)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.1},
        }
        last_error: Exception | None = None
        started = time.perf_counter()
        retry_count = 0
        for attempt in range(2):
            try:
                response = self._request_json("POST", "/api/chat", payload)
                message = response.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise OllamaServiceError("Ollama /api/chat response has no message.content")
                text = message["content"].strip()
                validation = self.validate_response(text, prediction)
                return {
                    "status": "PASS",
                    "base_url": self.base_url,
                    "model": model,
                    "prompt_sha256": prompt_hash,
                    "retry_count": retry_count,
                    "generation_seconds": time.perf_counter() - started,
                    "response": text,
                    "response_sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
                    "validation": validation,
                    "image_sent_to_ollama": False,
                }
            except OllamaServiceError as exc:
                last_error = exc
                if attempt == 0:
                    retry_count = 1
                    time.sleep(0.5)
                    continue
                break
        raise OllamaServiceError(str(last_error) if last_error else "Unknown Ollama failure")

