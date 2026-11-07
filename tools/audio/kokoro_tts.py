"""Kokoro-82M local text-to-speech provider.

Speech is synthesized by ``experiments/kokoro/generate_speech.py`` under the
isolated interpreter at ``experiments/kokoro/.venv``. This module never imports
Kokoro into the OpenMontage environment.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class KokoroTTS(BaseTool):
    name = "kokoro_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "kokoro"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.LOCAL

    install_instructions = (
        "Kokoro TTS uses its own virtualenv and must not be installed into "
        "the OpenMontage environment.\n"
        "  Interpreter: experiments/kokoro/.venv/Scripts/python.exe\n"
        "  Worker: experiments/kokoro/generate_speech.py\n"
        "The validated voice is pm_alex (Brazilian Portuguese, lang_code p, 24 kHz WAV)."
    )
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "offline_generation",
    ]
    supports = {
        "voice_cloning": False,
        "multilingual": False,
        "brazilian_portuguese": True,
        "offline": True,
        "native_audio": True,
    }
    best_for = [
        "Brazilian Portuguese narration",
        "offline local speech synthesis",
        "privacy-sensitive local-only workflows",
    ]
    not_good_for = [
        "English-first narration",
        "voice clone matching",
        "languages other than Brazilian Portuguese",
    ]

    DEFAULT_VOICE = "pm_alex"
    DEFAULT_LANG = "p"
    DEFAULT_SPEED = 1.0
    DEFAULT_MODEL = "kokoro-82m"
    SAMPLE_RATE = 24000
    timeout_seconds = 600

    _PT_BR_LANGS = {
        "p",
        "pt",
        "pt-br",
        "pt_br",
        "portuguese",
        "portugues",
        "português",
    }

    input_schema = {
        "type": "object",
        "required": ["text", "output_path"],
        "properties": {
            "text": {
                "type": "string",
                "minLength": 1,
                "description": "Narration text. Brazilian Portuguese is the supported language.",
            },
            "voice": {
                "type": "string",
                "minLength": 1,
                "default": "pm_alex",
                "description": "Kokoro voice id. pm_alex is the validated Brazilian Portuguese voice.",
            },
            "speed": {
                "type": "number",
                "minimum": 0.5,
                "maximum": 2.0,
                "default": 1.0,
                "description": "Speaking-rate multiplier. 1.0 is the validated default.",
            },
            "lang_code": {
                "type": "string",
                "enum": ["p"],
                "default": "p",
                "description": "Kokoro language code. p is Brazilian Portuguese.",
            },
            "output_path": {
                "type": "string",
                "minLength": 1,
                "description": "Destination WAV path. The file is written at 24 kHz.",
            },
            "model": {
                "type": "string",
                "enum": ["kokoro-82m"],
                "default": "kokoro-82m",
            },
        },
    }

    output_schema = {
        "type": "object",
        "required": [
            "provider",
            "model",
            "voice",
            "output",
            "format",
            "sample_rate",
            "audio_duration_seconds",
            "generation_seconds",
        ],
        "properties": {
            "provider": {"type": "string"},
            "model": {"type": "string"},
            "voice": {"type": "string"},
            "lang_code": {"type": "string"},
            "speed": {"type": "number"},
            "text_length": {"type": "integer"},
            "output": {"type": "string"},
            "format": {"type": "string"},
            "sample_rate": {"type": "integer"},
            "audio_duration_seconds": {"type": "number"},
            "generation_seconds": {"type": "number"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=4, ram_mb=2048, vram_mb=0, disk_mb=500, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=[])
    idempotency_key_fields = ["text", "voice", "speed", "lang_code", "model"]
    side_effects = ["writes a 24 kHz WAV file to output_path"]
    fallback = "piper_tts"
    fallback_tools = ["piper_tts"]
    user_visible_verification = [
        "Listen to the WAV for Brazilian Portuguese pronunciation and pacing"
    ]

    def python_executable(self) -> Path:
        """Return the isolated Kokoro interpreter, never the project interpreter."""
        root = self.kokoro_root() / ".venv"
        if platform.system() == "Windows":
            return root / "Scripts" / "python.exe"
        return root / "bin" / "python"

    def worker_script(self) -> Path:
        return self.kokoro_root() / "generate_speech.py"

    def kokoro_root(self) -> Path:
        return Path(__file__).resolve().parents[2] / "experiments" / "kokoro"

    def get_status(self) -> ToolStatus:
        python = self.python_executable()
        worker = self.worker_script()
        if python.is_file() and worker.is_file():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        try:
            normalized = self._normalize_inputs(inputs)
        except ValueError as exc:
            return ToolResult(success=False, error=f"Invalid Kokoro TTS inputs: {exc}")

        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error="Kokoro TTS not available. " + self.install_instructions,
            )

        try:
            result = self._generate(normalized)
        except subprocess.TimeoutExpired:
            result = ToolResult(
                success=False,
                error=f"Kokoro TTS timed out after {self.timeout_seconds}s",
            )
        except Exception as exc:
            result = ToolResult(
                success=False,
                error=f"Local TTS generation failed: {exc}",
            )

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = 0.0
        return result

    def _normalize_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be an object")

        if "text" not in inputs or inputs.get("text") is None:
            raise ValueError("text is required")
        if not isinstance(inputs["text"], str):
            raise ValueError("text must be a string")
        if not inputs["text"].strip():
            raise ValueError("text must not be empty")

        if "output_path" not in inputs or inputs.get("output_path") is None:
            raise ValueError("output_path is required")
        if not isinstance(inputs["output_path"], str) or not inputs["output_path"].strip():
            raise ValueError("output_path is required")
        output_path = Path(inputs["output_path"].strip())
        if output_path.suffix.lower() != ".wav":
            raise ValueError("output_path must use a .wav extension")
        if output_path.exists() and output_path.is_dir():
            raise ValueError("output_path must be a file path")

        voice = self._one_voice(inputs.get("voice"), inputs.get("voice_id"))
        speed = self._one_speed(inputs)
        lang_code = self._one_lang(inputs.get("lang_code"), inputs.get("language_code"))
        model = inputs.get("model", self.DEFAULT_MODEL)
        if model != self.DEFAULT_MODEL:
            raise ValueError(
                f"Unsupported model {model!r}. Kokoro TTS uses {self.DEFAULT_MODEL}."
            )

        normalized = {
            "text": inputs["text"],
            "voice": voice,
            "speed": speed,
            "lang_code": lang_code,
            "output_path": str(output_path),
            "model": model,
        }
        self._validate_schema(normalized)
        return normalized

    def _one_voice(self, voice: Any, voice_id: Any) -> str:
        for label, value in (("voice", voice), ("voice_id", voice_id)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{label} must be a string")
        voice_text = voice.strip() if isinstance(voice, str) else ""
        voice_id_text = voice_id.strip() if isinstance(voice_id, str) else ""
        if voice_text and voice_id_text and voice_text != voice_id_text:
            raise ValueError("voice and voice_id disagree")
        return voice_text or voice_id_text or self.DEFAULT_VOICE

    def _one_speed(self, inputs: dict[str, Any]) -> float:
        present = [
            (name, inputs[name])
            for name in ("speed", "speaking_rate", "voice_speed")
            if name in inputs and inputs[name] is not None
        ]
        if not present:
            return self.DEFAULT_SPEED
        values: list[float] = []
        for name, value in present:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number between 0.5 and 2.0")
            values.append(float(value))
        if any(value != values[0] for value in values[1:]):
            raise ValueError("speed controls disagree")
        return values[0]

    def _one_lang(self, lang_code: Any, language_code: Any) -> str:
        supplied = [value for value in (lang_code, language_code) if value not in (None, "")]
        if not supplied:
            return self.DEFAULT_LANG
        normalized = [self._normalize_lang(value) for value in supplied]
        if any(value != normalized[0] for value in normalized[1:]):
            raise ValueError("lang_code and language_code disagree")
        return normalized[0]

    def _normalize_lang(self, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("lang_code must be a string")
        token = value.strip().lower()
        if token in self._PT_BR_LANGS:
            return self.DEFAULT_LANG
        raise ValueError(
            f"Unsupported lang_code {value!r}. "
            "Kokoro TTS supports Brazilian Portuguese (lang_code 'p')."
        )

    def _validate_schema(self, normalized: dict[str, Any]) -> None:
        from jsonschema import validate
        from jsonschema.exceptions import ValidationError

        try:
            validate(normalized, self.input_schema)
        except ValidationError as exc:
            raise ValueError(exc.message) from exc

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        output_path = Path(inputs["output_path"]).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": inputs["text"],
            "voice": inputs["voice"],
            "speed": inputs["speed"],
            "lang_code": inputs["lang_code"],
            "output_path": str(output_path),
        }
        proc = subprocess.run(
            [str(self.python_executable()), str(self.worker_script())],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            cwd=str(self.kokoro_root()),
            env=self._subprocess_env(),
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return ToolResult(
                success=False,
                error=f"Kokoro TTS failed (exit {proc.returncode}): {detail}",
            )
        if not output_path.is_file():
            return ToolResult(
                success=False,
                error=f"Kokoro output file missing: {output_path}",
            )
        if output_path.stat().st_size < 44:
            return ToolResult(
                success=False,
                error=f"Kokoro output file is empty: {output_path}",
            )

        metadata = self._parse_worker_metadata(proc.stdout)
        sample_rate = metadata.get("sample_rate")
        if sample_rate != self.SAMPLE_RATE:
            return ToolResult(
                success=False,
                error=f"Kokoro returned sample rate {sample_rate}, expected {self.SAMPLE_RATE}",
            )
        try:
            audio_duration = float(metadata["audio_duration_seconds"])
            generation_seconds = float(metadata["generation_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Kokoro metadata is incomplete: {exc}") from exc
        if audio_duration < 0 or generation_seconds < 0:
            raise ValueError("Kokoro metadata durations must be non-negative")

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": self.DEFAULT_MODEL,
                "voice": inputs["voice"],
                "lang_code": inputs["lang_code"],
                "speed": inputs["speed"],
                "text_length": len(inputs["text"]),
                "output": str(output_path),
                "format": "wav",
                "sample_rate": self.SAMPLE_RATE,
                "audio_duration_seconds": audio_duration,
                "generation_seconds": generation_seconds,
            },
            artifacts=[str(output_path)],
            model=self.DEFAULT_MODEL,
        )

    def _subprocess_env(self) -> dict[str, str]:
        """Give the worker the Kokoro venv without the project PYTHONPATH."""
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["VIRTUAL_ENV"] = str(self.kokoro_root() / ".venv")
        return env

    @staticmethod
    def _parse_worker_metadata(stdout: str) -> dict[str, Any]:
        for line in reversed((stdout or "").splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("Kokoro worker did not return JSON metadata")
