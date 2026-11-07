"""Kokoro-82M local TTS provider tests.

Generation is exercised with a fake subprocess. The main OpenMontage
interpreter is never asked to import Kokoro.
"""

from __future__ import annotations

import ast
import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.audio.kokoro_tts import KokoroTTS
from tools.audio.piper_tts import PiperTTS
from tools.base_tool import ToolResult, ToolRuntime, ToolStatus, ToolTier
from tools.tool_registry import ToolRegistry


WORKER = PROJECT_ROOT / "experiments" / "kokoro" / "generate_speech.py"
KOKORO_PACKAGES = {"kokoro", "soundfile", "espeakng_loader", "phonemizer", "numpy"}


def _forbid_subprocess(*_args, **_kwargs):
    raise AssertionError("subprocess should not run")


def _stage_runtime(monkeypatch, tmp_path) -> tuple[Path, Path]:
    python = tmp_path / "python.exe"
    worker = tmp_path / "generate_speech.py"
    python.write_bytes(b"")
    worker.write_text("# worker\n", encoding="utf-8")
    monkeypatch.setattr(KokoroTTS, "python_executable", lambda self: python)
    monkeypatch.setattr(KokoroTTS, "worker_script", lambda self: worker)
    return python, worker


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["kokoro"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _metadata(output_path: str, **overrides) -> dict:
    payload = {
        "output": output_path,
        "format": "wav",
        "sample_rate": 24000,
        "audio_duration_seconds": 1.25,
        "generation_seconds": 0.4,
        "voice": "pm_alex",
        "lang_code": "p",
        "speed": 1.0,
        "model": "kokoro-82m",
    }
    payload.update(overrides)
    return payload


def _write_wav(path: Path) -> None:
    path.write_bytes(b"RIFF" + b"\x00" * 60)


# ---- Contract and discovery ----


class TestDiscovery:
    def test_identity(self):
        tool = KokoroTTS()
        assert tool.name == "kokoro_tts"
        assert tool.capability == "tts"
        assert tool.provider == "kokoro"
        assert tool.runtime == ToolRuntime.LOCAL
        assert tool.tier == ToolTier.VOICE
        assert tool.fallback == "piper_tts"
        assert tool.SAMPLE_RATE == 24000
        assert tool.DEFAULT_VOICE == "pm_alex"

    def test_registry_discovers_kokoro_and_keeps_piper(self):
        registry = ToolRegistry()
        registry.discover("tools")
        kokoro = registry.get("kokoro_tts")
        piper = registry.get("piper_tts")
        assert kokoro is not None
        assert isinstance(piper, PiperTTS)
        names = [tool.name for tool in registry.get_by_capability("tts")]
        assert "kokoro_tts" in names
        assert "piper_tts" in names
        assert "kokoro_tts" in registry.get("tts_selector").fallback_tools

    def test_windows_interpreter_path(self):
        path = KokoroTTS().python_executable()
        assert path.as_posix().endswith("experiments/kokoro/.venv/Scripts/python.exe")

    def test_available_when_isolated_runtime_exists(self):
        tool = KokoroTTS()
        if not tool.python_executable().is_file() or not tool.worker_script().is_file():
            pytest.skip("Kokoro isolated environment is not installed")
        assert tool.get_status() == ToolStatus.AVAILABLE


class TestValidation:
    @pytest.mark.parametrize(
        ("inputs", "message"),
        [
            ({"output_path": "out.wav"}, "text is required"),
            ({"text": "   ", "output_path": "out.wav"}, "text must not be empty"),
            ({"text": "Olá", "output_path": "out.mp3"}, ".wav"),
            ({"text": "Olá"}, "output_path is required"),
            ({"text": "Olá", "output_path": "out.wav", "speed": 3}, "maximum"),
            ({"text": "Olá", "output_path": "out.wav", "speed": True}, "number"),
            ({"text": "Olá", "output_path": "out.wav", "voice": "a", "voice_id": "b"}, "disagree"),
            ({"text": "Olá", "output_path": "out.wav", "lang_code": "en"}, "Brazilian Portuguese"),
            (
                {"text": "Olá", "output_path": "out.wav", "speed": 1.0, "speaking_rate": 1.2},
                "disagree",
            ),
        ],
    )
    def test_rejects_invalid_inputs(self, monkeypatch, inputs, message):
        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", _forbid_subprocess)
        result = KokoroTTS().execute(inputs)
        assert result.success is False
        assert message in result.error

    def test_defaults_voice_speed_and_language(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generate(self, inputs):
            captured.update(inputs)
            return ToolResult(success=True, data={"ok": True})

        monkeypatch.setattr(KokoroTTS, "_generate", fake_generate)
        monkeypatch.setattr(KokoroTTS, "get_status", lambda self: ToolStatus.AVAILABLE)
        result = KokoroTTS().execute(
            {"text": "Olá", "output_path": str(tmp_path / "out.wav")}
        )
        assert result.success is True
        assert captured["voice"] == "pm_alex"
        assert captured["speed"] == 1.0
        assert captured["lang_code"] == "p"
        assert captured["model"] == "kokoro-82m"

    def test_accepts_selector_aliases(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generate(self, inputs):
            captured.update(inputs)
            return ToolResult(success=True, data={"ok": True})

        monkeypatch.setattr(KokoroTTS, "_generate", fake_generate)
        monkeypatch.setattr(KokoroTTS, "get_status", lambda self: ToolStatus.AVAILABLE)
        result = KokoroTTS().execute(
            {
                "text": "Olá",
                "output_path": str(tmp_path / "out.wav"),
                "voice_id": "pm_alex",
                "speaking_rate": 1.1,
                "language_code": "pt-BR",
            }
        )
        assert result.success is True
        assert captured["voice"] == "pm_alex"
        assert captured["speed"] == 1.1
        assert captured["lang_code"] == "p"


# ---- Generation and metadata ----


class TestGeneration:
    def test_generates_wav_from_worker_metadata(self, monkeypatch, tmp_path):
        python, worker = _stage_runtime(monkeypatch, tmp_path)
        monkeypatch.setenv("PYTHONPATH", r"C:\OpenMontage\should-not-leak")
        output = tmp_path / "narration.wav"
        seen = {}

        def fake_run(cmd, input=None, **kwargs):
            seen["cmd"] = cmd
            seen["input"] = input
            seen["kwargs"] = kwargs
            payload = json.loads(input)
            destination = Path(payload["output_path"])
            _write_wav(destination)
            meta = _metadata(
                str(destination),
                voice=payload["voice"],
                speed=payload["speed"],
            )
            return _completed(0, stdout=json.dumps(meta) + "\n")

        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", fake_run)
        result = KokoroTTS().execute(
            {
                "text": "Olá, OpenMontage.",
                "voice": "pm_alex",
                "speed": 1.0,
                "output_path": str(output),
            }
        )

        assert result.success is True
        assert result.model == "kokoro-82m"
        assert result.cost_usd == 0.0
        assert result.artifacts == [str(output.resolve())]
        assert result.data["provider"] == "kokoro"
        assert result.data["voice"] == "pm_alex"
        assert result.data["format"] == "wav"
        assert result.data["sample_rate"] == 24000
        assert result.data["audio_duration_seconds"] == 1.25
        assert result.data["generation_seconds"] == 0.4
        assert result.data["text_length"] == len("Olá, OpenMontage.")
        assert result.duration_seconds >= 0
        assert seen["cmd"] == [str(python), str(worker)]
        assert seen["cmd"][0] != sys.executable
        assert "PYTHONPATH" not in seen["kwargs"]["env"]
        assert seen["kwargs"]["env"]["VIRTUAL_ENV"].endswith(
            str(Path("experiments") / "kokoro" / ".venv")
        )
        request = json.loads(seen["input"])
        assert request["text"] == "Olá, OpenMontage."
        assert request["lang_code"] == "p"
        assert Path(request["output_path"]).is_absolute()

    def test_reads_metadata_after_worker_logs(self, monkeypatch, tmp_path):
        _stage_runtime(monkeypatch, tmp_path)
        output = tmp_path / "speech.wav"

        def fake_run(cmd, input=None, **kwargs):
            payload = json.loads(input)
            destination = Path(payload["output_path"])
            _write_wav(destination)
            stdout = "loading model\n" + json.dumps(_metadata(str(destination))) + "\n"
            return _completed(0, stdout=stdout)

        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", fake_run)
        result = KokoroTTS().execute({"text": "Olá", "output_path": str(output)})
        assert result.success is True
        assert result.data["audio_duration_seconds"] == 1.25


# ---- Failures ----


class TestFailures:
    def test_unavailable_when_interpreter_is_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            KokoroTTS,
            "python_executable",
            lambda self: tmp_path / "missing-python.exe",
        )
        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", _forbid_subprocess)
        tool = KokoroTTS()
        assert tool.get_status() == ToolStatus.UNAVAILABLE
        result = tool.execute({"text": "Olá", "output_path": str(tmp_path / "out.wav")})
        assert result.success is False
        assert "not available" in result.error

    def test_worker_exit_code(self, monkeypatch, tmp_path):
        _stage_runtime(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "tools.audio.kokoro_tts.subprocess.run",
            lambda *args, **kwargs: _completed(1, stderr="espeak library missing"),
        )
        result = KokoroTTS().execute(
            {"text": "Olá", "output_path": str(tmp_path / "out.wav")}
        )
        assert result.success is False
        assert "exit 1" in result.error
        assert "espeak library missing" in result.error

    def test_missing_output_file(self, monkeypatch, tmp_path):
        _stage_runtime(monkeypatch, tmp_path)
        output = tmp_path / "missing.wav"

        def fake_run(cmd, input=None, **kwargs):
            payload = json.loads(input)
            return _completed(0, stdout=json.dumps(_metadata(payload["output_path"])))

        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", fake_run)
        result = KokoroTTS().execute({"text": "Olá", "output_path": str(output)})
        assert result.success is False
        assert "missing" in result.error

    def test_empty_output_file(self, monkeypatch, tmp_path):
        _stage_runtime(monkeypatch, tmp_path)
        output = tmp_path / "empty.wav"

        def fake_run(cmd, input=None, **kwargs):
            payload = json.loads(input)
            Path(payload["output_path"]).write_bytes(b"")
            return _completed(0, stdout=json.dumps(_metadata(payload["output_path"])))

        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", fake_run)
        result = KokoroTTS().execute({"text": "Olá", "output_path": str(output)})
        assert result.success is False
        assert "empty" in result.error

    def test_malformed_metadata(self, monkeypatch, tmp_path):
        _stage_runtime(monkeypatch, tmp_path)
        output = tmp_path / "speech.wav"

        def fake_run(cmd, input=None, **kwargs):
            payload = json.loads(input)
            _write_wav(Path(payload["output_path"]))
            return _completed(0, stdout="not json\n")

        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", fake_run)
        result = KokoroTTS().execute({"text": "Olá", "output_path": str(output)})
        assert result.success is False
        assert "JSON metadata" in result.error

    def test_unexpected_sample_rate(self, monkeypatch, tmp_path):
        _stage_runtime(monkeypatch, tmp_path)
        output = tmp_path / "speech.wav"

        def fake_run(cmd, input=None, **kwargs):
            payload = json.loads(input)
            _write_wav(Path(payload["output_path"]))
            meta = _metadata(payload["output_path"], sample_rate=16000)
            return _completed(0, stdout=json.dumps(meta))

        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", fake_run)
        result = KokoroTTS().execute({"text": "Olá", "output_path": str(output)})
        assert result.success is False
        assert "24000" in result.error

    def test_timeout(self, monkeypatch, tmp_path):
        _stage_runtime(monkeypatch, tmp_path)

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1))

        monkeypatch.setattr("tools.audio.kokoro_tts.subprocess.run", fake_run)
        result = KokoroTTS().execute(
            {"text": "Olá", "output_path": str(tmp_path / "out.wav")}
        )
        assert result.success is False
        assert "timed out" in result.error


# ---- Selector ----


class TestSelector:
    def _registry(self, monkeypatch):
        registry = ToolRegistry()
        monkeypatch.setattr("tools.tool_registry.registry", registry)
        registry.discover("tools")
        monkeypatch.setattr(KokoroTTS, "get_status", lambda self: ToolStatus.AVAILABLE)
        return registry

    def test_preferred_provider(self, monkeypatch):
        registry = self._registry(monkeypatch)

        def fake_execute(self, inputs):
            return ToolResult(success=True, data={"voice": inputs.get("voice")}, artifacts=[])

        monkeypatch.setattr(KokoroTTS, "execute", fake_execute)
        result = registry.get("tts_selector").execute(
            {
                "text": "Olá",
                "preferred_provider": "kokoro",
                "output_path": "out.wav",
            }
        )
        assert result.success is True
        assert result.data["selected_tool"] == "kokoro_tts"
        assert result.data["selected_provider"] == "kokoro"

    def test_preferred_tool_name(self, monkeypatch):
        registry = self._registry(monkeypatch)

        def fake_execute(self, inputs):
            return ToolResult(success=True, data={}, artifacts=[])

        monkeypatch.setattr(KokoroTTS, "execute", fake_execute)
        result = registry.get("tts_selector").execute(
            {
                "text": "Olá",
                "preferred_tool": "kokoro_tts",
                "output_path": "out.wav",
            }
        )
        assert result.success is True
        assert result.data["selected_tool"] == "kokoro_tts"


# ---- Isolation from the project environment ----


class TestIsolation:
    def test_tool_module_does_not_import_kokoro_packages(self):
        source = (PROJECT_ROOT / "tools" / "audio" / "kokoro_tts.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(KOKORO_PACKAGES)
        assert "kokoro" not in sys.modules

    def test_worker_compiles_without_executing_kokoro(self, tmp_path):
        py_compile.compile(str(WORKER), cfile=str(tmp_path / "generate_speech.pyc"), doraise=True)
        source = WORKER.read_text(encoding="utf-8")
        assert "SAMPLE_RATE = 24000" in source
        assert "voice=voice" in source
        assert "KPipeline" in source

    def test_project_interpreter_cannot_import_kokoro(self):
        proc = subprocess.run(
            [sys.executable, "-c", "import kokoro"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0
