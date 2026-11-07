"""Narration bed: manifest lookup, WAV placement, Remotion src, HyperFrames segments."""

from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.audio.narration_bed import prepare_narration_bed
from tools.base_tool import ToolResult
from tools.video.hyperframes_compose import HyperFramesCompose
from tools.video.video_compose import VideoCompose

FFMPEG = shutil.which("ffmpeg")
pytestmark_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg required")


class _RecordingMixer:
    def __init__(self):
        self.calls = []

    def execute(self, inputs):
        self.calls.append(inputs)
        destination = Path(inputs["output_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"RIFF" + b"\x00" * 64)
        return ToolResult(
            success=True,
            data={"output": str(destination)},
            artifacts=[str(destination)],
        )


def _manifest(tmp_path: Path, names: list[str]) -> dict:
    assets = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"audio")
        assets.append(
            {
                "id": Path(name).stem,
                "type": "audio",
                "path": str(path),
                "source_tool": "piper_tts" if name.endswith(".mp3") else "kokoro_tts",
                "scene_id": Path(name).stem,
            }
        )
    return {"version": "1.0", "assets": assets}


def _edit(segments: list[dict]) -> dict:
    return {"audio": {"narration": {"segments": segments}}}


def _sine(path: Path, freq: int, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}:sample_rate=24000",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )


def _duration(path: Path) -> float:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return float(probe.stdout.strip())


def _rms(path: Path, start: float, end: float) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        width = handle.getsampwidth()
        handle.setpos(int(start * rate))
        raw = handle.readframes(int((end - start) * rate))
    step = width
    samples = [
        int.from_bytes(raw[i:i + step], "little", signed=True)
        for i in range(0, len(raw) - step + 1, step)
    ]
    if not samples:
        return 0.0
    return (sum(sample * sample for sample in samples) / len(samples)) ** 0.5


class TestResolve:
    def test_missing_asset_id(self, tmp_path):
        edit = _edit([{"asset_id": "missing", "start_seconds": 0}])
        mixer = _RecordingMixer()
        result = prepare_narration_bed(
            edit, {"version": "1.0", "assets": []}, mixer=mixer, project_dir=tmp_path
        )
        assert result["action"] == "error"
        assert "missing" in result["error"]
        assert mixer.calls == []

    def test_missing_wav(self, tmp_path):
        manifest = {
            "version": "1.0",
            "assets": [
                {
                    "id": "s1",
                    "type": "audio",
                    "path": str(tmp_path / "gone.wav"),
                    "source_tool": "kokoro_tts",
                    "scene_id": "s1",
                }
            ],
        }
        mixer = _RecordingMixer()
        result = prepare_narration_bed(
            _edit([{"asset_id": "s1", "start_seconds": 0}]),
            manifest,
            mixer=mixer,
            project_dir=tmp_path,
        )
        assert result["action"] == "error"
        assert "missing" in result["error"]
        assert mixer.calls == []

    def test_relative_wav_and_segment_order(self, tmp_path):
        first = tmp_path / "assets" / "audio" / "s1.wav"
        second = tmp_path / "assets" / "audio" / "s2.wav"
        first.parent.mkdir(parents=True)
        first.write_bytes(b"a")
        second.write_bytes(b"b")
        manifest = {
            "version": "1.0",
            "assets": [
                {"id": "s1", "type": "audio", "path": "assets/audio/s1.wav", "source_tool": "kokoro_tts", "scene_id": "s1"},
                {"id": "s2", "type": "audio", "path": "assets/audio/s2.wav", "source_tool": "kokoro_tts", "scene_id": "s2"},
            ],
        }
        edit = _edit(
            [
                {"asset_id": "s2", "start_seconds": 3},
                {"asset_id": "s1", "start_seconds": 0},
            ]
        )
        mixer = _RecordingMixer()
        result = prepare_narration_bed(edit, manifest, project_dir=tmp_path, mixer=mixer)
        assert result["action"] == "created"
        starts = [track["start_seconds"] for track in mixer.calls[0]["tracks"]]
        assert starts == [0.0, 3.0]
        assert mixer.calls[0]["tracks"][0]["path"].endswith("s1.wav")
        assert edit["audio"]["narration"]["segments"][0]["asset_id"] == "s2"
        assert edit["audio"]["narration"]["src"].endswith("narration_bed.wav")
        assert any(asset["id"] == "narration-bed" for asset in manifest["assets"])

    def test_keeps_existing_src_for_older_projects(self, tmp_path):
        ready = tmp_path / "already.wav"
        ready.write_bytes(b"ready")
        section = tmp_path / "s1.wav"
        section.write_bytes(b"section")
        edit = _edit([{"asset_id": "s1", "start_seconds": 0}])
        edit["audio"]["narration"]["src"] = str(ready)
        manifest = {
            "version": "1.0",
            "assets": [
                {"id": "s1", "type": "audio", "path": str(section), "source_tool": "piper_tts", "scene_id": "s1"}
            ],
        }
        mixer = _RecordingMixer()
        result = prepare_narration_bed(edit, manifest, project_dir=tmp_path, mixer=mixer)
        assert result["action"] == "kept"
        assert result["src"] == str(ready.resolve())
        assert mixer.calls == []

    def test_piper_mp3_segments_are_scheduled(self, tmp_path):
        manifest = _manifest(tmp_path, ["s1.mp3", "s2.mp3"])
        edit = _edit(
            [
                {"asset_id": "s1", "start_seconds": 0},
                {"asset_id": "s2", "start_seconds": 4},
            ]
        )
        mixer = _RecordingMixer()
        result = prepare_narration_bed(edit, manifest, project_dir=tmp_path, mixer=mixer)
        assert result["action"] == "created"
        assert [track["path"].endswith(".mp3") for track in mixer.calls[0]["tracks"]] == [True, True]
        assert [track["start_seconds"] for track in mixer.calls[0]["tracks"]] == [0.0, 4.0]


@pytestmark_ffmpeg
class TestConsolidate:
    def test_two_segments_keep_the_gap(self, tmp_path):
        first = tmp_path / "s1.wav"
        second = tmp_path / "s2.wav"
        _sine(first, 440, 1)
        _sine(second, 880, 1)
        manifest = {
            "version": "1.0",
            "assets": [
                {"id": "s1", "type": "audio", "path": str(first), "source_tool": "kokoro_tts", "scene_id": "s1"},
                {"id": "s2", "type": "audio", "path": str(second), "source_tool": "kokoro_tts", "scene_id": "s2"},
            ],
        }
        edit = _edit(
            [
                {"asset_id": "s1", "start_seconds": 0},
                {"asset_id": "s2", "start_seconds": 3},
            ]
        )
        bed_path = tmp_path / "assets" / "audio" / "narration_bed.wav"
        result = prepare_narration_bed(
            edit, manifest, project_dir=tmp_path, output_path=bed_path
        )
        assert result["action"] == "created"
        assert Path(result["src"]).is_file()
        duration = _duration(Path(result["src"]))
        assert duration == pytest.approx(4.0, abs=0.25)
        assert result["duration_seconds"] == pytest.approx(duration, abs=0.05)
        assert _rms(Path(result["src"]), 1.4, 2.4) < _rms(Path(result["src"]), 0.1, 0.6) / 5


class TestRuntimes:
    def test_remotion_receives_narration_src(self, tmp_path):
        bed = tmp_path / "narration_bed.wav"
        bed.write_bytes(b"RIFFxxxxWAVEfmt " + b"\x00" * 32)
        props = {
            "audio": {"narration": {"src": str(bed), "volume": 1}},
            "cuts": [{"source": "not-a-file"}],
        }
        public = tmp_path / "public"
        staged = VideoCompose._stage_remotion_media(props, public)
        staged_name = props["audio"]["narration"]["src"]
        assert staged == 1
        assert Path(staged_name).name == staged_name
        assert (public / staged_name).is_file()
        assert props["audio"]["narration"]["volume"] == 1

    def test_hyperframes_still_resolves_segment_files(self, tmp_path):
        workspace = tmp_path / "hyperframes"
        audio_dir = workspace / "assets"
        audio_dir.mkdir(parents=True)
        first = audio_dir / "s1.wav"
        second = audio_dir / "s2.wav"
        first.write_bytes(b"one")
        second.write_bytes(b"two")
        manifest = {
            "assets": [
                {"id": "s1", "path": str(first)},
                {"id": "s2", "path": str(second)},
            ]
        }
        audio = {
            "narration": {
                "src": str(tmp_path / "narration_bed.wav"),
                "segments": [
                    {"asset_id": "s1", "start_seconds": 0},
                    {"asset_id": "s2", "start_seconds": 3},
                ],
            }
        }
        resolved = HyperFramesCompose()._resolve_audio_refs(audio, manifest["assets"], workspace)
        assert [item["start_seconds"] for item in resolved["narration"]] == [0.0, 3.0]
        assert all(Path(item["src"]).is_file() for item in resolved["narration"])

    def test_ffmpeg_hook_uses_bed_without_replacing_a_caller_mix(self, tmp_path, monkeypatch):
        bed = tmp_path / "assets" / "audio" / "narration_bed.wav"

        def fake_prepare(edit_decisions, asset_manifest, project_dir=None, **kwargs):
            edit_decisions["audio"]["narration"]["src"] = str(bed)
            return {"action": "created", "src": str(bed)}

        monkeypatch.setattr("tools.audio.narration_bed.prepare_narration_bed", fake_prepare)
        edit = _edit([{"asset_id": "s1", "start_seconds": 0}])
        manifest = {"version": "1.0", "assets": []}
        output = tmp_path / "renders" / "out.mp4"
        ffmpeg_inputs: dict = {}
        VideoCompose()._consolidate_narration_bed(
            ffmpeg_inputs, edit, manifest, output, "ffmpeg"
        )
        assert ffmpeg_inputs["audio_path"] == str(bed)

        supplied = {"audio_path": "projects/demo/assets/audio/final_mix.wav"}
        VideoCompose()._consolidate_narration_bed(
            supplied, edit, manifest, output, "ffmpeg"
        )
        assert supplied["audio_path"] == "projects/demo/assets/audio/final_mix.wav"

        remotion_inputs: dict = {}
        VideoCompose()._consolidate_narration_bed(
            remotion_inputs, edit, manifest, output, "remotion"
        )
        assert "audio_path" not in remotion_inputs
        assert edit["audio"]["narration"]["src"] == str(bed)
        assert edit["audio"]["narration"]["segments"][0]["asset_id"] == "s1"
