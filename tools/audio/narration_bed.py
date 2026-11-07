"""Build one narration WAV from edit_decisions segments.

Section files stay in the asset manifest. Runtimes that play a single
``narration.src`` or ``audio_path`` receive a bed laid out with the existing
``audio_mixer`` delays, so silence between sections is kept. HyperFrames
continues to read the original segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.analysis.audio_probe import probe_duration
from tools.audio.audio_mixer import AudioMixer

NARRATION_BED_ID = "narration-bed"


def prepare_narration_bed(
    edit_decisions: dict[str, Any],
    asset_manifest: dict[str, Any],
    *,
    project_dir: Path | None = None,
    output_path: Path | None = None,
    mixer: AudioMixer | None = None,
) -> dict[str, Any]:
    """Resolve narration segments and write ``narration.src`` when it is missing.

    An existing ``narration.src`` that points at a real file is left in place
    so older compositions keep the narration they already selected. Segments
    are not removed.
    """
    audio = edit_decisions.get("audio")
    if audio is None:
        return {"action": "skipped", "src": None}
    if not isinstance(audio, dict):
        return {"action": "error", "error": "edit_decisions.audio must be an object"}

    narration = audio.get("narration")
    if narration is None:
        return {"action": "skipped", "src": None}
    if not isinstance(narration, dict):
        return {"action": "error", "error": "edit_decisions.audio.narration must be an object"}

    existing_src = narration.get("src")
    if isinstance(existing_src, str) and existing_src.strip():
        existing_path = _resolve_media_path(existing_src, project_dir)
        if existing_path is not None and existing_path.is_file():
            return {
                "action": "kept",
                "src": str(existing_path),
                "duration_seconds": probe_duration(existing_path),
            }

    segments = narration.get("segments") or []
    if not segments:
        return {"action": "skipped", "src": None}

    if not isinstance(segments, list):
        return {"action": "error", "error": "narration.segments must be a list"}

    assets = asset_manifest.setdefault("assets", [])
    if not isinstance(assets, list):
        return {"action": "error", "error": "asset_manifest.assets must be a list"}
    lookup = {
        asset.get("id"): asset
        for asset in assets
        if isinstance(asset, dict) and asset.get("id")
    }

    ordered = sorted(
        enumerate(segments),
        key=lambda item: (
            float(item[1].get("start_seconds", 0) or 0) if isinstance(item[1], dict) else 0.0,
            item[0],
        ),
    )

    tracks: list[dict[str, Any]] = []
    for _, segment in ordered:
        if not isinstance(segment, dict):
            return {"action": "error", "error": "Each narration segment must be an object"}
        asset_id = segment.get("asset_id")
        if not asset_id or asset_id not in lookup:
            return {
                "action": "error",
                "error": f"Narration asset_id not found in asset_manifest: {asset_id}",
            }
        raw_path = lookup[asset_id].get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return {"action": "error", "error": f"Narration asset {asset_id} has no path"}
        media = _resolve_media_path(raw_path, project_dir)
        if media is None or not media.is_file():
            return {
                "action": "error",
                "error": f"Narration file missing for {asset_id}: {raw_path}",
            }
        try:
            start_seconds = float(segment.get("start_seconds", 0) or 0)
        except (TypeError, ValueError):
            return {
                "action": "error",
                "error": f"Invalid start_seconds for narration asset {asset_id}",
            }
        if start_seconds < 0:
            return {
                "action": "error",
                "error": f"start_seconds must be >= 0 for narration asset {asset_id}",
            }
        tracks.append(
            {
                "path": str(media),
                "role": "speech",
                "start_seconds": start_seconds,
                "volume": 1.0,
            }
        )

    destination = Path(output_path) if output_path is not None else _default_bed_path(project_dir, tracks)
    destination.parent.mkdir(parents=True, exist_ok=True)

    mix = (mixer or AudioMixer()).execute(
        {
            "operation": "full_mix",
            "tracks": tracks,
            "ducking": {"enabled": False},
            "normalize": False,
            "output_path": str(destination),
        }
    )
    if not mix.success:
        return {"action": "error", "error": mix.error or "Narration bed mix failed"}
    if not destination.is_file():
        return {"action": "error", "error": f"Narration bed file missing: {destination}"}

    duration = probe_duration(destination)
    src = str(destination.resolve())
    narration["src"] = src
    _register_bed_asset(assets, src, duration, tracks[0]["path"])
    return {
        "action": "created",
        "src": src,
        "duration_seconds": duration,
        "tracks": tracks,
        "asset_id": NARRATION_BED_ID,
    }


def _register_bed_asset(
    assets: list[Any],
    src: str,
    duration: float | None,
    first_source: str,
) -> None:
    record = {
        "id": NARRATION_BED_ID,
        "type": "audio",
        "subtype": "narration",
        "path": src,
        "source_tool": "audio_mixer",
        "scene_id": "narration",
        "format": "wav",
        "generation_summary": f"Consolidated from section narration, first source {first_source}",
    }
    if duration is not None:
        record["duration_seconds"] = round(duration, 3)
    for asset in assets:
        if isinstance(asset, dict) and asset.get("id") == NARRATION_BED_ID:
            asset.update(record)
            return
    assets.append(record)


def _default_bed_path(project_dir: Path | None, tracks: list[dict[str, Any]]) -> Path:
    if project_dir is not None:
        return Path(project_dir) / "assets" / "audio" / "narration_bed.wav"
    return Path(tracks[0]["path"]).resolve().parent / "narration_bed.wav"


def _resolve_media_path(raw: str, project_dir: Path | None) -> Path | None:
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    if project_dir is not None:
        rooted = Path(project_dir) / raw
        if rooted.is_file():
            return rooted.resolve()
    return candidate if candidate.is_file() else None
