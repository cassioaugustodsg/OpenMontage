"""Isolated Kokoro-82M speech worker.

Run this with the Kokoro virtualenv, not the OpenMontage interpreter:

    experiments/kokoro/.venv/Scripts/python.exe experiments/kokoro/generate_speech.py

The process reads one JSON object from stdin and writes one JSON object to
stdout. Speech generation lives here so the OpenMontage tool does not import
Kokoro, NumPy, or soundfile into the project environment.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SAMPLE_RATE = 24000
DEFAULT_VOICE = "pm_alex"
DEFAULT_LANG = "p"
DEFAULT_SPEED = 1.0


def generate_speech(
    text: str,
    voice: str,
    speed: float,
    lang_code: str,
    output_path: str,
) -> dict:
    """Synthesize ``text`` to a 24 kHz WAV file and return its metadata."""
    import numpy as np
    import soundfile as sf
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    # Point eSpeak NG at the bundled library. No global install is required.
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

    from kokoro import KPipeline

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipeline = KPipeline(lang_code=lang_code, device="cpu")
    chunks = [
        audio
        for _, _, audio in pipeline(text, voice=voice, speed=speed)
    ]
    if not chunks:
        raise RuntimeError("Kokoro did not generate audio.")

    audio = np.concatenate(chunks)
    sf.write(destination, audio, SAMPLE_RATE)
    generation_seconds = time.perf_counter() - started
    if not destination.is_file():
        raise RuntimeError(f"Kokoro output file missing: {destination}")

    return {
        "output": str(destination),
        "format": "wav",
        "sample_rate": SAMPLE_RATE,
        "audio_duration_seconds": round(len(audio) / SAMPLE_RATE, 2),
        "generation_seconds": round(generation_seconds, 2),
        "voice": voice,
        "lang_code": lang_code,
        "speed": speed,
        "model": "kokoro-82m",
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Kokoro request must be a JSON object")
        text = payload.get("text")
        output_path = payload.get("output_path")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError("output_path is required")
        result = generate_speech(
            text=text,
            voice=str(payload.get("voice") or DEFAULT_VOICE),
            speed=float(payload.get("speed", DEFAULT_SPEED)),
            lang_code=str(payload.get("lang_code") or DEFAULT_LANG),
            output_path=output_path,
        )
    except Exception as exc:
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
