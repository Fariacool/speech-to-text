#!/usr/bin/env python3
"""Run global speaker diarization with pyannote.audio."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from stt_logging import format_seconds, log, log_step_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global speaker diarization and write RTTM/JSON.")
    parser.add_argument("input", type=Path, help="Input audio/video file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs-diarization"))
    parser.add_argument("--prefix", help="Output file prefix. Defaults to input stem.")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--token", help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--num-speakers", type=int, help="Exact known number of speakers.")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers.")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers.")
    parser.add_argument(
        "--no-normalize-audio",
        action="store_true",
        help="Pass input directly to pyannote instead of first converting to 16 kHz mono WAV.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing executable: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("`ffmpeg` is required. Install it first, e.g. `sudo apt-get install ffmpeg`.")


def normalize_audio(input_path: Path, wav_path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )


def diarization_to_rows(diarization: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        rows.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    rows.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    return rows


def main() -> int:
    script_started = time.monotonic()
    args = parse_args()
    log("Starting pyannote-diarize.")
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    log(f"Input: {input_path}")
    if not args.no_normalize_audio:
        require_ffmpeg()

    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        raise SystemExit("Use either --num-speakers or --min-speakers/--max-speakers, not both.")

    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Missing Hugging Face token. Set HF_TOKEN or pass --token after accepting the pyannote model terms."
        )

    try:
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError as exc:
        raise SystemExit(
            "Missing pyannote.audio. Install it with `uv pip install --python .venv/bin/python -e '.[diarization]'`."
        ) from exc

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"

    log(f"Loading diarization model: {args.model}")
    load_started = time.monotonic()
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    pipeline.to(torch.device(device_name))
    log_step_done("Diarization model load", load_started)
    call_kwargs: dict[str, Any] = {}
    if args.num_speakers is not None:
        call_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            call_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            call_kwargs["max_speakers"] = args.max_speakers

    with tempfile.TemporaryDirectory(prefix="pyannote_diarize_") as tmp_name:
        diarization_input = input_path
        if not args.no_normalize_audio:
            normalize_started = time.monotonic()
            diarization_input = Path(tmp_name) / "input-16k-mono.wav"
            log(f"Normalizing audio for pyannote: {diarization_input}")
            normalize_audio(input_path, diarization_input)
            log_step_done("Audio normalization", normalize_started)

        log(f"Running diarization on {device_name}: {diarization_input}")
        diarize_started = time.monotonic()
        with ProgressHook() as hook:
            diarization = pipeline(str(diarization_input), hook=hook, **call_kwargs)
    log_step_done("Diarization inference", diarize_started)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem
    rttm_path = args.output_dir / f"{prefix}.rttm"
    json_path = args.output_dir / f"{prefix}.speakers.json"

    write_started = time.monotonic()
    with rttm_path.open("w", encoding="utf-8") as file:
        diarization.write_rttm(file)

    rows = diarization_to_rows(diarization)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    log_step_done("Diarization output write", write_started)

    log("Done. Wrote:")
    log(f"  {rttm_path}")
    log(f"  {json_path}")
    log(
        f"Total pyannote-diarize elapsed: {format_seconds(time.monotonic() - script_started)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
