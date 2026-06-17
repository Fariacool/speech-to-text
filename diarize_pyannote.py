#!/usr/bin/env python3
"""Run global speaker diarization with pyannote.audio."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


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
    return parser.parse_args()


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
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

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
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        device_name = "cpu"

    print(f"Loading diarization model: {args.model}", flush=True)
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    pipeline.to(torch.device(device_name))
    print(f"Running diarization on {device_name}: {input_path}", flush=True)

    call_kwargs: dict[str, Any] = {}
    if args.num_speakers is not None:
        call_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers is not None:
            call_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            call_kwargs["max_speakers"] = args.max_speakers

    with ProgressHook() as hook:
        diarization = pipeline(str(input_path), hook=hook, **call_kwargs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem
    rttm_path = args.output_dir / f"{prefix}.rttm"
    json_path = args.output_dir / f"{prefix}.speakers.json"

    with rttm_path.open("w", encoding="utf-8") as file:
        diarization.write_rttm(file)

    rows = diarization_to_rows(diarization)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done. Wrote:", flush=True)
    print(f"  {rttm_path}", flush=True)
    print(f"  {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
