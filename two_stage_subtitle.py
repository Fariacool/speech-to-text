#!/usr/bin/env python3
"""Run chunked ASR, global diarization, and speaker merge as one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from stt_logging import command_to_string, format_seconds, log, log_step_done


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HF_TIMESTAMP_MODEL = (
    "alextomcat/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)


def script_path(name: str) -> str:
    return str(SCRIPT_DIR / name)


def run_command(label: str, cmd: list[str]) -> None:
    started = time.monotonic()
    log(f"Start step: {label}")
    log(f"Command: {command_to_string(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        log(f"Failed step: {label}, missing executable: {cmd[0]}")
        raise SystemExit(f"Missing executable: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        log(f"Failed step: {label}, exit_code={exc.returncode}")
        raise SystemExit(exc.returncode) from exc
    log_step_done(f"Step {label}", started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command two-stage subtitle pipeline: FunASR ASR + pyannote diarization + merge."
    )
    parser.add_argument("input", type=Path, help="Input audio/video file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs-two-stage"))
    parser.add_argument("--prefix", help="Output file prefix. Defaults to input stem.")
    parser.add_argument("--format", choices=("srt", "vtt", "txt", "json", "all"), default="all")

    parser.add_argument("--device", default="cuda:0", help="FunASR device, e.g. cuda:0 or cpu.")
    parser.add_argument("--diarization-device", default="cuda", help="pyannote device, e.g. cuda or cpu.")
    parser.add_argument("--chunk-minutes", type=int, default=30, help="ASR physical chunk size.")
    parser.add_argument("--hotword", action="append", default=[], help="ASR hotword. Can be repeated.")
    parser.add_argument("--hub", choices=("hf", "ms"), default="hf", help="FunASR model hub.")
    parser.add_argument(
        "--model",
        help="FunASR ASR model id. Defaults to a timestamp-capable Paraformer model for two-stage mode.",
    )
    parser.add_argument("--vad-model", help="FunASR VAD model id.")
    parser.add_argument("--punc-model", help="FunASR punctuation model id.")
    parser.add_argument("--batch-size-s", type=int, default=300)
    parser.add_argument("--batch-threshold-s", type=int, default=60)
    parser.add_argument("--max-single-segment-ms", type=int, default=60_000)
    parser.add_argument("--funasr-log-level", default="ERROR")
    parser.add_argument("--show-funasr-progress", action="store_true")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--no-partial", action="store_true")

    parser.add_argument("--diarization-model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--token", help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.")
    parser.add_argument("--segmentation-batch-size", type=int, default=32)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--diarization-output",
        choices=("auto", "exclusive", "regular"),
        default="auto",
        help="Which pyannote diarization output to merge. auto prefers exclusive when available.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 for pyannote CUDA inference. This may slightly affect reproducibility.",
    )
    parser.add_argument("--num-speakers", type=int, help="Exact known number of speakers.")
    parser.add_argument("--min-speakers", type=int, help="Minimum number of speakers.")
    parser.add_argument("--max-speakers", type=int, help="Maximum number of speakers.")
    parser.add_argument("--speaker-map", action="append", default=[], help="Map speaker labels, e.g. SPEAKER_00=Name.")

    parser.add_argument("--skip-asr", action="store_true", help="Reuse existing ASR output in output-dir/asr.")
    parser.add_argument(
        "--skip-diarization",
        action="store_true",
        help="Reuse existing diarization output in output-dir/diarization.",
    )
    parser.add_argument("--skip-merge", action="store_true", help="Run ASR/diarization but do not merge.")
    return parser.parse_args()


def add_optional(cmd: list[str], flag: str, value: str | int | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def main() -> int:
    pipeline_started = time.monotonic()
    args = parse_args()
    log("Starting two-stage subtitle pipeline.")

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        raise SystemExit("Use either --num-speakers or --min-speakers/--max-speakers, not both.")

    output_root = args.output_dir.expanduser().resolve()
    asr_dir = output_root / "asr"
    diarization_dir = output_root / "diarization"
    final_dir = output_root / "final"
    prefix = args.prefix or input_path.stem

    log(f"Input: {input_path}")
    log(f"Output root: {output_root}")
    log(f"Prefix: {prefix}")
    output_root.mkdir(parents=True, exist_ok=True)
    asr_dir.mkdir(parents=True, exist_ok=True)
    diarization_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    segments_path = asr_dir / f"{prefix}.segments.json"
    speakers_path = diarization_dir / f"{prefix}.speakers.json"

    if args.skip_asr:
        log(f"Skipping ASR. Expecting existing segments: {segments_path}")
    else:
        asr_model = args.model
        if asr_model is None and args.hub == "hf":
            asr_model = DEFAULT_HF_TIMESTAMP_MODEL

        asr_cmd = [
            sys.executable,
            script_path("transcribe_funasr.py"),
            str(input_path),
            "--device",
            args.device,
            "--hub",
            args.hub,
            "--chunk-minutes",
            str(args.chunk_minutes),
            "--output-dir",
            str(asr_dir),
            "--prefix",
            prefix,
            "--format",
            "all",
            "--require-timestamps",
            "--batch-size-s",
            str(args.batch_size_s),
            "--batch-threshold-s",
            str(args.batch_threshold_s),
            "--max-single-segment-ms",
            str(args.max_single_segment_ms),
            "--funasr-log-level",
            args.funasr_log_level,
        ]
        add_optional(asr_cmd, "--model", asr_model)
        add_optional(asr_cmd, "--vad-model", args.vad_model)
        add_optional(asr_cmd, "--punc-model", args.punc_model)
        for hotword in args.hotword:
            asr_cmd.extend(["--hotword", hotword])
        if args.show_funasr_progress:
            asr_cmd.append("--show-funasr-progress")
        if args.keep_audio:
            asr_cmd.append("--keep-audio")
        if args.no_partial:
            asr_cmd.append("--no-partial")
        run_command("ASR with FunASR, chunked without --spk", asr_cmd)

    if not segments_path.exists():
        raise SystemExit(f"Expected ASR segments not found: {segments_path}")

    if args.skip_diarization:
        log(f"Skipping diarization. Expecting existing speakers: {speakers_path}")
    else:
        diarize_cmd = [
            sys.executable,
            script_path("diarize_pyannote.py"),
            str(input_path),
            "--device",
            args.diarization_device,
            "--model",
            args.diarization_model,
            "--segmentation-batch-size",
            str(args.segmentation_batch_size),
            "--embedding-batch-size",
            str(args.embedding_batch_size),
            "--diarization-output",
            args.diarization_output,
            "--output-dir",
            str(diarization_dir),
            "--prefix",
            prefix,
        ]
        add_optional(diarize_cmd, "--token", args.token)
        add_optional(diarize_cmd, "--num-speakers", args.num_speakers)
        add_optional(diarize_cmd, "--min-speakers", args.min_speakers)
        add_optional(diarize_cmd, "--max-speakers", args.max_speakers)
        if args.allow_tf32:
            diarize_cmd.append("--allow-tf32")
        run_command("Global speaker diarization with pyannote", diarize_cmd)

    if not speakers_path.exists():
        raise SystemExit(f"Expected speaker file not found: {speakers_path}")

    if args.skip_merge:
        log("Skipping merge step.")
    else:
        merge_cmd = [
            sys.executable,
            script_path("merge_speakers.py"),
            str(segments_path),
            str(speakers_path),
            "--output-dir",
            str(final_dir),
            "--prefix",
            prefix,
            "--format",
            args.format,
        ]
        for item in args.speaker_map:
            merge_cmd.extend(["--speaker-map", item])
        run_command("Merge global speakers into subtitles", merge_cmd)

    log(f"Pipeline output root: {output_root}")
    log(f"Final subtitles: {final_dir}")
    log(f"Total two-stage pipeline elapsed: {format_seconds(time.monotonic() - pipeline_started)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
