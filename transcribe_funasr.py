#!/usr/bin/env python3
"""Transcribe long Chinese audio/video files with FunASR Paraformer."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing executable: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing executable: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc


def require_ffmpeg() -> None:
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise SystemExit(f"`{name}` is required. Install it first, e.g. `sudo apt-get install ffmpeg`.")


def log(message: str) -> None:
    print(message, flush=True)


def format_seconds(seconds: int | float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def ffmpeg_input_args(input_path: Path, start_seconds: float, duration_seconds: float) -> list[str]:
    args = ["-i", str(input_path)]
    if start_seconds > 0:
        args = ["-ss", str(start_seconds), *args]
    if duration_seconds > 0:
        args.extend(["-t", str(duration_seconds)])
    return args


def to_wav(input_path: Path, wav_path: Path, start_seconds: float = 0, duration_seconds: float = 0) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *ffmpeg_input_args(input_path, start_seconds, duration_seconds),
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


def split_to_wavs(
    input_path: Path,
    out_dir: Path,
    chunk_seconds: int,
    start_seconds: float = 0,
    duration_seconds: float = 0,
) -> list[Path]:
    pattern = out_dir / "chunk_%06d.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *ffmpeg_input_args(input_path, start_seconds, duration_seconds),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            str(pattern),
        ]
    )
    return sorted(out_dir.glob("chunk_*.wav"))


def duration_ms(path: Path) -> int:
    out = capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    return int(float(out) * 1000)


def ms_to_srt(ms: int | float | None) -> str:
    if ms is None or math.isnan(float(ms)):
        ms = 0
    total = max(0, int(round(float(ms))))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def ms_to_vtt(ms: int | float | None) -> str:
    return ms_to_srt(ms).replace(",", ".")


def first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def as_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    if isinstance(value, str):
        try:
            return int(round(float(value)))
        except ValueError:
            return None
    return None


def sentence_entries(result: Any, offset_ms: int, include_speaker: bool) -> list[dict[str, Any]]:
    records = result if isinstance(result, list) else [result]
    entries: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        sentences = record.get("sentence_info")
        if isinstance(sentences, list) and sentences:
            for sentence in sentences:
                if not isinstance(sentence, dict):
                    continue
                text = str(first_present(sentence, ("text", "sentence", "onebest")) or "").strip()
                if not text:
                    continue

                start = as_ms(first_present(sentence, ("start", "begin", "start_ms", "ts_start")))
                end = as_ms(first_present(sentence, ("end", "stop", "end_ms", "ts_end")))
                if start is None or end is None:
                    timestamp = sentence.get("timestamp")
                    if isinstance(timestamp, list) and timestamp:
                        first = timestamp[0]
                        last = timestamp[-1]
                        if isinstance(first, (list, tuple)) and len(first) >= 2:
                            start = as_ms(first[0])
                        if isinstance(last, (list, tuple)) and len(last) >= 2:
                            end = as_ms(last[1])

                speaker = first_present(sentence, ("spk", "speaker", "speaker_id"))
                if include_speaker and speaker is not None:
                    text = f"[SPEAKER_{speaker}] {text}"

                entries.append(
                    {
                        "start_ms": (start or 0) + offset_ms,
                        "end_ms": (end or start or 0) + offset_ms,
                        "speaker": speaker,
                        "text": text,
                    }
                )
            continue

        text = str(record.get("text") or "").strip()
        if text:
            entries.append(
                {
                    "start_ms": offset_ms,
                    "end_ms": offset_ms,
                    "speaker": None,
                    "text": text,
                }
            )

    return entries


def normalize_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = sorted(entries, key=lambda item: (item["start_ms"], item["end_ms"]))
    for idx, item in enumerate(entries):
        start = int(item["start_ms"])
        end = int(item["end_ms"])
        if end <= start:
            next_start = entries[idx + 1]["start_ms"] if idx + 1 < len(entries) else start + 2000
            end = max(start + 500, min(int(next_start), start + 6000))
        item["start_ms"] = start
        item["end_ms"] = end
    return entries


def write_srt(entries: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for idx, item in enumerate(entries, 1):
        lines.extend(
            [
                str(idx),
                f"{ms_to_srt(item['start_ms'])} --> {ms_to_srt(item['end_ms'])}",
                item["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(entries: list[dict[str, Any]], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for item in entries:
        lines.extend(
            [
                f"{ms_to_vtt(item['start_ms'])} --> {ms_to_vtt(item['end_ms'])}",
                item["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(entries: list[dict[str, Any]], path: Path) -> None:
    lines = [f"[{ms_to_srt(item['start_ms'])}] {item['text']}" for item in entries]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    entries: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
    output_dir: Path,
    prefix: str,
    output_format: str,
    suffix: str = "",
) -> list[Path]:
    normalized = normalize_entries(entries)
    raw_json_path = output_dir / f"{prefix}{suffix}.raw.json"
    entries_json_path = output_dir / f"{prefix}{suffix}.segments.json"
    raw_json_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    entries_json_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs: list[Path] = [raw_json_path, entries_json_path]
    if output_format in ("srt", "all"):
        path = output_dir / f"{prefix}{suffix}.srt"
        write_srt(normalized, path)
        outputs.append(path)
    if output_format in ("vtt", "all"):
        path = output_dir / f"{prefix}{suffix}.vtt"
        write_vtt(normalized, path)
        outputs.append(path)
    if output_format in ("txt", "all"):
        path = output_dir / f"{prefix}{suffix}.txt"
        write_txt(normalized, path)
        outputs.append(path)
    if output_format == "json":
        outputs = [raw_json_path, entries_json_path]
    return outputs


def model_names(hub: str, args: argparse.Namespace) -> dict[str, str]:
    if hub == "hf":
        defaults = {
            "model": "funasr/paraformer-zh",
            "vad_model": "funasr/fsmn-vad",
            "punc_model": "funasr/ct-punc",
            "spk_model": "funasr/campplus",
        }
    else:
        defaults = {
            "model": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            if args.spk
            else "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
            "spk_model": "cam++",
        }

    return {
        "model": args.model or defaults["model"],
        "vad_model": args.vad_model or defaults["vad_model"],
        "punc_model": args.punc_model or defaults["punc_model"],
        "spk_model": args.spk_model or defaults["spk_model"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Chinese audio/video to subtitles with FunASR Paraformer."
    )
    parser.add_argument("input", type=Path, help="Input video/audio file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--prefix", help="Output file prefix. Defaults to input stem.")
    parser.add_argument("--format", choices=("srt", "vtt", "txt", "json", "all"), default="all")
    parser.add_argument("--device", default="cuda:0", help="cuda:0, cuda, or cpu.")
    parser.add_argument("--hub", choices=("hf", "ms"), default="hf", help="Use Hugging Face or ModelScope model IDs.")
    parser.add_argument("--spk", action="store_true", help="Enable speaker diarization.")
    parser.add_argument("--hotword", action="append", default=[], help="Hotword. Can be repeated.")
    parser.add_argument("--chunk-minutes", type=int, default=0, help="Optional physical chunk size for long files.")
    parser.add_argument("--keep-audio", action="store_true", help="Keep extracted WAV files under output-dir/audio.")
    parser.add_argument("--sample-minutes", type=float, default=0, help="Only transcribe a short sample.")
    parser.add_argument("--sample-start-minutes", type=float, default=0, help="Sample start offset in minutes.")
    parser.add_argument("--no-partial", action="store_true", help="Do not write partial outputs after each chunk.")
    parser.add_argument("--preset-spk-num", type=int, help="Known number of speakers, e.g. 2 for interviews.")
    parser.add_argument("--spk-mode", choices=("default", "vad_segment", "punc_segment"), default="punc_segment")
    parser.add_argument("--show-funasr-progress", action="store_true", help="Show FunASR internal tqdm bars.")
    parser.add_argument(
        "--funasr-log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        default="ERROR",
    )
    parser.add_argument("--batch-size-s", type=int, default=300)
    parser.add_argument("--batch-threshold-s", type=int, default=60)
    parser.add_argument("--max-single-segment-ms", type=int, default=60_000)
    parser.add_argument("--model")
    parser.add_argument("--vad-model")
    parser.add_argument("--punc-model")
    parser.add_argument("--spk-model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_ffmpeg()
    logging.getLogger().setLevel(getattr(logging, args.funasr_log_level))

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SystemExit("Missing Python package `funasr`. Run `./install.sh` or `uv sync` first.") from exc

    if args.spk and args.hub == "hf" and args.model is None:
        log("Speaker diarization needs timestamp-capable ASR output; switching --hub to ms timestamp Paraformer preset.")
        args.hub = "ms"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_start_seconds = args.sample_start_minutes * 60
    sample_seconds = args.sample_minutes * 60
    if args.prefix:
        prefix = args.prefix
    elif args.sample_minutes > 0:
        prefix = f"{input_path.stem}.sample-{args.sample_start_minutes:g}m-{args.sample_minutes:g}m"
    else:
        prefix = input_path.stem

    source_duration_s = duration_ms(input_path) / 1000
    log(f"Input: {input_path}")
    log(f"Source duration: {format_seconds(source_duration_s)}")
    if args.sample_minutes > 0:
        log(
            "Sample mode: "
            f"start={format_seconds(sample_start_seconds)}, duration={format_seconds(sample_seconds)}"
        )

    names = model_names(args.hub, args)
    model_kwargs: dict[str, Any] = {
        "model": names["model"],
        "hub": args.hub,
        "vad_model": names["vad_model"],
        "vad_kwargs": {"max_single_segment_time": args.max_single_segment_ms},
        "punc_model": names["punc_model"],
        "device": args.device,
        "disable_update": True,
        "disable_pbar": not args.show_funasr_progress,
        "log_level": args.funasr_log_level,
    }
    if args.spk:
        model_kwargs["spk_model"] = names["spk_model"]
        model_kwargs["spk_mode"] = args.spk_mode
        if args.preset_spk_num is None:
            log("Tip: known-speaker interviews usually improve with --preset-spk-num 2.")

    log(f"Model: {names['model']} (hub={args.hub}, device={args.device}, spk={args.spk})")
    log(f"Loading FunASR model on {args.device}...")
    model = AutoModel(**model_kwargs)

    with tempfile.TemporaryDirectory(prefix="funasr_subs_") as tmp_name:
        tmp_dir = Path(tmp_name)
        audio_dir = args.output_dir / "audio" if args.keep_audio else tmp_dir
        audio_dir.mkdir(parents=True, exist_ok=True)

        if args.chunk_minutes > 0:
            log(f"Extracting and splitting audio into {args.chunk_minutes} minute chunks...")
            wavs = split_to_wavs(
                input_path,
                audio_dir,
                args.chunk_minutes * 60,
                start_seconds=sample_start_seconds,
                duration_seconds=sample_seconds,
            )
        else:
            wav = audio_dir / f"{prefix}.wav"
            log("Extracting audio...")
            to_wav(input_path, wav, start_seconds=sample_start_seconds, duration_seconds=sample_seconds)
            wavs = [wav]

        wav_durations_ms = [duration_ms(wav) for wav in wavs]
        total_audio_ms = sum(wav_durations_ms)
        log(f"Prepared {len(wavs)} chunk(s), audio to transcribe: {format_seconds(total_audio_ms / 1000)}")

        generate_kwargs: dict[str, Any] = {
            "batch_size_s": args.batch_size_s,
            "batch_size_threshold_s": args.batch_threshold_s,
        }
        if args.hotword:
            generate_kwargs["hotword"] = " ".join(args.hotword)
        if args.preset_spk_num is not None:
            generate_kwargs["preset_spk_num"] = args.preset_spk_num

        all_entries: list[dict[str, Any]] = []
        raw_results: list[dict[str, Any]] = []
        offset = 0
        processed_ms = 0
        total_started = time.monotonic()
        for idx, (wav, wav_duration_ms) in enumerate(zip(wavs, wav_durations_ms), 1):
            chunk_started = time.monotonic()
            progress = processed_ms / total_audio_ms if total_audio_ms else 0
            log(
                f"[{idx}/{len(wavs)}] Start {wav.name}: "
                f"offset={ms_to_srt(offset)}, duration={format_seconds(wav_duration_ms / 1000)}, "
                f"progress={progress:.1%}"
            )
            result = model.generate(input=str(wav), **generate_kwargs)
            chunk_entries = sentence_entries(result, offset, include_speaker=args.spk)
            raw_results.append({"file": str(wav), "offset_ms": offset, "result": result})
            all_entries.extend(chunk_entries)
            processed_ms += wav_duration_ms
            offset += wav_duration_ms

            chunk_elapsed_s = time.monotonic() - chunk_started
            total_elapsed_s = time.monotonic() - total_started
            avg_speed = (processed_ms / 1000) / total_elapsed_s if total_elapsed_s > 0 else 0
            remaining_s = max(0, (total_audio_ms - processed_ms) / 1000)
            eta_s = remaining_s / avg_speed if avg_speed > 0 else 0
            log(
                f"[{idx}/{len(wavs)}] Done {wav.name}: "
                f"new_segments={len(chunk_entries)}, elapsed={format_seconds(chunk_elapsed_s)}, "
                f"overall={processed_ms / total_audio_ms:.1%}, avg_speed={avg_speed:.1f}x, "
                f"eta={format_seconds(eta_s)}"
            )
            if not args.no_partial:
                partial_outputs = write_outputs(
                    all_entries,
                    raw_results,
                    args.output_dir,
                    prefix,
                    args.format,
                    suffix=".partial",
                )
                log(f"[{idx}/{len(wavs)}] Partial outputs updated: {partial_outputs[-1]}")

    outputs = write_outputs(all_entries, raw_results, args.output_dir, prefix, args.format)

    log("Done. Wrote:")
    for path in outputs:
        log(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
