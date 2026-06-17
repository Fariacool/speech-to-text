#!/usr/bin/env python3
"""Transcribe long Chinese audio/video files with FunASR Paraformer."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
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


def to_wav(input_path: Path, wav_path: Path) -> None:
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


def split_to_wavs(input_path: Path, out_dir: Path, chunk_seconds: int) -> list[Path]:
    pattern = out_dir / "chunk_%06d.wav"
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
            "model": "paraformer-zh",
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

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise SystemExit("Missing Python package `funasr`. Run `./install.sh` or `uv sync` first.") from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem
    raw_json_path = args.output_dir / f"{prefix}.raw.json"
    entries_json_path = args.output_dir / f"{prefix}.segments.json"

    names = model_names(args.hub, args)
    model_kwargs: dict[str, Any] = {
        "model": names["model"],
        "hub": args.hub,
        "vad_model": names["vad_model"],
        "vad_kwargs": {"max_single_segment_time": args.max_single_segment_ms},
        "punc_model": names["punc_model"],
        "device": args.device,
    }
    if args.spk:
        model_kwargs["spk_model"] = names["spk_model"]

    print(f"Loading FunASR model on {args.device}...")
    model = AutoModel(**model_kwargs)

    with tempfile.TemporaryDirectory(prefix="funasr_subs_") as tmp_name:
        tmp_dir = Path(tmp_name)
        audio_dir = args.output_dir / "audio" if args.keep_audio else tmp_dir
        audio_dir.mkdir(parents=True, exist_ok=True)

        if args.chunk_minutes > 0:
            print(f"Extracting and splitting audio into {args.chunk_minutes} minute chunks...")
            wavs = split_to_wavs(input_path, audio_dir, args.chunk_minutes * 60)
        else:
            wav = audio_dir / f"{prefix}.wav"
            print("Extracting audio...")
            to_wav(input_path, wav)
            wavs = [wav]

        generate_kwargs: dict[str, Any] = {
            "batch_size_s": args.batch_size_s,
            "batch_size_threshold_s": args.batch_threshold_s,
        }
        if args.hotword:
            generate_kwargs["hotword"] = " ".join(args.hotword)

        all_entries: list[dict[str, Any]] = []
        raw_results: list[dict[str, Any]] = []
        offset = 0
        for idx, wav in enumerate(wavs, 1):
            print(f"Transcribing chunk {idx}/{len(wavs)}: {wav.name}")
            result = model.generate(input=str(wav), **generate_kwargs)
            raw_results.append({"file": str(wav), "offset_ms": offset, "result": result})
            all_entries.extend(sentence_entries(result, offset, include_speaker=args.spk))
            if len(wavs) > 1:
                offset += duration_ms(wav)

    entries = normalize_entries(all_entries)
    raw_json_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    entries_json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs: list[Path] = [raw_json_path, entries_json_path]
    if args.format in ("srt", "all"):
        path = args.output_dir / f"{prefix}.srt"
        write_srt(entries, path)
        outputs.append(path)
    if args.format in ("vtt", "all"):
        path = args.output_dir / f"{prefix}.vtt"
        write_vtt(entries, path)
        outputs.append(path)
    if args.format in ("txt", "all"):
        path = args.output_dir / f"{prefix}.txt"
        write_txt(entries, path)
        outputs.append(path)
    if args.format == "json":
        outputs = [raw_json_path, entries_json_path]

    print("Done. Wrote:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
