#!/usr/bin/env python3
"""Merge diarization speaker turns into FunASR subtitle segments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from stt_logging import format_seconds, log, log_step_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge speaker turns into subtitle segments.")
    parser.add_argument("segments_json", type=Path, help="FunASR *.segments.json file.")
    parser.add_argument("speakers", type=Path, help="Diarization *.speakers.json or *.rttm file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs-final"))
    parser.add_argument("--prefix", help="Output file prefix. Defaults to segments file stem.")
    parser.add_argument("--speaker-map", action="append", default=[], help="Map speaker labels, e.g. SPEAKER_00=Name.")
    parser.add_argument("--format", choices=("srt", "vtt", "txt", "json", "all"), default="all")
    return parser.parse_args()


def ms_to_srt(ms: int | float | None) -> str:
    total = max(0, int(round(float(ms or 0))))
    hours, rest = divmod(total, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def ms_to_vtt(ms: int | float | None) -> str:
    return ms_to_srt(ms).replace(",", ".")


def parse_speaker_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --speaker-map value: {item}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def load_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in data:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "start_ms": int(item["start_ms"]),
                "end_ms": int(item["end_ms"]),
                "text": text,
            }
        )
    return rows


def load_speakers(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [
            {"start": float(row["start"]), "end": float(row["end"]), "speaker": str(row["speaker"])}
            for row in rows
        ]

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        rows.append({"start": start, "end": start + duration, "speaker": speaker})
    return rows


def overlap_seconds(seg_start: float, seg_end: float, spk_start: float, spk_end: float) -> float:
    return max(0.0, min(seg_end, spk_end) - max(seg_start, spk_start))


def nearest_speaker(midpoint: float, speakers: list[dict[str, Any]]) -> str:
    best = ""
    best_distance = float("inf")
    for item in speakers:
        if item["start"] <= midpoint <= item["end"]:
            return item["speaker"]
        distance = min(abs(midpoint - item["start"]), abs(midpoint - item["end"]))
        if distance < best_distance:
            best = item["speaker"]
            best_distance = distance
    return best or "UNKNOWN"


def assign_speakers(
    segments: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    speaker_map: dict[str, str],
) -> list[dict[str, Any]]:
    speakers = sorted(speakers, key=lambda row: (row["start"], row["end"]))
    merged: list[dict[str, Any]] = []
    for segment in segments:
        start_s = segment["start_ms"] / 1000
        end_s = segment["end_ms"] / 1000
        scores: dict[str, float] = {}
        for speaker in speakers:
            if speaker["end"] <= start_s:
                continue
            if speaker["start"] >= end_s:
                break
            score = overlap_seconds(start_s, end_s, speaker["start"], speaker["end"])
            if score > 0:
                scores[speaker["speaker"]] = scores.get(speaker["speaker"], 0.0) + score
        if scores:
            speaker = max(scores.items(), key=lambda item: item[1])[0]
        else:
            speaker = nearest_speaker((start_s + end_s) / 2, speakers)

        speaker = speaker_map.get(speaker, speaker)
        merged.append({**segment, "speaker": speaker, "text": f"[{speaker}] {segment['text']}"})
    return merged


def write_srt(rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = []
    for idx, item in enumerate(rows, 1):
        lines.extend(
            [
                str(idx),
                f"{ms_to_srt(item['start_ms'])} --> {ms_to_srt(item['end_ms'])}",
                item["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for item in rows:
        lines.extend(
            [
                f"{ms_to_vtt(item['start_ms'])} --> {ms_to_vtt(item['end_ms'])}",
                item["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [f"[{ms_to_srt(item['start_ms'])}] {item['text']}" for item in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    script_started = time.monotonic()
    args = parse_args()
    log("Starting merge-speakers.")
    if not args.segments_json.exists():
        raise SystemExit(f"Segments file not found: {args.segments_json}")
    if not args.speakers.exists():
        raise SystemExit(f"Speakers file not found: {args.speakers}")

    log(f"Segments input: {args.segments_json}")
    log(f"Speakers input: {args.speakers}")
    segments_started = time.monotonic()
    segments = load_segments(args.segments_json)
    log_step_done(f"Loaded {len(segments)} ASR segments", segments_started)
    speakers_started = time.monotonic()
    speakers = load_speakers(args.speakers)
    log_step_done(f"Loaded {len(speakers)} speaker turns", speakers_started)
    if not speakers:
        raise SystemExit(f"No speaker turns found in: {args.speakers}")

    assign_started = time.monotonic()
    rows = assign_speakers(segments, speakers, parse_speaker_map(args.speaker_map))
    log_step_done(f"Assigned speakers to {len(rows)} segments", assign_started)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.segments_json.name.replace(".segments.json", "")

    write_started = time.monotonic()
    json_path = args.output_dir / f"{prefix}.speaker_segments.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = [json_path]

    if args.format in ("srt", "all"):
        path = args.output_dir / f"{prefix}.srt"
        write_srt(rows, path)
        outputs.append(path)
    if args.format in ("vtt", "all"):
        path = args.output_dir / f"{prefix}.vtt"
        write_vtt(rows, path)
        outputs.append(path)
    if args.format in ("txt", "all"):
        path = args.output_dir / f"{prefix}.txt"
        write_txt(rows, path)
        outputs.append(path)

    log_step_done("Merged subtitle output write", write_started)
    log("Done. Wrote:")
    for path in outputs:
        log(f"  {path}")
    log(f"Total merge-speakers elapsed: {format_seconds(time.monotonic() - script_started)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
