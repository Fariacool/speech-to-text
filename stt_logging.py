"""Small logging helpers shared by the subtitle scripts."""

from __future__ import annotations

import shlex
import time
from datetime import datetime

PROCESS_STARTED_AT = time.monotonic()


def format_seconds(seconds: int | float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def process_elapsed() -> str:
    return format_seconds(time.monotonic() - PROCESS_STARTED_AT)


def log(message: str) -> None:
    print(f"[{timestamp()} elapsed={process_elapsed()}] {message}", flush=True)


def log_step_done(label: str, started_at: float) -> None:
    log(f"{label} completed in {format_seconds(time.monotonic() - started_at)}")


def command_to_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)
