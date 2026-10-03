"""Structured run logging (brief SS6): config hash, git SHA, timestamp, wall time,
molecules evaluated. One JSONL record appended per run.
"""

from __future__ import annotations

import json
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@contextmanager
def run_log(log_path: str | Path, config_hash: str, run_name: str) -> Iterator[dict]:
    """Yield a mutable state dict, written out as one JSONL record on exit.

    Any key the caller sets is recorded, not just the ones named here: run
    provenance is write-once (a run cannot be re-interrogated after the fact
    for something it never wrote down), so silently dropping unknown keys would
    lose exactly the facts that matter later. `molecules_evaluated` is
    pre-seeded so a run that fails early still produces a well-formed record.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"molecules_evaluated": 0}
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        record = {
            "run_name": run_name,
            "config_hash": config_hash,
            "git_sha": git_sha(),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": round(time.perf_counter() - t0, 3),
            **state,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
