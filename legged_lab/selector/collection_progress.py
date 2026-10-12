"""Wall-clock heartbeat and explicit stages for long standalone collection."""

from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


class CollectionProgress:
    """The daemon prints only; all file writes happen on the main thread."""

    def __init__(self, status_path=None, interval_s=30., *, stream=None):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("heartbeat interval must be positive and finite")
        self.status_path = Path(status_path) if status_path is not None else None
        self.interval_s = float(interval_s)
        self.stream = sys.stdout if stream is None else stream
        self._state = {"stage": "starting", "pid": os.getpid()}
        self._stage_started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True,
                                        name="selector-collection-heartbeat")

    def start(self):
        self._thread.start()
        return self

    def set_stage(self, stage, **fields):
        with self._lock:
            self._stage_started = time.monotonic()
            self._state = {"stage": stage, "pid": self._state["pid"], **fields}
            state = dict(self._state)
        # The callback can be called for every replay; write one tiny status
        # record so a second terminal can see the last reached stage.
        if self.status_path is not None:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.status_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n",
                                 encoding="utf-8")
            temporary.replace(self.status_path)

    def announce(self, stage, **fields):
        self.set_stage(stage, **fields)
        print(f"[selector stage] {stage} {json.dumps(fields, ensure_ascii=False)}",
              file=self.stream, flush=True)

    def snapshot(self):
        with self._lock:
            return {**self._state, "stage_wall_s": round(time.monotonic()-self._stage_started, 1)}

    def _heartbeat(self):
        while not self._stop.wait(self.interval_s):
            print(f"[selector heartbeat] {json.dumps(self.snapshot(), ensure_ascii=False)}",
                  file=self.stream, flush=True)

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.)


def close_simulation(application, *, failed, full_cleanup=False):
    """Exit only after the caller has closed all synchronous dataset/report files.

    Older Kit fast exits force status 0, while graceful failure cleanup can
    block indefinitely. Exit the process with status 1 on those versions;
    full_cleanup explicitly opts into Kit's complete teardown instead.
    """
    try:
        signature = inspect.signature(application.close)
        supported = signature.parameters
    except (TypeError, ValueError):
        supported = {}
    if failed and not full_cleanup and "exit_code" not in supported:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    kwargs = {}
    if "wait_for_replicator" in supported:
        # Selector tensors are saved synchronously with torch.save, not by a
        # Replicator writer. Waiting for its camera workflows is unnecessary.
        kwargs["wait_for_replicator"] = False
    fast = not full_cleanup
    if "skip_cleanup" in supported:
        kwargs["skip_cleanup"] = fast
    if "exit_code" in supported:
        kwargs["exit_code"] = 1 if failed else 0
    application.close(**kwargs)
