#!/usr/bin/env python
"""Poll a wandb sweep, report failed/crashed runs, and (optionally) delete them.

Once a failed run is deleted from wandb, the sweep should re-issue that config
to a future agent (preserving the grid).

Usage:
    python script_cos/sweep_monitor.py <entity/project/sweep_id> [--auto-delete]

Emits one line per event. Each line is a self-contained notification that the
Monitor tool can pick up:
    SUMMARY total=N running=R finished=F failed=X crashed=Y killed=Z
    FAILED  run_id=... state=... config={...}
    DELETED run_id=...
    ERROR   ...
    DONE    all_runs_finished
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import wandb


TERMINAL_FAIL_STATES = {"failed", "crashed", "killed"}


def short_config(cfg: dict) -> dict:
    """Compact relevant fields for one-line logging."""
    interesting = (
        "root-dir",
        "train-data-name",
        "eval-data-name",
        "attribution-method",
        "selection-method",
        "k2",
        "model-id",
    )
    out = {}
    for k in interesting:
        v = cfg.get(k)
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
        if v is not None:
            out[k] = v
    return out


def poll(sweep_id: str, *, auto_delete: bool, poll_seconds: int) -> int:
    api = wandb.Api()
    handled: set[str] = set()  # run ids we've already reported/deleted

    while True:
        try:
            sweep = api.sweep(sweep_id)
            # api caches; refresh runs list
            sweep.load(force=True)
            runs = sweep.runs
        except Exception as e:
            print(f"ERROR poll-failed: {e!s}", flush=True)
            time.sleep(poll_seconds)
            continue

        counts = {"running": 0, "finished": 0, "failed": 0,
                  "crashed": 0, "killed": 0, "pending": 0, "other": 0}
        for r in runs:
            state = (r.state or "").lower()
            if state in counts:
                counts[state] += 1
            else:
                counts["other"] += 1

        total = len(runs)
        expected = getattr(sweep, "expected_run_count", None) or "?"
        print(
            f"SUMMARY total={total} expected={expected} "
            f"running={counts['running']} finished={counts['finished']} "
            f"failed={counts['failed']} crashed={counts['crashed']} "
            f"killed={counts['killed']} pending={counts['pending']}",
            flush=True,
        )

        for r in runs:
            state = (r.state or "").lower()
            if state in TERMINAL_FAIL_STATES and r.id not in handled:
                handled.add(r.id)
                cfg = short_config(r.config or {})
                cfg_json = json.dumps(cfg, sort_keys=True)
                print(
                    f"FAILED run_id={r.id} state={state} name={r.name!r} config={cfg_json}",
                    flush=True,
                )
                if auto_delete:
                    try:
                        r.delete(delete_artifacts=False)
                        print(f"DELETED run_id={r.id}", flush=True)
                    except Exception as e:
                        print(f"ERROR delete-failed run_id={r.id} err={e!s}", flush=True)

        # done condition: no more in-flight runs and reached/exceeded expected
        in_flight = counts["running"] + counts["pending"]
        if expected != "?" and total >= expected and in_flight == 0:
            print("DONE all_runs_terminal", flush=True)
            return 0

        time.sleep(poll_seconds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_id", help="entity/project/sweep_id, e.g. data_attribution/downstream_v2/jic2n61g")
    p.add_argument("--auto-delete", action="store_true",
                   help="Try to delete failed runs (frees the config so the sweep can re-issue it).")
    p.add_argument("--poll-seconds", type=int, default=120)
    args = p.parse_args()

    sys.exit(poll(args.sweep_id, auto_delete=args.auto_delete, poll_seconds=args.poll_seconds))


if __name__ == "__main__":
    main()
