"""Wait for both the GPU lock and truly idle hardware before a queued job.

Older AntMaze processes on these hosts do not all hold the shared flock. Their
GPU processes must therefore disappear before a new worker claims a slot.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


POLL_SECONDS = 5.0
IDLE_SECONDS = 20.0


def nvidia_query(*arguments: str) -> list[str]:
    output = subprocess.check_output(["nvidia-smi", *arguments], text=True,
                                     stderr=subprocess.PIPE, timeout=20)
    return [line.strip() for line in output.splitlines() if line.strip()]


def gpu_uuid(index: int) -> str:
    for line in nvidia_query(f"--id={index}", "--query-gpu=index,uuid", "--format=csv,noheader"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and int(parts[0]) == index:
            return parts[1]
    raise RuntimeError(f"GPU {index} is absent from nvidia-smi")


def compute_pids(uuid: str) -> set[int]:
    entries = set()
    for line in nvidia_query(f"--id={uuid}", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == uuid:
            entries.add(int(parts[1]))
    return entries


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("missing guarded command")
    uuid = gpu_uuid(args.gpu)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    idle_since = None
    last_report = 0.0
    with args.lock.open("a+") as lock:
        while True:
            now = time.monotonic()
            pids = compute_pids(uuid)
            if pids:
                idle_since = None
            elif idle_since is None:
                idle_since = now
            if not pids and now - idle_since >= IDLE_SECONDS:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    # Recheck after taking the lock. Existing unguarded jobs
                    # can hold the GPU even when no lock holder exists.
                    if not compute_pids(uuid):
                        print(f"GPU {args.gpu} idle and locked; starting {command}", flush=True)
                        return subprocess.run(command, check=False,
                                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)).returncode
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    idle_since = None
            if now - last_report >= 60:
                print(f"waiting for GPU {args.gpu}: compute PIDs {sorted(pids)}",
                      flush=True)
                last_report = now
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
