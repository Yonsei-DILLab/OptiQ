"""Repair GIF delay metadata without decoding/re-encoding image data.

ImageIO's Pillow writer uses milliseconds. Older experiment exports passed
seconds and therefore rounded to zero centiseconds. Original files are archived.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from .target import RESULTS


def rewrite_delay(data, milliseconds):
    assert data[:6] in (b"GIF87a", b"GIF89a")
    ticks = milliseconds // 10
    assert 1 <= ticks <= 65535
    offset = 13
    if data[10] & 128:
        offset += 3 * (2 ** ((data[10] & 7) + 1))
    chunks = [data[:offset]]
    pending_control = False
    frames = 0
    inserted = 0
    old_delays = []

    def subblocks_end(start):
        while True:
            size = data[start]
            start += 1
            if size == 0:
                return start
            start += size
            if start > len(data):
                raise ValueError("Truncated GIF subblock")

    while offset < len(data):
        marker = data[offset]
        if marker == 0x3B:
            chunks.append(data[offset:])
            break
        if marker == 0x21:
            end = subblocks_end(offset + 2)
            block = data[offset:end]
            if data[offset + 1] == 0xF9:
                assert len(block) == 8 and block[2] == 4 and block[-1] == 0
                old_delays.append(int.from_bytes(block[4:6], "little") * 10)
                block = block[:4] + ticks.to_bytes(2, "little") + block[6:]
                pending_control = True
            chunks.append(block)
        elif marker == 0x2C:
            if not pending_control:
                chunks.append(b"\x21\xf9\x04\x00" + ticks.to_bytes(2, "little") + b"\x00\x00")
                old_delays.append(None)
                inserted += 1
            end = offset + 10
            packed = data[offset + 9]
            if packed & 128:
                end += 3 * (2 ** ((packed & 7) + 1))
            end = subblocks_end(end + 1)  # LZW minimum code size, then compressed blocks
            chunks.append(data[offset:end])  # image descriptor, palette, LZW bytes unchanged
            pending_control = False
            frames += 1
        else:
            raise ValueError(f"Unexpected GIF block {marker:#x} at {offset}")
        offset = end
    assert frames > 0
    return b"".join(chunks), dict(frames=frames, old_delays_ms=old_delays,
                                  new_delay_ms=milliseconds, inserted_controls=inserted)


def repair(path, milliseconds, min_age=10):
    if time.time() - path.stat().st_mtime < min_age:
        return "deferred"
    original = path.read_bytes()
    corrected, info = rewrite_delay(original, milliseconds)
    if corrected == original:
        return "unchanged"
    checksum = hashlib.sha256(original).hexdigest()
    archive = RESULTS / "diagnostics/gif_timing_originals" / path.relative_to(RESULTS)
    archive = archive.with_name(archive.stem + "." + checksum[:16] + ".gif")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_bytes(original)
    if path.read_bytes() != original:  # a live exporter replaced it meanwhile
        return "deferred"
    temporary = path.with_name(path.name + f".timing-{os.getpid()}.tmp")
    temporary.write_bytes(corrected)
    temporary.replace(path)
    info.update(path=str(path.relative_to(RESULTS)), original=str(archive.relative_to(RESULTS)),
                original_sha256=checksum, corrected_sha256=hashlib.sha256(corrected).hexdigest(),
                timestamp=time.time(), image_data="original palettes and compressed image blocks preserved byte-for-byte")
    with (RESULTS / "diagnostics/gif_timing_repairs.jsonl").open("a") as stream:
        stream.write(json.dumps(info) + "\n")
    return "repaired"


def repair_published_run(folder):
    records = folder / "metrics.jsonl"
    if not records.exists():
        return dict(repaired=0, unchanged=0, deferred=0)
    paths = {}
    for line in records.read_text().splitlines():
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            break
        moving = "updates" in result
        step = result["updates"] if moving else result["step"]
        directory = folder / "evaluations" / f"{'update' if moving else 'step'}_{step:07d}"
        path = directory / ("environment_rollout.gif" if moving else "generation_rollout.gif")
        if path.exists():
            # Native one-step policies have two displayed endpoints; others have >2.
            _, structure = rewrite_delay(path.read_bytes(), 120)
            paths[path] = 120 if moving or structure["frames"] > 2 else 800
    evolution = folder / "training_evolution.gif"
    if evolution.exists():
        paths[evolution] = 1000
    counts = dict(repaired=0, unchanged=0, deferred=0)
    for path, milliseconds in paths.items():
        counts[repair(path, milliseconds)] += 1
    return counts


if __name__ == "__main__":
    queue = json.loads((RESULTS / "queue.json").read_text())
    for job in queue["jobs"]:
        folder = RESULTS / job["name"]
        print(json.dumps(dict(run=job["name"], **repair_published_run(folder))))
