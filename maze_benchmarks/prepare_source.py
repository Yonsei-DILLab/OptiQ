"""Make an immutable benchmark source tree including pinned upstream adapters.

Git archives omit submodule contents. This exporter verifies the two gitlinks,
then copies only the source needed by MFPO and MEOW from their pinned commits.
No Git metadata, credentials, checkpoints or generated logs enter the snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


DEPENDENCIES = {
    "gmm40-baseline/DIPO": ("c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc", ["agent"]),
    "gmm40-baseline/MFPO": ("d8b3977d29d4ef2d315e871337e5826f2eb79eb2", ["configs", "jaxrl5"]),
    "gmm40-baseline/meow": ("b786d27aa9b03e4242ee8904ff884b21fe65e2f7", ["cleanrl/cleanrl"]),
}


def git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *arguments], text=True).strip()


def export(source: Path, output: Path, commit: str):
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to mutate a frozen source: {output}")
    if git(source, "rev-parse", "HEAD") != commit:
        raise ValueError("requested source commit is not checked out")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked source has uncommitted edits")

    revisions = {}
    for relative, (expected, paths) in DEPENDENCIES.items():
        tree = git(source, "ls-tree", commit, "--", relative).split()
        if len(tree) != 4 or tree[2] != expected:
            raise ValueError(f"{relative}: gitlink does not match pinned revision")
        checkout = source / relative
        if git(checkout, "rev-parse", "HEAD") != expected:
            raise ValueError(f"{relative}: checkout does not match pinned revision")
        if git(checkout, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"{relative}: upstream checkout has tracked changes")
        revisions[relative] = dict(commit=expected, exported_paths=paths)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="maze-source-", dir=output.parent) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        with (Path(temporary) / "main.tar").open("wb") as stream:
            subprocess.run(["git", "-C", str(source), "archive", "--format=tar", commit],
                           check=True, stdout=stream)
        with tarfile.open(Path(temporary) / "main.tar") as archive:
            archive.extractall(staging, filter="data")
        for relative, detail in revisions.items():
            tar_path = Path(temporary) / (relative.split("/")[-1] + ".tar")
            with tar_path.open("wb") as stream:
                subprocess.run(["git", "-C", str(source / relative), "archive", "--format=tar",
                                detail["commit"], *detail["exported_paths"]],
                               check=True, stdout=stream)
            with tarfile.open(tar_path) as archive:
                archive.extractall(staging / relative, filter="data")

        # Verify the complete committed training tree as well as pinned
        # submodule contents; the manifest itself is added afterwards.
        checksums = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                checksums[str(path.relative_to(staging))] = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar = dict(source_commit=commit, dependencies=revisions, sha256=checksums)
        (staging / "maze_source_manifest.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        shutil.move(str(staging), str(output))
    return sidecar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    arguments = parser.parse_args()
    result = export(arguments.source, arguments.output, arguments.commit)
    print(json.dumps(dict(source_commit=result["source_commit"],
                          dependencies=result["dependencies"],
                          files=len(result["sha256"])), indent=2))
