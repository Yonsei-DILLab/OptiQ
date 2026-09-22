"""Check that antmaze/ is the complete, unchanged pinned DDiffPG source tree."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys


def verify(root):
    manifest = json.loads((root / "docs/antmaze-ddiffpg-upstream.json").read_text())
    source = root / "antmaze"
    expected = manifest["files"]
    actual = {p.relative_to(source).as_posix(): p
              for p in source.rglob("*") if p.is_file() or p.is_symlink()}
    errors = []
    if set(actual) != set(expected):
        errors.append({"missing": sorted(set(expected) - set(actual)),
                       "extra": sorted(set(actual) - set(expected))})
    python_files = 0
    for name in sorted(set(actual) & set(expected)):
        path, entry = actual[name], expected[name]
        if path.is_symlink():
            errors.append({"path": name, "error": "unexpected symlink"})
            continue
        data = path.read_bytes()
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        if (hashlib.sha256(data).hexdigest() != entry["sha256"] or
                len(data) != entry["bytes"] or mode != entry["mode"]):
            errors.append({"path": name, "error": "content/size/mode mismatch"})
        if name.endswith(".py"):
            ast.parse(data, filename=name)
            python_files += 1
    result = dict(passed=not errors, upstream=manifest["repository"],
                  commit=manifest["commit"], files=len(actual),
                  expected_files=len(expected), python_files_parsed=python_files,
                  errors=errors)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.exit(verify(args.root))
