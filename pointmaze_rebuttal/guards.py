"""Reject anonymous destinations and fingerprint the code actually executed."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def assert_work_branch(path=ROOT):
    result = subprocess.run(['git', '-C', str(path), 'branch', '--show-current'],
                            text=True, capture_output=True, check=True)
    branch = result.stdout.strip()
    if not branch or 'anonymous' in branch.lower():
        raise RuntimeError('PointMaze requires a named non-anonymous work branch')
    return branch


def assert_output_boundary(path):
    parent = Path(path).resolve()
    if parent.is_relative_to(ROOT / 'pointmaze_rebuttal'):
        raise RuntimeError('Generated outputs must stay outside the experiment source folder')
    while not parent.exists():
        parent = parent.parent
    result = subprocess.run(['git', '-C', str(parent), 'rev-parse', '--show-toplevel'],
                            text=True, capture_output=True)
    if result.returncode == 0:
        assert_work_branch(result.stdout.strip())


def code_hashes():
    files = []
    for directory in ('pointmaze_rebuttal',):
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and p.suffix in ('.py', '.json', '.yaml', '.xml')
                     and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}
