"""Prepare and verify pinned source dependencies before scheduling GPU jobs."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

MFPO_PATH = 'gmm40-baseline/MFPO'
MFPO_FILES = ('configs/mfpo_config.py', 'jaxrl5/agents/mean_flow_learner.py')
ROOT = Path(__file__).resolve().parents[1]


def _git(source, *arguments):
    return subprocess.check_output(['git', '-C', str(source), *arguments], text=True).strip()


def _mfpo_revision(source):
    entry = _git(source, 'ls-tree', 'HEAD', '--', MFPO_PATH).split()
    if len(entry) != 4 or entry[:2] != ['160000', 'commit']:
        raise RuntimeError(f'{source}: MFPO must be a pinned Git submodule')
    return entry[2]


def verify_dependencies(source=ROOT, methods=('mfpo',)):
    """Read-only validation; never repair an already running source tree."""
    if 'mfpo' not in methods:
        return {}
    source = Path(source).resolve()
    expected = _mfpo_revision(source)
    dependency = source / MFPO_PATH
    missing = [name for name in MFPO_FILES if not (dependency / name).is_file()]
    if missing:
        raise RuntimeError(
            f'MFPO submodule is not initialized in {source}: missing {missing}. '
            'Prepare this new source before scheduling jobs with '
            f'python -m antmaze_experiments.dependencies --source {source} --prepare')
    # An empty submodule directory otherwise resolves Git commands to its parent.
    if Path(_git(dependency, 'rev-parse', '--show-toplevel')).resolve() != dependency:
        raise RuntimeError(f'MFPO dependency is not its own Git checkout: {dependency}')
    actual = _git(dependency, 'rev-parse', 'HEAD')
    if actual != expected:
        raise RuntimeError(f'MFPO revision mismatch: expected {expected}, found {actual}')
    if _git(dependency, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError(f'MFPO dependency has uncommitted changes: {dependency}')
    return {'mfpo': {'path': MFPO_PATH, 'commit': actual, 'verified': True}}


def prepare_dependencies(source=ROOT, methods=('mfpo',)):
    """Populate an empty submodule in a fresh checkout, retaining the gitlink pin."""
    if 'mfpo' not in methods:
        return {}
    source = Path(source).resolve()
    _mfpo_revision(source)
    dependency = source / MFPO_PATH
    # Refuse to overwrite partially populated, dirty or mismatched dependencies.
    if dependency.exists() and any(dependency.iterdir()):
        return verify_dependencies(source, methods)
    subprocess.run(['git', '-C', str(source), 'submodule', 'update', '--init',
                    '--recursive', '--checkout', '--', MFPO_PATH], check=True)
    return verify_dependencies(source, methods)


def load_mfpo_config(source=ROOT):
    """Load MFPO's own config without relying on the generic `configs` package."""
    source = Path(source).resolve()
    verify_dependencies(source)
    path = source / MFPO_PATH / 'configs/mfpo_config.py'
    spec = importlib.util.spec_from_file_location('antmaze_mfpo_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--methods', nargs='+', choices=['optiq', 'sac', 'dipo', 'mfpo'], default=['mfpo'])
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    action = prepare_dependencies if args.prepare else verify_dependencies
    print(json.dumps(action(args.source, args.methods), indent=2))


if __name__ == '__main__':
    main()
