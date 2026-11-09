"""Back up authored experiment controls without adding runtime outputs to Git."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'rebuttal_archive'
SKIP = {'__pycache__', 'source', 'payload', 'references', 'remote-verification'}


def selected(path):
    if set(path.parts) & SKIP or path.name.startswith('._'):
        return False
    if path.suffix in ('.py', '.sh'):
        return True
    if 'live-dashboard' in path.parts and path.suffix in ('.html', '.js', '.css', '.svg'):
        return True
    return path.name in ('protocol.json', 'campaign.json') or (
        path.suffix == '.json' and path.name.startswith('queue-config-'))


def main():
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if not branch or 'anonymous' in branch.lower():
        raise RuntimeError('Control-source archives require a non-anonymous branch')
    files = []
    for path in sorted((ROOT / 'outputs').rglob('*')):
        if not path.is_file() or not selected(path.relative_to(ROOT / 'outputs')):
            continue
        relative = path.relative_to(ROOT / 'outputs')
        target = DESTINATION / 'controls' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files.append(dict(original=str(path.relative_to(ROOT)), archived=str(target.relative_to(ROOT)),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
    manifest = dict(format_version=1, private_boundary='Never export to anonymous branches',
                    restore='Copy archived files to their original paths before executing deployment-specific controls',
                    excluded='Checkpoints, logs, caches, downloaded references, generated reports, and run-state/result data',
                    files=files)
    (DESTINATION / 'control-sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Archived {len(files)} control, diagnostic, dashboard, and protocol files')


if __name__ == '__main__':
    main()
