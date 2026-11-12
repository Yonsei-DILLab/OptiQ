"""Verify or restore a hash-pinned experiment source tree; never starts a job."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='20261112')
    parser.add_argument('--snapshot', help='Name from index.json; omit to verify all')
    parser.add_argument('--destination', type=Path, help='New directory for one restored source tree')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / args.date
    index = json.loads((root / 'index.json').read_text())
    names = [args.snapshot] if args.snapshot else sorted(index['snapshots'])
    if args.destination and (not args.snapshot or args.destination.exists()):
        parser.error('--destination requires one --snapshot and a directory that does not exist')
    if any(name not in index['snapshots'] for name in names):
        parser.error('Unknown snapshot; inspect index.json for available names')
    verified = {}
    for name in names:
        snapshot = index['snapshots'][name]
        for filename, digest in snapshot['files'].items():
            relative = PurePosixPath(filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe relative path: {filename}')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid SHA256 digest')
            data = (root / 'objects' / digest).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'Hash mismatch: {name}/{filename}')
        manifest = root / 'manifests' / name / 'source-sha256.json'
        raw = manifest.read_bytes()
        if hashlib.sha256(raw).hexdigest() != snapshot['manifest_sha256']:
            raise ValueError(f'Manifest hash mismatch: {name}')
        if json.loads(raw) != snapshot['files']:
            raise ValueError(f'Manifest contents mismatch: {name}')
        verified[name] = len(snapshot['files'])
    for category in ['operator_files', 'validation_files']:
        for filename, record in index[category].items():
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Archive file hash mismatch: {filename}')
    if args.destination:
        args.destination.mkdir(parents=True, exist_ok=False)
        snapshot = index['snapshots'][args.snapshot]
        for filename, digest in snapshot['files'].items():
            target = args.destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / 'objects' / digest).read_bytes())
        manifest = root / 'manifests' / args.snapshot / 'source-sha256.json'
        (args.destination / 'source-sha256.json').write_bytes(manifest.read_bytes())
    print(json.dumps({'passed': True, 'snapshots': verified,
                      'destination': str(args.destination) if args.destination else None}, indent=2))


if __name__ == '__main__':
    main()
