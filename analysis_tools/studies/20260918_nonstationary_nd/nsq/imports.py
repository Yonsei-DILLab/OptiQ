"""Reuse completed common-Q streams without relabeling their numerical provenance."""
import json
from pathlib import Path
from .io import sha

def verify_import(root, name, src, consumer_code):
    manifest=Path(root)/'IMPORTED_Q_SOURCES.json'
    entries=json.loads(manifest.read_text()) if manifest.exists() else {}
    if name not in entries:
        return consumer_code  # Locally generated validation source.
    item=entries[name]
    assert Path(src).resolve()==Path(item['path']).resolve(), 'Unexpected source location'
    assert sha(Path(src)/'COMPLETE.json')==item['complete_sha256'], 'Source completion changed'
    assert sha(Path(src)/'ARTIFACTS_SHA256.json')==item['artifacts_manifest_sha256'], 'Source artifact manifest changed'
    complete=json.loads((Path(src)/'COMPLETE.json').read_text())
    assert complete['commit']==item['commit'] and complete['source_code_id']==item['source_code_id']
    return item['source_code_id']
