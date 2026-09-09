"""Concurrent checkpoint reads and sample identity must fail safely."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/gmm40_checkpoint_evaluation_worker.py'
spec = importlib.util.spec_from_file_location('checkpoint_worker', SOURCE)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_checkpoint_waits_for_complete_serialization_record(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'actor_state_10.msgpack'
    checkpoint.write_bytes(b'checkpoint')
    job = dict(checkpoint=str(checkpoint), update=10, parent_program='training')
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='training RUNNING pid 1'))
    timing = tmp_path / 'training_timing.jsonl'
    timing.write_text('{"update":10,"checkpoint":"actor_state_10.msgpack"}')
    assert worker.checkpoint_ready(job)[0] is False
    timing.write_text(timing.read_text() + '\n' + '{"update":20,')
    assert worker.checkpoint_ready(job)[0] is True
    job['update'] = 20
    assert worker.checkpoint_ready(job)[0] is False


def test_final_checkpoint_requires_completed_requested_budget(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'actor_state_final.msgpack'
    checkpoint.write_bytes(b'checkpoint')
    (tmp_path / 'training_timing.jsonl').write_text(json.dumps(dict(update=10, checkpoint=checkpoint.name)) + '\n')
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='training RUNNING pid 1'))
    job = dict(checkpoint=str(checkpoint), update=10, require_completed=True, parent_program='training')
    assert worker.checkpoint_ready(job)[0] is False
    completed = tmp_path / 'completed.json'
    completed.write_text(json.dumps(dict(finished=False, updates=10)))
    assert worker.checkpoint_ready(job)[0] is False
    completed.write_text(json.dumps(dict(finished=True, updates=10)))
    assert worker.checkpoint_ready(job)[0] is True


def test_terminal_parent_without_checkpoint_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='training EXITED'))
    with pytest.raises(RuntimeError, match='terminal'):
        worker.checkpoint_ready(dict(checkpoint=str(tmp_path / 'missing.msgpack'), update=10, parent_program='training'))


def test_export_rejects_tampered_samples_and_wrong_update(tmp_path):
    checkpoint = tmp_path / 'actor_state_10.msgpack'
    checkpoint.write_bytes(b'fixed model')
    export = tmp_path / 'export'
    export.mkdir()
    samples = export / 'samples.npy'
    samples.write_bytes(b'fixed samples')
    info = dict(checkpoint=str(checkpoint), actor_update=10, count=100000, seed=20260930,
                checkpoint_sha256=worker.digest(checkpoint), samples_sha256=worker.digest(samples),
                parent_config=dict(seed=0, temperature=1.0))
    (export / 'summary.json').write_text(json.dumps(info))
    job = dict(checkpoint=str(checkpoint), update=10)
    assert worker.verify_export(job, export) == info['samples_sha256']
    job['expected_config'] = dict(seed=0, temperature=1.0)
    assert worker.verify_export(job, export) == info['samples_sha256']
    job['expected_config']['seed'] = 1
    with pytest.raises(RuntimeError, match='frozen configuration'):
        worker.verify_export(job, export)
    job['expected_config']['seed'] = 0
    job['update'] = 11
    with pytest.raises(RuntimeError, match='identity mismatch'):
        worker.verify_export(job, export)
    job['update'] = 10
    samples.write_bytes(b'changed samples')
    with pytest.raises(RuntimeError, match='identity mismatch'):
        worker.verify_export(job, export)
