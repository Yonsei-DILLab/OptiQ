import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from pointmaze_rebuttal.guards import ROOT, assert_work_branch, assert_output_boundary
from pointmaze_rebuttal.protocol import configuration


def test_historical_budgets_and_smoke_are_explicit():
    ibolt = configuration('ibolt', 'simple')
    dipo = configuration('dipo', 'hard')
    assert (ibolt['steps'], ibolt['num_envs'], ibolt['batch_size'], ibolt['expected_updates']) == (1000192,256,4096,62000)
    assert (dipo['steps'], dipo['num_envs'], dipo['batch_size'], dipo['expected_updates']) == (1001472,2048,4096,15520)
    assert ibolt['utd'] == 1/16 and dipo['utd'] == 1/64
    for method in ('ibolt', 'dipo'):
        cfg = configuration(method, 'medium', smoke=True)
        assert cfg['expected_updates'] == 2 and cfg['steps'] == 8
        assert cfg['final_episodes'] == 5 and cfg['smoke']
    for override in ({'steps':1000000}, {'batch_size':9000}, {'final_episodes':3}):
        with pytest.raises(ValueError):
            configuration('ibolt', 'simple', **override)


def test_anonymous_branch_and_output_are_blocked(tmp_path):
    repo = tmp_path/'protected'
    subprocess.run(['git','init','-q','-b','codex/ibolt-anonymous',str(repo)],check=True)
    with pytest.raises(RuntimeError, match='non-anonymous'):
        assert_work_branch(repo)
    output = repo/'outputs'/'forbidden'
    with pytest.raises(RuntimeError):
        assert_output_boundary(output)
    assert not output.exists()
    subprocess.run(['git','-C',str(repo),'symbolic-ref','HEAD','refs/heads/heechan-rebuttal'],check=True)
    assert assert_work_branch(repo) == 'heechan-rebuttal'
    assert_output_boundary(output)


def test_frozen_core_and_environment_have_only_declared_import_adaptation():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'SOURCE.json').read_text())
    for name, item in manifest['files'].items():
        data = (root/name).read_bytes()
        if name == 'ibolt_core/policy.py':
            data = data.replace(b'from .critic import VectorCritic', b'from models.critic import VectorCritic')
        assert hashlib.sha256(data).hexdigest() == item['sha256'], name
    # Production modules must not depend on the mutable root experiment packages.
    for folder in ('agents','ibolt_core','envs'):
        for path in (root/folder).glob('*.py'):
            text = path.read_text()
            assert 'from pointmaze.' not in text and 'from ibolt.' not in text


def test_generated_outputs_cannot_enter_experiment_source():
    output = ROOT/'pointmaze_rebuttal'/'generated-test-output'
    with pytest.raises(RuntimeError, match='source folder'):
        assert_output_boundary(output)
    assert not output.exists()
