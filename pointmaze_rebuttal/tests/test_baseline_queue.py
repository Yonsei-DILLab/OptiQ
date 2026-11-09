import json
from pathlib import Path
import threading
import time
import sys
import os
import pytest
from pointmaze_rebuttal.baselines.profiles import METHODS, configuration
from pointmaze_rebuttal.baselines.queue import make_plan, execute_stages, dependency_status, run_logged_command


def test_baseline_profiles_and_complete_order(tmp_path):
    plan = make_plan('medium', tmp_path)
    assert len(plan['jobs']) == 30
    for method in METHODS:
        c = configuration(method, 'hard')
        assert (c['utd'], c['batch_size'], c['expected_updates']) == (1, 256, 992000)
    starts, finishes = [], []
    lock = threading.Lock()
    def execute(job):
        with lock:
            for earlier in METHODS[:METHODS.index(job['method'])]:
                assert len([x for x in finishes if x[0] == earlier]) == 5
            if job['seed'] == 4:
                assert (job['method'], 0) in finishes
            starts.append((job['method'], job['seed']))
        time.sleep(.001)
        with lock:
            finishes.append((job['method'], job['seed']))
    execute_stages(plan, execute, lambda: None)
    assert plan['status'] == 'completed' and len(starts) == 30


def test_failure_blocks_later_methods(tmp_path):
    plan = make_plan('hard', tmp_path)
    def fail(job):
        raise RuntimeError('test failure')
    with pytest.raises(RuntimeError, match='pending work'):
        execute_stages(plan, fail, lambda: None)
    assert plan['status'] == 'blocked_by_failure'
    assert all(j['status'] == 'queued' for j in plan['jobs'] if j['method'] != 'sac')


def test_dependency_requires_all_five_completed_results(tmp_path):
    jobs = []
    for seed in range(5):
        out = tmp_path/str(seed)
        out.mkdir()
        (out/'completed.json').write_text(json.dumps(dict(completed=True, updates=992000)))
        jobs.append(dict(seed=seed, status='completed', output=str(out)))
    path = tmp_path/'campaign.json'
    path.write_text(json.dumps(dict(jobs=jobs)))
    assert dependency_status(path)
    jobs[4]['status'] = 'queued'
    path.write_text(json.dumps(dict(jobs=jobs)))
    assert not dependency_status(path)
    jobs[4]['status'] = 'failed'
    path.write_text(json.dumps(dict(jobs=jobs)))
    with pytest.raises(RuntimeError, match='failed or stopped'):
        dependency_status(path)


def test_actual_child_launch_creates_missing_log_directory_without_overwriting(tmp_path):
    log = tmp_path/'not-created'/'logs'/'preflight.log'
    result = tmp_path/'completed.json'
    code = 'from pathlib import Path; Path("completed.json").write_text("{}\\n"); print("child ran")'
    pids = []
    command = [sys.executable, '-c', code]
    assert run_logged_command(command, tmp_path, dict(os.environ), log, pids.append) == 0
    assert result.exists() and 'child ran' in log.read_text() and pids[0] > 0
    with pytest.raises(FileExistsError):
        run_logged_command(command, tmp_path, dict(os.environ), log, pids.append)
