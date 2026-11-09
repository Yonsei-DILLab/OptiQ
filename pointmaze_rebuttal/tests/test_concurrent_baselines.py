import collections
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from pointmaze_rebuttal.baselines.concurrent import gpu_assignment, admission, process_info, wait_adopted, GIB
from pointmaze_rebuttal.baselines.profiles import METHODS


def test_all_thirty_jobs_distribute_over_all_four_gpus():
    counts = collections.Counter(gpu_assignment(m, s) for m in METHODS for s in range(5))
    assert [counts[g] for g in range(4)] == [8, 8, 7, 7]
    assert [gpu_assignment('sac', s) for s in range(4)] == list(range(4))


def test_memory_reservations_prevent_admission_before_allocations_show_up():
    job = dict(gpu=0)
    current = dict(gpus={0:dict(uuid='gpu',used=GIB,total=32*GIB)}, apps={},host_used=20*GIB,host_max=360*GIB)
    assert admission(job, [], current)[0]
    reserved = [dict(gpu=0, reserved=True) for _ in range(9)]
    assert not admission(job, reserved, current)[0]
    current['host_used'] = 330*GIB
    assert not admission(job, [], current)[0]


@pytest.mark.skipif(not Path('/proc/self/stat').exists(), reason='Linux process adoption')
def test_adopts_an_existing_child_without_restart(tmp_path):
    expected=7
    code='import json,time;from pathlib import Path;time.sleep(.1);Path("completed.json").write_text(json.dumps(dict(completed=True,backend="gpu",updates=7)))'
    child=subprocess.Popen([sys.executable,'-c',code],cwd=tmp_path)
    info=process_info(child.pid)
    job=dict(pid=child.pid,start_ticks=info['start_ticks'],output=str(tmp_path),settings=dict(expected_updates=expected))
    result=wait_adopted(job,interval=.01)
    assert child.wait()==0 and result['updates']==expected
