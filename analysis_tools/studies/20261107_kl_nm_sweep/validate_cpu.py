"""Validate task coverage and resume accounting without allocating a GPU."""
import ast
import contextlib
import io
import json
import tempfile
from pathlib import Path
from common import SOURCE, file_sha, tasks, write
from report import report


def main():
    for path in SOURCE.rglob('*.py'):
        ast.parse(path.read_text())
    declared = json.loads((SOURCE/'TASKS.json').read_text())
    assert declared == {p:[{k:v for k,v in t.items() if k!='config'} for t in tasks(p)]
                        for p in ['pilot','reuse','train']}
    runs=tasks('reuse')+tasks('train')
    assert len(runs)==108 and len({t['id'] for t in runs})==108
    for case in ['t00_reference','n00_spike_ramp']:
        for method in ['forward','reverse']:
            got={(t['n'],t['m'],t['seed']) for t in runs if t['target']==case and t['method']==method}
            assert got=={(n,m,s) for n in [32,128,512] for m in [32,128,512] for s in [0,1,2]}
    archived=json.loads((SOURCE/'ARCHIVED_KERNEL_HASHES.json').read_text())
    for name,sha in archived['files'].items():assert file_sha(SOURCE/name)==sha
    assert archived['files']['experiments/kl_mode_missing_search_1d/core.py']=='3ec41bb4d357f80841de38f6b7a2adc285fa6656f2f06fdd5ef85afe2e9c9574'
    # A failed attempt plus a successful resume must both count once, while
    # overlapping worker reservation is reported separately, not added twice.
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);write(root/'SOURCE_MANIFEST.json',dict(commit='cpu-fixture',files={}))
        task=tasks('train')[0];out=root/'runtime/train'/task['id']
        for i,hours in enumerate([.25,.5]):
            aid=str(i);attempt=out/'attempts'/aid
            data=dict(task,attempt_id=aid,source_commit='cpu-fixture',gpu={'model':'fixture','uuid':'fixture'},
                      allocated_gpu_hours=hours,exit_code=1 if i==0 else 0)
            write(attempt/'BEGIN.json',data);write(attempt/'END.json',data)
            write(out/f'timing_{aid}.json',dict(seconds={'training':hours*1800}))
        write(root/'allocations/fixture.json',dict(allocated_gpu_hours=.8,start_epoch=0))
        with contextlib.redirect_stdout(io.StringIO()):report(root)
        result=json.loads((root/'CAMPAIGN_STATUS.json').read_text())
        assert result['attempt_gpu_hours']==.75
        assert result['synchronized_training_gpu_hours']==.375
        assert result['worker_reserved_gpu_hours']==.8
    print('PASS: syntax, immutable archived kernels, full factorial task coverage, failure/resume cost accounting')


if __name__=='__main__':main()
