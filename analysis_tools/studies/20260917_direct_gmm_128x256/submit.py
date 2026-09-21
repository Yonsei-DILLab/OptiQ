"""Validation -> five 4-seed arrays, strict afterok order; record every submitted ID."""
import subprocess
import time
from common import ROOT, verify, read, write


def jobs(p, previous=None):
    for stage in ['validate', *p['order']]:
        cmd = ['sbatch', '--parsable', '--partition=' + p['partitions'], '--qos=' + p['qos'],
               '--exclude=' + p['excluded_nodes'], '--job-name=gmm128-' + stage,
               '--output=' + str(ROOT / 'logs' / (stage + '-%A_%a.out'))]
        if stage == 'validate':
            cmd += ['--time=00:20:00']
        else:
            cmd += ['--array=0-3%4']
        if previous:
            cmd += ['--dependency=afterok:' + previous]
        cmd += [str(ROOT / 'job.sbatch'), stage]
        previous = yield stage, cmd


def main():
    p, d = verify()
    assert read(ROOT / 'PREFLIGHT.json')['passed']
    dest = ROOT / 'SUBMISSION.json'
    assert not dest.exists(), 'Already registered (possibly partial); inspect IDs before retrying'
    (ROOT / 'logs').mkdir(exist_ok=True)
    record = dict(commit=d['commit'], source_code_id=p['source_code_id'], total_runs=20,
                  n=p['n'], m=p['m'], temperature=p['temperature'], order=p['order'],
                  seeds=p['seeds'], registered_unix=time.time(), jobs=[])
    write(dest, record)
    gen = jobs(p)
    stage, cmd = next(gen)
    while True:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        record['jobs'].append(dict(stage=stage, job_id=job, command=cmd))
        write(dest, record)
        try:
            stage, cmd = gen.send(job)
        except StopIteration:
            break
    print(dest.read_text())


if __name__ == '__main__':
    main()
