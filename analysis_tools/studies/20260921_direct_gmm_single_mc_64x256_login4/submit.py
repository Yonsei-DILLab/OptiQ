"""One shared GPU configuration check, then five independent Humanoid seeds."""
import subprocess
import time
from common import ROOT, verify, read, write


def main():
    p,d = verify()
    assert read(ROOT/'PREFLIGHT.json')['passed']
    dest = ROOT/'SUBMISSION.json'
    assert not dest.exists(), 'Already attempted submission; inspect IDs before retry'
    (ROOT/'logs').mkdir(exist_ok=True)
    rec = dict(commit=d['commit'], n=64,m=256,k=64,temperature=.25,seeds=p['seeds'],
               total_runs=5,time=time.time(),jobs=[])
    write(dest,rec)
    shared = ['sbatch','--parsable','--partition='+p['partitions'],'--qos='+p['qos'],
              '--exclude='+p['excluded_nodes']]
    stages = [('validate',['--time=00:20:00']), ('humanoid',['--array=0-4%5'])]
    validation = None
    for stage,extra in stages:
        cmd = shared+['--job-name=singleq64x256-'+stage,
            '--output='+str(ROOT/'logs'/(stage+'-%A_%a.out'))]+extra
        if validation:
            cmd += ['--dependency=afterok:'+validation, '--kill-on-invalid-dep=yes']
        cmd += [str(ROOT/'job.sbatch'),stage]
        result = subprocess.run(cmd,capture_output=True,text=True,check=True)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(),result.stdout
        rec['jobs'].append(dict(stage=stage,job_id=job,command=cmd))
        write(dest,rec)
        if stage == 'validate':validation=job
    print(dest.read_text())


if __name__ == '__main__':main()
