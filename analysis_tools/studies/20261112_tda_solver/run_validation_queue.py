"""Run independent original-N/M pairs on one explicitly assigned GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--indices',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).resolve().parent
    config=json.loads((src/'phase_a.json').read_text())
    manifest=dict(source_commit=a.commit,gpu=a.gpu,cases=[],
                  source_hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in src.iterdir() if q.is_file()})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    failed=False
    for index in map(int,a.indices.split(',')):
        case=config['cases'][index];stem=f'{case["env"]}_s{case["seed"]}_state{case["state"]}_n{config["n"]}'
        data=a.inputs/(stem+'.npz');old=a.out/(stem+'_legacy');new=a.out/(stem+'_verified')
        row=dict(case=case,input_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),stages=[])
        manifest['cases'].append(row)
        commands=[('legacy',[sys.executable,'-u',str(src/'legacy_diagnose.py'),'--data',str(data),
                   '--out',str(old),'--upstream',str(src/'upstream_basin_ph.py'),'--commit',a.commit]),
                  ('verified',[sys.executable,'-u',str(src/'run_verified.py'),'--data',str(data),
                   '--samples',str(old/'samples.npy'),'--out',str(new),'--upstream',str(src/'upstream_basin_ph.py'),'--commit',a.commit])]
        for stage,cmd in commands:
            job=dict(stage=stage,command=cmd,started_unix=time.time());row['stages'].append(job)
            (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print('START',stem,stage,flush=True)
            with (a.out/(stem+'_'+stage+'.log')).open('w') as log:
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                job['pid']=process.pid;rc=process.wait()
            job.update(returncode=rc,finished_unix=time.time())
            (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print('END',stem,stage,rc,flush=True)
            if rc:
                failed=True
                if stage=='legacy':break
    sys.exit(int(failed))


if __name__=='__main__':main()
