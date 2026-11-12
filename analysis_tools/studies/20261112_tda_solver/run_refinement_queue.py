"""Refine verified candidates using previously converged identical samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','runs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--indices',required=True);p.add_argument('--gpu',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    src=Path(__file__).resolve().parent;cases=json.loads((src/'phase_a.json').read_text())['cases']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    rows=[];failed=False
    for index in map(int,a.indices.split(',')):
        case=cases[index];stem=f'{case["env"]}_s{case["seed"]}_state{case["state"]}_n32768'
        old=next(a.runs.glob(f'validation-ca3f7e1-gpu*/{stem}_legacy'))
        prior=next(a.runs.glob(f'validation-ca3f7e1-gpu*/{stem}_verified'))
        proof=json.loads((prior/'result.json').read_text())
        if not proof['validation_passed']:raise ValueError('Prior endpoints did not pass')
        data=a.inputs/(stem+'.npz');samples=old/'samples.npy';cache=a.out/(stem+'_endpoints.npz')
        with np.load(prior/'endpoints.npz') as f:
            np.savez(cache,points=f['points'],residual=f['residual'],data_sha256=digest(data),samples_sha256=digest(samples))
        out=a.out/(stem+'_verified')
        cmd=[sys.executable,'-u',str(src/'run_verified.py'),'--data',str(data),'--samples',str(samples),
             '--endpoints',str(cache),'--out',str(out),'--upstream',str(src/'upstream_basin_ph.py'),'--commit',a.commit]
        row=dict(case=case,source_commit=a.commit,command=cmd,original_endpoint_sha256=digest(prior/'endpoints.npz'),
                 prior_result_sha256=digest(prior/'result.json'),derived_cache_sha256=digest(cache),started_unix=time.time())
        rows.append(row);(a.out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('START',stem,flush=True)
        with (a.out/(stem+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);row['pid']=child.pid;rc=child.wait()
        row.update(returncode=rc,finished_unix=time.time());(a.out/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('END',stem,rc,flush=True);failed|=bool(rc)
    sys.exit(int(failed))


if __name__=='__main__':main()
