"""Paired N/M pilot with common random numbers and validated endpoint caching."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    for name in ('inputs','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--gpu',required=True);p.add_argument('--indices',required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    src=Path(__file__).resolve().parent;cfg=json.loads((src/'convergence.json').read_text())
    cases=json.loads((src/'phase_a.json').read_text())['cases']
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=dict(source_commit=a.commit,gpu=a.gpu,config=cfg,jobs=[],
                  source_hashes={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in src.iterdir() if q.is_file()},
                  input_manifest_sha256=hashlib.sha256((a.inputs/'manifest.json').read_bytes()).hexdigest())
    failed=False
    for index in map(int,a.indices.split(',')):
        case=cases[index];name=f'{case["env"]}_s{case["seed"]}_state{case["state"]}'
        gen=torch.Generator(device='cuda').manual_seed(cfg['sampling_seed']+index)
        dim=3 if case['env']=='hopper' else 17
        comp=torch.randint(max(cfg['n_values']),(max(cfg['m_values']),),device='cuda',generator=gen)
        uniform=torch.rand((max(cfg['m_values']),dim),dtype=torch.float64,device='cuda',generator=gen)
        for n in cfg['n_values']:
            data=a.inputs/f'{name}_n{n}.npz'
            with np.load(data) as f:
                mu=torch.tensor(f['mu'][0],dtype=torch.float64,device='cuda')[comp%n]
                sd=torch.tensor(f['log_std'][0],dtype=torch.float64,device='cuda')[comp%n].exp()
            lo=torch.special.ndtr((-1-mu)/sd);hi=torch.special.ndtr((1-mu)/sd)
            u=(lo+(hi-lo)*uniform).clamp(1e-12,1-1e-12)
            x=(mu+sd*math.sqrt(2)*torch.erfinv(2*u-1)).clamp(-1,1)
            samplefile=a.out/f'{name}_n{n}_samples.npy';np.save(samplefile,x.cpu().numpy())
            cached=None
            for m in sorted(cfg['m_values'],reverse=True):
                out=a.out/f'{name}_n{n}_m{m}'
                cmd=[sys.executable,'-u',str(src/'run_verified.py'),'--data',str(data),'--samples',str(samplefile),
                     '--limit',str(m),'--out',str(out),'--upstream',str(src/'upstream_basin_ph.py'),'--commit',a.commit,'--k',str(cfg['k'])]
                if cached:cmd+=['--endpoints',str(cached)]
                job=dict(case=case,n=n,m=m,command=cmd,started_unix=time.time());manifest['jobs'].append(job)
                (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print('START',out.name,flush=True)
                with (a.out/(out.name+'.log')).open('w') as log:
                    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);job['pid']=process.pid;rc=process.wait()
                job.update(returncode=rc,finished_unix=time.time())
                (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print('END',out.name,rc,flush=True)
                if rc:failed=True
                elif cached is None:cached=out/'endpoints.npz'
    sys.exit(int(failed))


if __name__=='__main__':main()
