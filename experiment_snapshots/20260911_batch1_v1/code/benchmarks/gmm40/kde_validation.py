"""Diagnostic KDE likelihood on held-out target data; NOT the iDEM paper NLL."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import torch
import wandb

from .target_torch import GMM

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-seed',type=int,default=20261003)
    p.add_argument('--reference-count',type=int,default=5000)
    a=p.parse_args()
    for f in [os.environ.get('OPTIQ_ENV_FILE'),ROOT/'.env',ROOT.parent/'.env']:
        if f:load_dotenv(f,override=False)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg={'samples':str(a.samples),'samples_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest(),
         'reference_seed':a.reference_seed,'reference_count':a.reference_count,
         'bandwidths':[.2,.35,.5],'not_paper_nll':True,'diagnostic_only':True}
    run=wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),entity=os.environ.get('WANDB_ENTITY'),
        mode='online',dir=str(a.output),group='gmm40-idem-protocol',job_type='diagnostic',name='KDE-validation-'+a.samples.parent.name,config=cfg)
    try:
        torch.set_num_threads(1)
        target=GMM(2,40,40,log_var_scaling=1.,seed=0,device='cpu')
        torch.random.default_generator.manual_seed(a.reference_seed)
        reference=target.sample((a.reference_count,)).numpy()
        samples=np.load(a.samples)
        if samples.shape != (100000,2) or not np.isfinite(samples).all():
            raise ValueError('Use exactly 100000 finite direct generator samples')
        totals={h:np.full(len(reference),-np.inf) for h in cfg['bandwidths']}
        for start in range(0,len(samples),2048):
            d2=cdist(reference,samples[start:start+2048],metric='sqeuclidean')
            for h in totals:
                totals[h]=np.logaddexp(totals[h],logsumexp(-d2/(2*h*h),axis=1))
        result={'target_nll':float(-target.log_prob(torch.from_numpy(reference)).mean()),
                'not_paper_nll':True,'wandb_url':run.url,'config':cfg}
        for h,logs in totals.items():
            logs-=np.log(len(samples)*2*np.pi*h*h)
            result[f'kde_nll_h{h}']=float(-logs.mean())
        (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        run.summary.update({k:v for k,v in result.items() if k!='config'})
        artifact=wandb.Artifact('gmm40-kde-diagnostic-'+run.id,type='diagnostic')
        artifact.add_file(str(a.output/'summary.json'));run.log_artifact(artifact);run.finish()
        print(json.dumps(result),flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__=='__main__':main()
