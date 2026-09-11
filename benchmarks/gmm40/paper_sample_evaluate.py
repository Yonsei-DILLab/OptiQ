"""Evaluate direct generator samples and released baselines with paper metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import torch
import wandb

from .target_torch import GMM
from .metrics import evaluate_sample_tensor
from .paper_metrics import repeated_sample_metrics

ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--label',default='OptiQ tuning candidate')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-seed',type=int,default=20261001)
    p.add_argument('--count',type=int,default=1000)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--baselines',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--confirmation',action='store_true',
        help='Label a fixed-recipe independent-seed evaluation; metric calculations are unchanged')
    a=p.parse_args()
    for file in [os.environ.get('OPTIQ_ENV_FILE'),ROOT/'.env',ROOT.parent/'.env']:
        if file:load_dotenv(file,override=False)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg.update(samples_sha256=hashlib.sha256(a.samples.read_bytes()).hexdigest(),
        samples_are_direct_generator_outputs=True,tuning_only=not a.confirmation,
        spatial_tv_bins_per_axis=200,spatial_tv_ranges=['reference-derived','fixed [-56,56]^2'],
        nll_and_ess_are_not_measured_by_this_script=True)
    cfg['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'benchmarks/gmm40/paper_metrics.py',ROOT/'benchmarks/gmm40/metrics.py',ROOT/'benchmarks/gmm40/target_torch.py']}
    run=wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'),mode='online',dir=str(a.output),
        group='gmm40-idem-protocol',job_type='confirmation-sample-evaluation' if a.confirmation else 'sample-evaluation',name=a.label,config=cfg)
    try:
        torch.set_num_threads(1)
        target=GMM(2,40,40,log_var_scaling=1.,seed=0,device='cpu')
        samples=torch.from_numpy(np.load(a.samples)).float()
        sets={a.label:samples}
        sources={a.label:str(a.samples.resolve())}
        if a.baselines:
            for label,name in [('DiKL','dikl'),('iDEM','idem')]:
                path=ROOT/'outputs/gmm40/reference'/f'{name}_samples.pt'
                sets[label]=torch.load(path,map_location='cpu',weights_only=True).detach().float()
                sources[label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(a.reference_seed+100)
            sets['GT independent']=target.sample((10000,))
        records=[]
        for label,points in sets.items():
            if len(points)<10000 or not torch.isfinite(points).all():
                raise ValueError('Need at least 10000 finite direct samples per method')
            common=evaluate_sample_tensor(points[:10000],target,a.reference_seed)
            repeated=repeated_sample_metrics(points,target,a.reference_seed+1,a.count,a.repeats)
            record={'method':label,**{k.removeprefix('eval/'):v for k,v in common.items()},'repeats':repeated}
            for metric in ['w2','spatial_tv','spatial_tv_fixed_bounds']:
                values=[r[metric] for r in repeated]
                record[metric+'_mean']=float(np.mean(values))
                record[metric+'_resampling_sd']=float(np.std(values,ddof=1))
            records.append(record)
            run.log({label+'/'+k:v for k,v in record.items() if isinstance(v,(int,float))})
            print(json.dumps({k:v for k,v in record.items() if k!='repeats'}),flush=True)
        scope_note = ('One independent training seed under a fixed recipe. Aggregate all planned seeds before claiming replicated performance.'
                      if a.confirmation else 'Tuning snapshot only.')
        summary={'config':cfg,'sources':sources,'records':records,'wandb_url':run.url,
                 'interpretation':scope_note+' Paper likelihood metrics require the separate OT-CFM evaluator. Spatial TV range sensitivity is disclosed; published bounds are underspecified.'}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        rows=['# GMM40 paper sample metrics','',
              'Independent direct g(z) samples. '+scope_note,
              '', '| Method | Mean log p | GT reference | Absolute error | Modes /40 | W2 (1,000 points) | Spatial TV, reference bounds | Spatial TV, fixed bounds |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in records:
            rows.append(f"| {r['method']} | {r['mean_log_prob']:.4f} | {r['reference_mean_log_prob']:.4f} | {r['mean_log_prob_abs_error']:.4f} | {r['modes_covered']} | {r['w2_mean']:.4f} | {r['spatial_tv_mean']:.4f} | {r['spatial_tv_fixed_bounds_mean']:.4f} |")
        rows+=['','W2 and both 2D TV values average 10 repeats. TV uses 200×200 bins and preserves overflow mass. The fixed range is [-56,56]²; the other uses each reference sample range. NLL, importance ESS and log Z are measured separately using the iDEM OT-CFM protocol.',f'','[W&B]('+run.url+')']
        (a.output/'analysis.md').write_text('\n'.join(rows)+'\n')
        artifact=wandb.Artifact('gmm40-paper-samples-'+run.id,type='gmm40-evaluation')
        for name in ['summary.json','analysis.md']:artifact.add_file(str(a.output/name))
        run.log_artifact(artifact);run.finish()
    except BaseException as exc:
        (a.output/'failed.json').write_text(json.dumps({'error':repr(exc),'wandb_url':run.url})+'\n')
        run.finish(exit_code=1)
        raise


if __name__=='__main__':
    main()
