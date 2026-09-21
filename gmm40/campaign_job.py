"""Run one committed GMM40 job, with local artifacts primary and W&B mirrors."""
import argparse
import json
import os
from pathlib import Path
import traceback

from . import run
from .target import RESULTS
from .evaluation import atomic_json


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    parser.add_argument('--name',required=True)
    parser.add_argument('--method',required=True)
    parser.add_argument('--seed',type=int,default=0)
    args,_=parser.parse_known_args()
    folder=RESULTS/args.name
    wb=None; warnings=[]
    try:
        import wandb
        wb=wandb.init(entity='OptiQ',project='gmm-trg',group=os.environ['GMM40_CAMPAIGN'],
            name=args.name,job_type='gmm40-fixed-q',tags=['gmm40','fixed-q',args.method],
            config=dict(method=args.method,seed=args.seed,source_commit=os.environ['GMM40_SOURCE_COMMIT']),
            dir=os.environ['GMM40_WANDB_DIR'])
        wb.define_metric('updates')
        wb.define_metric('gmm40/*',step_metric='updates')
        wb.define_metric('train/*',step_metric='updates')
        wb.define_metric('gmm40_mu/*',step_metric='updates')
    except Exception as exc:
        warnings.append('W&B initialization: '+repr(exc))
        print(warnings[-1],flush=True)
    original=run.save_evaluation
    def evaluate(*a,**kw):
        result=original(*a,**kw)
        if wb:
            try:
                wb.config.update(json.loads((folder/'config.json').read_text()),allow_val_change=True)
                mu_path=folder/'evaluations'/f"step_{result['step']:07d}"/'metrics_mu_only.json'
                mu=json.loads(mu_path.read_text()) if mu_path.exists() else {}
                wb.log({'updates':result['step'],
                    **{'gmm40_mu/'+k:v for k,v in mu.items() if isinstance(v,(int,float))},
                    **{'gmm40/'+k:v for k,v in result.items() if isinstance(v,(int,float)) and k!='step'},
                    **{'train/'+k:v for k,v in result['training'].items() if isinstance(v,(int,float))}})
            except Exception as exc:
                warnings.append('W&B evaluation: '+repr(exc))
        return result
    run.save_evaluation=evaluate
    failed=False
    try:
        run.main()
        if wb:
            try:
                latest=json.loads((folder/'latest.json').read_text())
                final=folder/'evaluations'/f"step_{latest['step']:07d}"
                artifact=wandb.Artifact(args.name,type='gmm40-result')
                for p in [folder/'config.json',folder/'latest.json',folder/'model_sizes.json',
                          folder/'update_count_audit.json',final/'samples.npy',final/'samples.png']:
                    artifact.add_file(str(p),name=p.name)
                for name in ['samples_mu_only.npy','metrics_mu_only.json','metrics.json','samples_full_policy.png','visualization.json']:
                    p=final/name
                    if p.exists():artifact.add_file(str(p),name=p.name)
                wb.log_artifact(artifact)
            except Exception as exc:
                warnings.append('W&B final artifact: '+repr(exc))
    except BaseException:
        failed=True
        raise
    finally:
        if wb:
            try:wb.finish(exit_code=int(failed))
            except Exception as exc:warnings.append('W&B finish: '+repr(exc))
        if folder.exists():
            atomic_json(folder/'wandb_status.json',dict(url=wb.url if wb else None,warnings=warnings))


if __name__=='__main__':main()
