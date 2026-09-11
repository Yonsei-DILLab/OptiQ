"""Read-only numerical convergence audit of a fixed auxiliary CFM evaluator."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import torch
import wandb

from .idem_evaluate import MyMLP, log_density, sample_cfm, write_json, ROOT
from .target_torch import GMM


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--cfm-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tolerances',type=float,nargs='+',default=[1e-3,1e-4,1e-5])
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    a = p.parse_args()
    cfg = json.loads((a.cfm_run/'config.json').read_text())
    checkpoint = a.cfm_run/'best_cfm.pt'
    for file in [os.environ.get('OPTIQ_ENV_FILE'),ROOT/'.env',ROOT.parent/'.env']:
        if file:load_dotenv(file,override=False)
    a.output.mkdir(parents=True,exist_ok=False)
    record = dict(parent_config=cfg,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  tolerances=a.tolerances,diagnostic_only=True,model_fitted=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run = wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),entity=os.environ.get('WANDB_ENTITY'),
        mode='online',dir=str(a.output),group='gmm40-idem-protocol',job_type='numerical-diagnostic',
        name='CFM-numerics-'+a.cfm_run.name,config=record)
    try:
        torch.set_num_threads(1)
        model = MyMLP().to(a.device)
        saved = torch.load(checkpoint,map_location=a.device,weights_only=True)
        model.load_state_dict(saved['model']);model.eval()
        target = GMM(2,40,40,log_var_scaling=1.,seed=0,device='cpu')
        test = torch.from_numpy(np.load(a.cfm_run/'test_reference.npy')).to(a.device)
        old_samples = torch.from_numpy(np.load(a.cfm_run/'cfm_evaluation_samples.npy')).to(a.device)
        rows = []
        for tolerance in a.tolerances:
            logq,nfes = log_density(model,test,cfg['prior_std'],tolerance)
            for sample_mode in ['fixed_original','resampled_at_tolerance']:
                samples = (old_samples if sample_mode=='fixed_original' else
                    sample_cfm(model,cfg['eval_count'],cfg['prior_std'],a.device,tolerance,cfg['reference_seed']+3))
                sample_logq,_ = log_density(model,samples,cfg['prior_std'],tolerance)
                sample_logp = target.log_prob(samples.cpu()).to(a.device)
                logw = sample_logp-sample_logq
                weights = torch.softmax(logw,0)
                row = dict(tolerance=tolerance,sample_mode=sample_mode,
                    test_nll=float(-logq.mean()),normalized_ess=float(1/(len(samples)*weights.square().sum())),
                    log_z_lower_bound=float(logw.mean()),log_z_logmeanexp=float(torch.logsumexp(logw,0)-np.log(len(samples))),
                    max_normalized_weight=float(weights.max()),test_nfe_mean=float(np.mean(nfes)),
                    max_log_weight=float(logw.max()),cfm_samples_mean_logp=float(sample_logp.mean()))
                rows.append(row);run.log(row)
                with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(json.dumps(row),flush=True)
        record.update(wandb_url=run.url,results=rows,cfm_update=saved['step'])
        write_json(a.output/'summary.json',record)
        artifact = wandb.Artifact('gmm40-cfm-numerics-'+run.id,type='diagnostic')
        artifact.add_file(str(a.output/'summary.json'));run.log_artifact(artifact);run.finish()
    except BaseException as exc:
        write_json(a.output/'failed.json',dict(error=repr(exc),wandb_url=run.url))
        run.finish(exit_code=1)
        raise


if __name__=='__main__':
    main()
