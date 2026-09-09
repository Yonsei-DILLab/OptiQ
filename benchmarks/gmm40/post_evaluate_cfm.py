"""Evaluate a fixed CFM checkpoint with public generation batches and tolerance checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import torch
import wandb

from .idem_evaluate import MyMLP,log_density,sample_cfm,write_json,ROOT
from .target_torch import GMM


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--cfm-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tolerances',type=float,nargs='+',default=[1e-3,1e-4])
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--ess-batch16-repeats',type=int,default=0,
        help='Also evaluate independent16-sample ESS batches, matching the authors\' disclosed original paper sample count')
    p.add_argument('--reference-seed',type=int,help='Fresh test and sampling seeds; omit to reuse the original evaluation draws')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args()
    if a.batch_size < 1 or a.ess_batch16_repeats < 0 or any(t <= 0 for t in a.tolerances):
        p.error('Batch sizes/tolerances must be positive and repeat count nonnegative')
    cfg=json.loads((a.cfm_run/'config.json').read_text())
    checkpoint=a.cfm_run/'best_cfm.pt'
    for file in [os.environ.get('OPTIQ_ENV_FILE'),ROOT/'.env',ROOT.parent/'.env']:
        if file:load_dotenv(file,override=False)
    a.output.mkdir(parents=True,exist_ok=False)
    record=dict(parent_config=cfg,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        tolerances=a.tolerances,sampling_batch_size=a.batch_size,likelihood_batch_size=a.batch_size,
        reference_seed=a.reference_seed,evaluation_only=True,model_fitted=False,
        primary_tolerance=1e-3,secondary_tolerance=1e-4,
        ess_batch16_repeats=a.ess_batch16_repeats,
        ess_count_disclosure='Authors README states original paper ESS used16 samples; updated recommendation is1000. Both are reported when requested; no favorable batch-size selection.',
        ess_count_source='https://github.com/jarridrb/DEM#ess-computation-considerations',
        protocol_note=f'Generation and likelihood batches{a.batch_size};{cfg["eval_count"]} total primary samples. Paper tolerance and stricter checks are reported without choosing the favorable result.',
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),ROOT/'benchmarks/gmm40/idem_evaluate.py']})
    run=wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),entity=os.environ.get('WANDB_ENTITY'),
        mode='online',dir=str(a.output),group='gmm40-idem-protocol',job_type='cfm-checkpoint-evaluation',
        name='CFM-batched-'+a.cfm_run.name,config=record)
    try:
        torch.set_num_threads(1)
        model=MyMLP().to(a.device)
        saved=torch.load(checkpoint,map_location=a.device,weights_only=True)
        model.load_state_dict(saved['model']);model.eval()
        target=GMM(2,40,40,log_var_scaling=1.,seed=0,device='cpu')
        if a.reference_seed is None:
            test=torch.from_numpy(np.load(a.cfm_run/'test_reference.npy')).to(a.device)
            sample_seed=cfg['reference_seed']+3
        else:
            torch.random.default_generator.manual_seed(a.reference_seed)
            test=target.sample((cfg['eval_count'],)).to(a.device)
            sample_seed=a.reference_seed+1
        results=[]
        for tolerance in a.tolerances:
            logq,nfes=log_density(model,test,cfg['prior_std'],tolerance,a.batch_size)
            samples=sample_cfm(model,len(test),cfg['prior_std'],a.device,tolerance,sample_seed,a.batch_size)
            sample_logq,_=log_density(model,samples,cfg['prior_std'],tolerance,a.batch_size)
            logp=target.log_prob(samples.cpu()).to(a.device)
            logw=logp-sample_logq;weights=torch.softmax(logw,0)
            row=dict(tolerance=tolerance,test_nll=float(-logq.mean()),
                normalized_ess=float(1/(len(samples)*weights.square().sum())),
                log_z_lower_bound=float(logw.mean()),log_z_logmeanexp=float(torch.logsumexp(logw,0)-np.log(len(samples))),
                max_normalized_weight=float(weights.max()),nfe_mean=float(np.mean(nfes)),
                target_test_nll=float(-target.log_prob(test.cpu()).mean()),
                cfm_samples_mean_logp=float(logp.mean()))
            np.save(a.output/f'samples_tol{tolerance}.npy',samples.cpu().numpy())
            np.save(a.output/f'sample_logp_tol{tolerance}.npy',logp.cpu().numpy())
            np.save(a.output/f'sample_logq_tol{tolerance}.npy',sample_logq.cpu().numpy())
            np.save(a.output/f'test_logq_tol{tolerance}.npy',logq.cpu().numpy())
            small_batches=[]
            for i in range(a.ess_batch16_repeats):
                small=sample_cfm(model,16,cfg['prior_std'],a.device,tolerance,sample_seed+10000+i,16)
                small_logq,_=log_density(model,small,cfg['prior_std'],tolerance,16)
                small_logp=target.log_prob(small.cpu()).to(a.device)
                small_logw=small_logp-small_logq
                value=float(1/(16*torch.softmax(small_logw,0).square().sum()))
                small_batches.append(dict(seed=sample_seed+10000+i,normalized_ess=value,
                    log_z_lower_bound=float(small_logw.mean())))
                np.savez(a.output/f'ess16_tol{tolerance}_repeat{i}.npz',
                    samples=small.cpu().numpy(),logp=small_logp.cpu().numpy(),logq=small_logq.cpu().numpy())
            if small_batches:
                values=[b['normalized_ess'] for b in small_batches]
                row.update(normalized_ess_batch16_mean=float(np.mean(values)),
                    normalized_ess_batch16_resampling_sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                    ess_batch16_repeats=len(values),ess_batch16_results=small_batches)
            results.append(row);run.log(row)
            with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
        record.update(wandb_url=run.url,results=results,cfm_update=saved['step'],sample_seed=sample_seed)
        write_json(a.output/'summary.json',record)
        artifact=wandb.Artifact('gmm40-cfm-batched-'+run.id,type='gmm40-evaluation')
        artifact.add_file(str(a.output/'summary.json'));run.log_artifact(artifact);run.finish()
    except BaseException as exc:
        write_json(a.output/'failed.json',dict(error=repr(exc),wandb_url=run.url))
        run.finish(exit_code=1)
        raise


if __name__=='__main__':
    main()
