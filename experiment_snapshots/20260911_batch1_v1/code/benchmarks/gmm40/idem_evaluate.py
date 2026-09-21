"""Fit the paper's auxiliary OT-CFM evaluator to fixed sampler outputs.

This model is evaluation machinery only: its outputs never replace OptiQ g(z).
Uses unmodified iDEM MyMLP/CNF sources, exact minibatch OT and exact divergence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from dotenv import load_dotenv
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torchdiffeq import odeint
import wandb

from .idem_reference.mlp import MyMLP
from .idem_reference.cnf import CNF
from .target_torch import GMM

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def log_density(model, points, prior_std=50., tolerance=1e-3, batch_size=256):
    cnf = CNF(model, is_diffusion=False, use_exact_likelihood=True,
              method='dopri5', atol=tolerance, rtol=tolerance,
              max_steps_till_fallback=10000)
    logs, nfes = [], []
    for x in points.split(batch_size):
        cnf.nfe = 0
        augmented = torch.cat([x, torch.zeros_like(x[:, :1])], dim=-1)
        result = cnf.integrate(augmented)[-1]
        z, logdet = result[:, :2], result[:, 2]
        prior_logp = -.5 * (z / prior_std).square().sum(-1) - np.log(2*np.pi*prior_std**2)
        logs.append((prior_logp + logdet).detach())
        nfes.append(cnf.nfe)
    return torch.cat(logs), nfes


@torch.no_grad()
def sample_cfm(model, count, prior_std, device, tolerance, seed, batch_size=None):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        torch.manual_seed(seed)
        z = torch.randn(count, 2, device=device) * prior_std
    def vector(t, x):
        return model(torch.ones(x.shape[0], device=x.device)*t, x)
    # Match the public evaluator's generation batches when requested. Keeping
    # None preserves the historical full-batch path for numerical comparisons.
    batch_size = count if batch_size is None else batch_size
    return torch.cat([odeint(vector, chunk, torch.tensor([0., 1.], device=device),
                   method='dopri5', atol=tolerance, rtol=tolerance)[-1]
                   for chunk in z.split(batch_size)],dim=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--samples', type=Path)
    parser.add_argument('--gt-control', action='store_true', help='Explicit evaluator calibration, never an OptiQ result')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--updates', type=int, default=100000)
    parser.add_argument('--train-count', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--eval-count', type=int, default=1000)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--learning-rate', type=float, default=.0005)
    parser.add_argument('--prior-std', type=float, default=50.)
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--sampling-batch-size',type=int,default=256)
    parser.add_argument('--reference-seed', type=int, default=20260921)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--job-type', default='cfm-evaluation')
    args = parser.parse_args()
    if bool(args.samples) == args.gt_control:
        parser.error('Select either a fixed sample file or the explicit GT control')
    for file in [os.environ.get('OPTIQ_ENV_FILE'), ROOT/'.env', ROOT.parent/'.env']:
        if file:
            load_dotenv(file, override=False)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    cfg.update(hostname=socket.gethostname(), git_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=list(__import__('sys').argv), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        evaluation_only=True, model='unmodified DEM MyMLP, hidden_size=128, hidden_layers=3, sinusoidal embeddings=128',
        solver='unmodified DEM CNF, exact divergence, dopri5', target_samples_used_for_sampler_training=False)
    if args.samples:
        cfg['samples_sha256'] = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    cfg['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT/'benchmarks/gmm40/idem_reference/mlp.py', ROOT/'benchmarks/gmm40/idem_reference/cnf.py']}
    run = wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'), mode='online', dir=str(args.output),
        group='gmm40-idem-protocol', job_type=args.job_type,
        name=f"cfm-{'GT-control' if args.gt_control else args.samples.stem}-seed{args.seed}", config=cfg)
    cfg['wandb_url'] = run.url
    write_json(args.output/'config.json', cfg)
    print(json.dumps({'wandb_url':run.url,'output':str(args.output)}), flush=True)
    try:
        torch.set_num_threads(1)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA required for CFM evaluation')
        target = GMM(2,40,40,log_var_scaling=1.,seed=0,device='cpu')
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.reference_seed)
            validation = target.sample((args.eval_count,)).to(args.device)
            torch.random.default_generator.manual_seed(args.reference_seed+1)
            test = target.sample((args.eval_count,)).to(args.device)
            if args.gt_control:
                torch.random.default_generator.manual_seed(args.reference_seed+2)
                training = target.sample((args.train_count,))
            elif args.samples.suffix == '.npy':
                training = torch.from_numpy(np.load(args.samples)).float()
            else:
                training = torch.load(args.samples,map_location='cpu',weights_only=True).float()
        if training.shape != (args.train_count,2) or not torch.isfinite(training).all():
            raise ValueError('Expected exactly train-count finite 2D samples in ORIGINAL coordinates')
        training = training.to(args.device)
        np.save(args.output/'validation_reference.npy',validation.cpu().numpy())
        np.save(args.output/'test_reference.npy',test.cpu().numpy())
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = MyMLP().to(args.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
        best = float('inf')
        started = time.monotonic()
        for step in range(1,args.updates+1):
            model.train()
            x0 = torch.randn(args.batch_size,2,device=args.device)*args.prior_std
            x1 = training[torch.randint(len(training),(args.batch_size,),device=args.device)]
            cost = torch.cdist(x0,x1).square().detach().cpu().numpy()
            rows, columns = linear_sum_assignment(cost)
            # Exact equal-mass OT pairs; random t supplies the CFM interpolation.
            x0, x1 = x0[rows], x1[columns]
            t = torch.rand(args.batch_size,device=args.device)
            xt = (1-t[:,None])*x0+t[:,None]*x1
            loss = (model(t,xt)-(x1-x0)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite CFM training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                run.log({'cfm_update':step,'cfm_loss':float(loss.detach()),'elapsed_seconds':time.monotonic()-started})
            if step % args.eval_interval == 0 or step == args.updates:
                model.eval()
                logq, nfes = log_density(model,validation,args.prior_std,args.tolerance)
                nll = float(-logq.mean())
                row = {'cfm_update':step,'validation_nll':nll,'nfe_mean':float(np.mean(nfes)),
                       'elapsed_seconds':time.monotonic()-started}
                run.log(row)
                with (args.output/'history.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                if nll < best:
                    best = nll
                    torch.save({'model':model.state_dict(),'step':step,'validation_nll':nll},args.output/'best_cfm.pt')
        saved = torch.load(args.output/'best_cfm.pt',map_location=args.device,weights_only=True)
        model.load_state_dict(saved['model']); model.eval()
        logq_test, nfes = log_density(model,test,args.prior_std,args.tolerance)
        cfm_samples = sample_cfm(model,args.eval_count,args.prior_std,args.device,args.tolerance,args.reference_seed+3,args.sampling_batch_size)
        logq_cfm, _ = log_density(model,cfm_samples,args.prior_std,args.tolerance)
        logp_cfm = target.log_prob(cfm_samples.cpu()).to(args.device)
        logw = logp_cfm-logq_cfm
        ess = 1/(args.eval_count*torch.softmax(logw,dim=0).square().sum())
        result = {'test_nll':float(-logq_test.mean()),'normalized_ess':float(ess),
                  'log_z_lower_bound':float(logw.mean()),'log_z_logmeanexp':float(torch.logsumexp(logw,0)-np.log(args.eval_count)),
                  'best_validation_nll':best,'best_cfm_update':saved['step'],
                  'test_target_nll':float(-target.log_prob(test.cpu()).mean()),
                  'cfm_samples_mean_log_p':float(logp_cfm.mean()),'wandb_url':run.url,
                  'gt_control':args.gt_control,'evaluation_only':True,'elapsed_seconds':time.monotonic()-started}
        np.save(args.output/'cfm_evaluation_samples.npy',cfm_samples.cpu().numpy())
        write_json(args.output/'summary.json',result)
        run.summary.update(result)
        artifact = wandb.Artifact('cfm-evaluation-'+run.id,type='gmm40-evaluation')
        for file in args.output.iterdir():
            if file.is_file(): artifact.add_file(str(file))
        run.log_artifact(artifact)
        run.finish()
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        write_json(args.output/'failed.json',{'error':repr(exc),'wandb_url':run.url})
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
