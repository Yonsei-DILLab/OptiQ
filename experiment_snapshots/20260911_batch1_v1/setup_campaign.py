"""Run on login4; isolate code and freeze the requested two-version protocol."""
import argparse
import ast
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
parser.add_argument('--out',required=True);parser.add_argument('--payload',required=True)
args=parser.parse_args();base=Path(args.base);out=Path(args.out);payload=Path(args.payload)
out.mkdir(parents=True,exist_ok=False)
for name in ('runs','logs','references'): (out/name).mkdir()
shutil.copytree(base/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
code=out/'code'
# Keep the base critic/training implementation. Transplant only the audited
# actor method, whose additions are unbounded KDE and external latent support.
algorithm=code/'optiq_dime/algorithm.py'
old=algorithm.read_text();new=(payload/'gmm_source/optiq_dime/algorithm.py').read_text()
def actor_range(text):
    node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='update_actor')
    return min([node.lineno]+[d.lineno for d in node.decorator_list])-1,node.end_lineno
start,end=actor_range(old);new_start,new_end=actor_range(new)
lines=old.splitlines(keepends=True)
lines[start:end]=new.splitlines(keepends=True)[new_start:new_end]
text=''.join(lines).replace('from .transport import (\n','from .transport import (\n    GaussianKDE,\n',1)
algorithm.write_text(text)
shutil.copy2(payload/'gmm_source/optiq_dime/transport.py',code/'optiq_dime/transport.py')
shutil.copytree(payload/'gmm_source/benchmarks/gmm40',code/'benchmarks/gmm40')
(code/'benchmarks/__init__.py').touch()
shutil.copytree(payload/'analysis_batch1',code/'analysis_batch1')
for name in ('submit_campaign.py','summarize.py'):shutil.copy2(payload/name,out/name)
import sys
sys.path.insert(0,str(code))
# Loading task configuration must not initialize JAX/CUDA on a login node.
# Read the pure config factory through AST, without importing runtime modules.
tree=ast.parse((code/'analysis_batch1/core.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='config_for')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'config_for','exec'),namespace)
config_for=namespace['config_for']
cases=['modes2d_8','gmm40','separable_8','modes2d_4','separable_4','unimodal',
       'asymmetric_1.3_0.08','constant','symmetric_1.3_0.12',
       'asymmetric_0.4_0.08','asymmetric_0.4_0.16','asymmetric_0.8_0.08',
       'asymmetric_0.8_0.16','asymmetric_1.3_0.16']
tasks=[]
for seed in range(5):
    for case in cases:
        for version in ('ver1','ver2'):
            cfg=config_for(case,version,seed)
            tasks.append(dict(name=case+'_'+version+'_seed'+str(seed),config=cfg))
assert len(tasks)==140
def write(name,value): (out/name).write_text(json.dumps(value,indent=2)+'\n')
write('tasks.json',tasks)
write('protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
    seeds=list(range(5)),tasks=len(tasks),cases=cases,initializations=['random'],
    common=dict(batch_size=1,temperature=1.,proposal_std=1.,transport_target_mode='argmax',density_beta=1.),
    ver1=dict(actor_samples=16,random_per_center=4,anchor_per_center=1,ot=[16,80]),
    ver2=dict(actor_samples=2048,random_per_center=1,anchor_per_center=0,ot=[2048,2048]),
    user_correction='Frozen-Q retains truncated KDE, perturbation limit .5 and action bounds [-1,1]. GMM alone unbounded.',
    frozen=dict(updates=20000,hidden_dims=[256]*3,sinkhorn_epsilon=.05,sinkhorn_iterations=30,
                learning_rate=.0003,latent_sampling='iid',q_unchanged=True,reference_temperature=1.),
    gmm=dict(updates=85000,hidden_dims=[512]*5,sinkhorn_epsilon_start=.01,
        sinkhorn_epsilon_final=.0001,sinkhorn_anneal_updates=15000,sinkhorn_iterations=300,
        learning_rate=.0003,learning_rate_after_50k=.0001,latent_sampling='grid',
        proposal_std_constant=1.,actor_sample_count_constant_within_each_version=True,coordinate_scale=1.),
    independent_evaluation=True,initial_actor_and_optimizer_paired=True,reference_sampling_seed=20260911,
    primary_backup_k=50,final_eval_samples=100000,intermediate_eval_samples=20000,
    equal_updates_within_case=True,equal_compute_budget=False,max_concurrent_gpus=2,
    prior_frozen_base_commit='7e2da67d2f0988f6f211b635f7311d32a0c8e8c6',
    gmm_source_commit='1d9394ad7e35d18856115f47983d19f81e4f136e',
    production_checkout_changed=False,previous_results_changed=False))
files=[p for folder in ('analysis_batch1','analysis_boltzmann','optiq_dime','benchmarks/gmm40','common','models','diffusion')
       for p in (code/folder).rglob('*.py')]
write('source_hashes.json',{str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
print(json.dumps(dict(campaign=str(out),tasks=len(tasks)),indent=2))
