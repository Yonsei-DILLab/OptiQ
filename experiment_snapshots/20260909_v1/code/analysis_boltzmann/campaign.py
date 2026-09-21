"""Immutable Slurm campaign, two concurrent GPUs, validated 5 -> 20 seeds."""
import argparse,csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from .problems import CASES

PYTHON='/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python'

def tasks_for(campaign,seeds):
    tasks=[]
    for seed in seeds:
        # Start with informative frozen cases, then the main MoveCar comparison.
        for case in CASES:
            for init in ('default','coverage'):
                out=campaign/'runs'/f'frozen_{case}_{init}_seed{seed}'
                tasks.append([['analysis_boltzmann.frozen','--case',case,'--seed',str(seed),'--initialization',init,'--out',str(out)]])
        for method in ('optiq','ddpg','sd2','td3','sd3'):
            out=campaign/'runs'/f'movecar_{method}_seed{seed}'; landscape=campaign/'runs'/f'landscape_{method}_seed{seed}'
            commands=[['analysis_boltzmann.movecar','--method',method,'--seed',str(seed),'--out',str(out)],['analysis_boltzmann.landscape','--method',method,'--seed',str(seed),'--start',str(out/'checkpoint_5001.msgpack'),'--end',str(out/'checkpoint_1000000.msgpack'),'--out',str(landscape)]]
            if method=='optiq': commands.append(['analysis_boltzmann.learned_backup','--seed',str(seed),'--checkpoint',str(out/'checkpoint_1000000.msgpack'),'--out',str(campaign/'runs'/f'learned_seed{seed}')])
            tasks.append(commands)
        for backup in ('max','boltzmann','actor','local'):
            tasks.append([['analysis_boltzmann.control','--backup',backup,'--seed',str(seed),'--grid','2056','--out',str(campaign/'runs'/f'control_{backup}_seed{seed}')]])
    return tasks

def cross_tasks(campaign,seeds):
    tasks=[]
    for seed in seeds:
        def checkpoint(method): return str(campaign/'runs'/f'movecar_{method}_seed{seed}'/'checkpoint_1000000.msgpack')
        commands=[]
        for first,last in [('optiq','optiq'),('ddpg','sd2'),('td3','sd3')]:
            start=str(campaign/'runs'/f'movecar_optiq_seed{seed}'/'checkpoint_5001.msgpack') if first=='optiq' else checkpoint(first)
            methods=['optiq','ddpg','sd2','td3','sd3'] if first=='optiq' else [first,last]
            commands.append(['analysis_boltzmann.landscape','--method',first,'--end-method',last,'--seed',str(seed),'--start',start,'--end',checkpoint(last),'--critics',*[m+':'+checkpoint(m) for m in methods],'--out',str(campaign/'runs'/f'cross_{first}_{last}_seed{seed}')])
        tasks.append(commands)
    return tasks

def submit_stage(campaign,stage,seeds):
    protocol=json.loads((campaign/'protocol.json').read_text())
    if stage in ('extension','crossextension') and not protocol.get('expansion_enabled',True):
        raise RuntimeError('Seed expansion disabled by the five-seed analysis scope')
    tasks=cross_tasks(campaign,seeds) if stage.startswith('cross') else tasks_for(campaign,seeds); tf=campaign/f'{stage}_tasks.json'; tf.write_text(json.dumps(tasks,indent=2))
    code=campaign/'code'
    cmd=['sbatch','--parsable','--partition=base_suma_rtx3090,dell_rtx3090','--exclude=node02,node04,node05,node14,node24,cs-gpu-01,node35','--array',f'0-{len(tasks)-1}%2','--time=06:00:00','--job-name',f'optiq-{stage}','--output',str(campaign/'logs/%A_%a.log'),'--export',f'ALL,OPTIQ_ANALYSIS_ROOT={code}',str(code/'analysis_boltzmann/job.sh'),'analysis_boltzmann.campaign','--worker','--campaign',str(campaign),'--stage',stage]
    jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    (campaign/f'{stage}_jobs.json').write_text(json.dumps({'array':jid,'gate':None,'command':cmd,'tasks':len(tasks)},indent=2))
    # No success-dependent seed selection. Gate checks implementation health only.
    script=campaign/f'{stage}_gate.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexport JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg\nunset LD_LIBRARY_PATH\ncd '+str(code)+'\nexec '+PYTHON+' -m analysis_boltzmann.campaign --gate --campaign '+str(campaign)+' --stage '+stage+'\n')
    gate=subprocess.check_output(['sbatch','--parsable','--partition=dell_cpu','--qos=cpu_qos','--cpus-per-task=2','--mem=8G','--time=02:00:00','--dependency=afterok:'+jid,'--job-name','optiq-gate','--output',str(campaign/'logs/gate-%j.log'),str(script)],text=True).strip()
    (campaign/f'{stage}_jobs.json').write_text(json.dumps({'array':jid,'gate':gate,'command':cmd,'tasks':len(tasks)},indent=2))
    print(json.dumps({'stage':stage,'array':jid,'gate':gate,'tasks':len(tasks)}),flush=True)

def validate(campaign,stage):
    tasks=json.loads((campaign/f'{stage}_tasks.json').read_text()); count=0
    for commands in tasks:
        for command in commands:
            out=Path(command[command.index('--out')+1]); assert (out/'COMPLETE').exists(),out
            for path in out.glob('*.csv'):
                for row in csv.DictReader(path.open()):
                    for field in ('mean','bias','rmse','truth'):
                        if field in row: assert np.isfinite(float(row[field])),(path,field)
            for path in out.glob('landscape*.npz'):
                with np.load(path) as arrays:
                    assert all(np.isfinite(arrays[k]).all() for k in arrays.files),path
            if (out/'grid_convergence.npz').exists():
                with np.load(out/'grid_convergence.npz') as grid_check:
                    err=float(grid_check['max_error'])
                    assert bool(grid_check['reference_converged']),('Reference did not converge',out)
                precision=json.loads((out/'reference_precision.json').read_text())
                assert precision['states']==256 and precision['consecutive_refinements']==2
                assert precision['history'][-1]['max_delta']<1e-4 and precision['history'][-2]['max_delta']<1e-4
                assert err<1e-3,('Control grid needs refinement; expansion paused',out,err)
            if (out/'learning.json').exists():
                rows=json.loads((out/'learning.json').read_text()); assert rows[-1]['step']==1000000
                for row in rows: assert all(np.isfinite(v) for v in row.values()),out
            count+=1
    return count


def main(args):
    campaign=Path(args.campaign).resolve()
    if args.worker:
        tasks=json.loads((campaign/f'{args.stage}_tasks.json').read_text()); index=int(os.environ['SLURM_ARRAY_TASK_ID'])
        for command in tasks[index]:
            out=Path(command[command.index('--out')+1])
            if (out/'COMPLETE').exists(): continue
            subprocess.run([sys.executable,'-m',*command],check=True)
        return
    if args.gate:
        count=validate(campaign,args.stage)
        from .plots import run
        run(campaign/'runs',campaign/f'figures_{args.stage}')
        (campaign/f'{args.stage}_passed.json').write_text(json.dumps({'validated_runs':count,'time':time.time()}))
        if args.stage=='pilot': submit_stage(campaign,'crosspilot',range(5))
        elif args.stage=='crosspilot': submit_stage(campaign,'extension',range(5,20))
        elif args.stage=='extension': submit_stage(campaign,'crossextension',range(5,20))
        else: (campaign/'COMPLETE').write_text('All 20 seeds complete; figures regenerated.\n')
        return
    root=Path(__file__).resolve().parents[1]
    campaign.mkdir(parents=True,exist_ok=False); (campaign/'logs').mkdir(); (campaign/'runs').mkdir(); code=campaign/'code'; code.mkdir()
    for folder in ('analysis_boltzmann','optiq_dime','common','diffusion','models','configs'):
        shutil.copytree(root/folder,code/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for file in ('requirements-mujoco.in','requirements-mujoco.lock'): shutil.copy2(root/file,code/file)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    (campaign/'protocol.json').write_text(json.dumps({'base_commit':commit,'seeds':list(range(20)),'pilot_seeds':list(range(5)),'temperature':.25,'max_concurrent_gpu_jobs':2,'frozen_updates':20000,'movecar_steps':1000000,'default_preserved':True,'control_is_modified_algorithm':True,'python':PYTHON,'source_root':str(code)},indent=2))
    submit_stage(campaign,'pilot',range(5))

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--campaign',required=True); ap.add_argument('--worker',action='store_true'); ap.add_argument('--gate',action='store_true'); ap.add_argument('--stage',default='pilot'); main(ap.parse_args())
