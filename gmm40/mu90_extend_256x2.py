"""Continue two 100k checkpoints to500k; preserve optimizer,RNG,and source."""
import argparse,json,os,subprocess,time
from pathlib import Path
from .campaign import read,write,occupied_gpus
from .mu90_launch import CTL,PYTHON,unchanged_algorithm
SOURCE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/heechan/optiq-experiments/gmm40-mu90-256x2-500k-20260921')

def preflight():
    import jax,numpy as np,tempfile
    from .optiq_trg import OptiQTRG
    from .target import Target
    m=read(ROOT/'manifest.json');j=m['jobs'][0];c=read(Path(j['parent'])/'config.json');t=Target()
    def agent():return OptiQTRG(t,seed=c['seed'],n=c['n'],m=c['m'],batch=c['batch'],hidden_dims=(c['width'],)*c['depth'],temperature=c['temperature'],log_std_max=c['trg_log_std_max'],initial_log_std=c['trg_initial_log_std'],teacher_std_floor=c['trg_teacher_std_floor'],mean_output_init_scale=c['mean_output_init_scale'])
    a=agent();a.restore(Path(j['checkpoint']));assert a.updates==100000 and int(a.state.step)==100000
    a.advance(1);p=ROOT/'preflight_checkpoint.bin';a.save(p);b=agent();b.restore(p)
    ia=a.advance(1);ib=b.advance(1)
    for x,y in zip(jax.tree.leaves((a.state,a.key)),jax.tree.leaves((b.state,b.key))):assert np.array_equal(np.asarray(x),np.asarray(y))
    assert a.updates==b.updates==100002 and all(np.isfinite(v) for v in ia.values())
    write(ROOT/'preflight.json',dict(status='passed',runner_commit=m['runner_commit'],restored_optimizer_step=100000,checked_step=100002,roundtrip_parameters_optimizer_rng_identical=True,time=time.time()))

def register():
    unchanged_algorithm(SOURCE);assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()
    assert not (set((2,3))&occupied_gpus()),'Wait for both source jobs to release their GPUs'
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip();jobs=[]
    for gpu,cap in [(2,'4p5'),(3,'4p0')]:
        campaign=f'gmm40-mu90-repro-256x2-capm{cap}-20260921';parent=ROOT.parent/campaign/'results'/(campaign+'-init16_nm64-s0');c=read(parent/'config.json');audit=read(parent/'update_count_audit.json')
        assert audit['status']=='passed' and audit['actor_updates']==100000
        assert c['n']==c['m']==64 and c['width']==256 and c['depth']==2
        name=f'mu90_256x2_capm{cap}_s0_500k';checkpoint=parent/'checkpoints/step_0100000.bin'
        argv=['--method','optiq_trg','--name',name,'--steps','500000','--resume',str(checkpoint)]
        for k in ('seed','n','m','batch','width','depth','temperature','eval_samples','trg_log_std_max','trg_initial_log_std','trg_teacher_std_floor','mean_output_init_scale'):argv+=['--'+k.replace('_','-'),str(c[k])]
        jobs.append(dict(name=name,gpu=gpu,parent=str(parent),checkpoint=str(checkpoint),training_source_commit=c['source_git_commit'],args=argv,steps=500000))
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'manifest.json').exists()
    write(ROOT/'manifest.json',dict(runner_commit=sha,source=str(SOURCE),jobs=jobs,algorithm_unchanged=unchanged_algorithm(SOURCE),created=time.time()))
    for d in ('logs','wandb','results'):(ROOT/d).mkdir(exist_ok=True)
    # Preflight uses the exact saved target metadata, without rewriting the original.
    import shutil
    (ROOT/'results/target').mkdir(exist_ok=True);shutil.copyfile(Path(jobs[0]['parent']).parent/'target/definition.json',ROOT/'results/target/definition.json')
    env=dict(os.environ,OPTIQ_SOURCE_DIR=str(SOURCE),GMM40_REPO_ROOT=str(SOURCE),GMM40_RESULTS_ROOT=str(ROOT/'results'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    subprocess.run(['/home/heechan/OptiQ-ops/run-gpu.sh','2','--branch','v5-gmm40',PYTHON,'-B','-u','-m','gmm40.mu90_extend_256x2','preflight'],cwd=SOURCE,env=env,check=True)
    # Retain the original target assets/metadata while recording the new runner commit.
    for j in jobs:
        original_source=Path('/home/heechan/OptiQ-ops/sources')/j['training_source_commit']
        e=dict(OPTIQ_SOURCE_DIR=str(SOURCE),GMM40_REPO_ROOT=str(original_source),GMM40_RESULTS_ROOT=str(ROOT/'results'),GMM40_SOURCE_COMMIT=j['training_source_commit'],GMM40_RUNNER_COMMIT=sha,GMM40_CAMPAIGN=ROOT.name,GMM40_WANDB_DIR=str(ROOT/'wandb'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg')
        name=ROOT.name+'-'+str(j['gpu']);p=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf');assert not p.exists()
        cmd=['/home/heechan/OptiQ-ops/run-gpu.sh',str(j['gpu']),'--branch','v5-gmm40',PYTHON,'-B','-u','-m','gmm40.campaign_job',*j['args']]
        p.write_text('\n'.join([f'[program:{name}]','command='+' '.join(cmd),f'directory={SOURCE}','autostart=true','autorestart=false','startsecs=3','startretries=0','stopasgroup=true','killasgroup=true','stopwaitsecs=30','redirect_stderr=true',f'stdout_logfile={ROOT}/logs/{j["name"]}.log','stdout_logfile_maxbytes=10MB','environment='+','.join(k+'="'+v+'"' for k,v in e.items()),'']))
    write(ROOT/'registration.json',dict(runner_commit=sha,services=[ROOT.name+'-'+str(j['gpu']) for j in jobs],automatic_restarts=False,time=time.time()))
    subprocess.run(CTL+['reread'],check=True)
    for j in jobs:subprocess.run(CTL+['update',ROOT.name+'-'+str(j['gpu'])],check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','preflight']);a=p.parse_args()
    if a.action=='register':register()
    else:preflight()
if __name__=='__main__':main()
