"""Prepare, never start, the explicitly requested two-task CrossQ repetition."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]

LAUNCH = '''"""Manual launcher. Default is a preview; the prepared plan is execution-held."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
plan=json.loads((root/'plan.json').read_text())
if not args.execute:
    print(json.dumps(dict(status=plan['status'],execution_authorized=plan['execution_authorized'],jobs=[dict(id=j['id'],gpu=j['gpu'],command=j['command']) for j in plan['jobs']]),indent=2))
    raise SystemExit(0)
if not plan['execution_authorized']:
    raise SystemExit('PREPARED ONLY: user said do not run. A later user instruction is required before releasing this plan.')
assert plan['status']=='prepared' and all(j['status']=='prepared' for j in plan['jobs'])
manifest=json.loads((root/'source-manifest.json').read_text())
for name,expected in manifest['files'].items():
    assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==expected,name
verify=subprocess.run([plan['python'],str(root/'validate_preparation.py')],cwd=root,text=True,capture_output=True)
if verify.returncode:raise SystemExit(verify.stderr+verify.stdout)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if apps:raise SystemExit('GPU compute processes already exist; refresh allocation before launch.')
if (root/'launch-receipt.json').exists():raise SystemExit('Already launched; refusing duplicate seeds.')
logdir=root/'console';logdir.mkdir(exist_ok=True)
launched=[]
try:
    for job in plan['jobs']:
        env=dict(os.environ,**plan['environment'],CUDA_VISIBLE_DEVICES=str(job['gpu']))
        with (logdir/(job['id']+'.log')).open('xb') as output:
            child=subprocess.Popen(job['command'],cwd=root/'source',env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        launched.append(dict(id=job['id'],pid=child.pid,gpu=job['gpu'],time=time.time()))
        (root/'launch-receipt.json').write_text(json.dumps(launched,indent=2))
        job.update(status='launched',pid=child.pid)
        plan['status']='running'
        (root/'plan.json').write_text(json.dumps(plan,indent=2))
finally:
    print(json.dumps(dict(launched=launched),indent=2))
'''

VALIDATE = '''"""Compose configuration and inspect metadata only: no model, environment step or training."""
import ast,hashlib,importlib.metadata,json,os,subprocess,time
from pathlib import Path
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
root=Path(__file__).resolve().parent
plan=json.loads((root/'plan.json').read_text())
manifest=json.loads((root/'source-manifest.json').read_text())
for name,expected in manifest['files'].items():
    path=root/'source'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
tree=ast.parse((root/'source/train.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_config')
scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<validate_config>','exec'),scope)
versions={name:importlib.metadata.version(name) for name in plan['expected_packages']}
assert versions==plan['expected_packages'],dict(actual=versions,expected=plan['expected_packages'])
resolved={}
for job in plan['jobs']:
    with initialize_config_dir(config_dir=str(root/'source/configs'),version_base=None):
        cfg=compose(config_name='train',overrides=job['overrides'])
    scope['validate_config'](cfg)
    actual=OmegaConf.to_container(cfg,resolve=True)
    for key,expected in plan['expected_settings'].items():
        value=actual
        for part in key.split('.'):value=value[part]
        assert value==expected,(job['id'],key,value,expected)
    assert actual['temperature']==job['temperature']==actual['alg']['actor']['temperature']
    assert actual['seed']==job['seed']
    assert actual['task']==job['task']
    resolved[job['id']]=actual
    target=root/'configs_resolved'/(job['id']+'.json');target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(actual,indent=2))
result=dict(preparation_validated=True,training_started=False,smoke_training_started=False,
            model_initialized=False,environment_stepped=False,checked_jobs=len(resolved),
            packages=versions,time=time.time(),configs=resolved)
(root/'preparation-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(preparation_validated=True,checked_jobs=len(resolved),training_started=False,packages=versions)))
'''

def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch and 'anonymous' not in branch.lower()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name='crossq-standard-humanoid-hopper-'+stamp
    out=ROOT/'outputs/mujoco-rebuttal'/name
    source=out/'source';source.mkdir(parents=True,exist_ok=False)
    remote=Path('/workspace')/('heechan-'+name)
    paths=[ROOT/name for name in ['train.py','environment.py','exploration.py']]
    for folder in ['ibolt','models','configs']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml'] and '__pycache__' not in p.parts)
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT);dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);hashes[str(relative)]=hashlib.sha256(dest.read_bytes()).hexdigest()
        if path.suffix=='.py':ast.parse(dest.read_text())
    (out/'source-manifest.json').write_text(json.dumps(dict(branch=branch,git_head=revision,
        source_type='current working-tree snapshot; uncommitted core edits identified by per-file hash',files=hashes),indent=2))
    settings={
        'alg.critic.type':'crossq','alg.critic.hs':[1024,1024],'alg.critic.n_critics':2,
        'alg.critic.bn_momentum':.99,'alg.critic.renorm_warmup_steps':100000,
        'alg.actor.hidden_dims':[256,256],'alg.actor.batch_renorm':True,'alg.actor.policy_delay':3,
        'alg.actor.bn_momentum':.99,'alg.actor.renorm_warmup_steps':100000,
        'alg.optimizer.lr_actor':.001,'alg.optimizer.lr_critic':.001,
        'alg.optimizer.actor_b1':.5,'alg.optimizer.critic_b1':.5,
        'alg.optimizer.actor_b2':.999,'alg.optimizer.critic_b2':.999,
        'alg.utd':1,'alg.batch_size':256,'alg.buffer_size':1000000,'alg.learning_starts':5000,
        'alg.actor.learning_starts':5000,'alg.actor.num_policy_samples':64,'alg.actor.proposals_per_policy_sample':1,
        'alg.actor.log_std_min':-5.,'alg.actor.log_std_max':-1.,'alg.actor.initial_log_std':-1.,
        'alg.actor.custom_log_std_bounds':False,'alg.actor.use_min_candidate_q':False,
        'alg.actor.density_correction_beta':1.,'alg.gamma':.99,
        'dacer.enabled':True,'dacer.target_entropy_per_dim':-.9,'dacer.initial_alpha':.27,
        'dacer.alpha_lr':.03,'dacer.interval_updates':10000,'dacer.components':3,
        'dacer.samples':200,'dacer.noise_scale':.1,'dacer.entropy_seed':42,
        'total_steps':1000000,'eval_interval':5000,'num_eval_episodes':10,
        'checkpoint_interval':100000,'diagnostic_interval':5000,'wandb.activate':False,'require_gpu':True,
    }
    versions={'jax':'0.6.2','jaxlib':'0.6.2','flax':'0.10.6','optax':'0.2.5','numpy':'1.26.4',
              'gymnasium':'1.2.2','mujoco':'2.3.7','stable-baselines3':'2.7.0','hydra-core':'1.3.2','omegaconf':'2.3.1','wandb':'0.29.0'}
    plan=dict(name=name,status='prepared',execution_authorized=False,execution_hold='User: prepare only, do not run yet.',
              server='ibolt-server-1',remote_root=str(remote),python='/workspace/heechan-adroit/venv/bin/python',
              expected_packages=versions,expected_settings=settings,jobs=[],
              environment=dict(XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',WANDB_MODE='disabled',
                               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                               PYTHONPATH=str(remote/'source'),JAX_COMPILATION_CACHE_DIR=str(remote/'jax-cache')))
    for task,temperature in [('humanoid',.5),('hopper',.05)]:
        for seed in range(5):
            i=len(plan['jobs']);jobid=f'{task}-ibolt-crossqAC-meanQ-T{temperature:g}-s{seed}'
            overrides=[f'benchmark={task}','+critic=crossq','+actor=crossq',f'temperature={temperature}',f'seed={seed}']
            overrides += [key+'='+json.dumps(value,separators=(',',':')) for key,value in settings.items()]
            overrides += [f'run_name={jobid}',f'output_root={remote}/runs/{jobid}',
                          f'hydra.run.dir={remote}/hydra/{jobid}','hydra.output_subdir=null','hydra.job.chdir=false']
            plan['jobs'].append(dict(id=jobid,task=task,seed=seed,temperature=temperature,status='prepared',gpu=i%4,
                cpu_affinity=[],overrides=overrides,command=[]))
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    (out/'launch.py').write_text(LAUNCH)
    (out/'validate_preparation.py').write_text(VALIDATE)
    for text in [LAUNCH,VALIDATE]:ast.parse(text)
    registry=json.loads((ROOT/'outputs/crossq-closure-20261111/analysis/run_registry.json').read_text())
    exact={}
    for task,temp in [('humanoid',.5),('hopper',.05)]:
        exact[task]=[dict(path=r['path'],seed=r.get('seed'),norm=r.get('norm')) for r in registry
            if r['task']==task and r.get('temperature')==temp and r.get('bounds')==[-5.,-1.] and r.get('utd')==1
            and r.get('actor_brn') and not r.get('minq')]
    history=dict(raw_registry_matches=exact,
        humanoid_historical_curve=dict(seeds=[0,1,2,3,4],steps=1000000,temperature=.5,mu_only_last5_mean=9626.953830244802,
            evidence='results/tsweep/humanoid_vs_crossq.pdf; historical PDF digitization in closure archive',
            limitation='Historical curve and plotting protocol exist; complete per-seed original configs/raw logs are not in the accessible archive.'),
        hopper_note='Available T0.05 raw/cache runs use min-Q or logstd upper=-3.5. No full matching mean-Q five-seed cohort was identified.',
        running_now='Previous remote CrossQ runs on servers1,3,4 were stopped and deleted; server2 unreachable in last audit.')
    (out/'existing-experiment-audit.json').write_text(json.dumps(history,indent=2))
    readme=f'''# iBOLT+CrossQ 표준 설정 준비

상태: **준비만 완료할 대상. 학습·smoke training·자동 큐 등록 금지.**
최신 사용자 지시: “준비만 하고 일단 돌리진 마”. 실행 권한 플래그는 false다.

| 환경 | T | 시드 | 학습 예산 |
| --- | --- | --- | --- |
| Humanoid-v4 | 0.5 | 0–4 | 각각 1M transitions |
| Hopper-v4 | 0.05 | 0–4 | 각각 1M transitions |

총 10개 실행을 서버1의 GPU 0–3에 3/3/2/2개씩 나누는 계획이다. 학습은 단일 환경이며 UTD1·batch256이다.
원래 iBOLT 대조군이나 SBX SAC/CrossQ를 추가하는 계획은 아니다.

Critic 1024×1024×2, actor 256×256+BRN, momentum0.99, warmup100000, actor delay3,
actor·critic LR0.001, Adam β1=0.5, β2=0.999. 후보64개, mean-Q, density correction β=1,
logstd [-5,-1], 초기 -1. 후보 Q는 저장 EMA로 평가하고 critic은 joint BRN으로 학습한다.
Target network 업데이트는 CrossQ 경로에서 사용하지 않는다. Plain TD 및 DACER 기본값을 유지한다.
DACER: target/dim -0.9, 초기 α0.27, α LR0.03, interval10000 updates, noise scale0.1.

평가: 5k마다 10 episodes, stochastic z·μ-only를 주 지표로 사용한다. z=0과 z+σ도 별도로 저장한다.
checkpoint는 100k 간격 및 마지막에 저장하고, training CSV와 설정 원본을 보관한다.
랜덤 warmup은 5000 transitions. 예정 critic updates=995000, actor updates=331666이다.

BRN의 100K는 환경 스텝이 아닌 각 모듈의 학습 forward 카운터다.
UTD1·delay3에서는 critic 약105k, actor 약305k 환경 스텝부터 renorm 보정이 적용된다.
이전 구현의 동작을 유지했으며 actor warmup을 환경100k로 바꾸지 않았다.

## 기존 기록

Humanoid T0.5·actor BRN·logstd[-5,-1]의 과거 5시드 1M 비교 곡선은 존재한다.
최근5회 μ-only 평균은 약9627이며, vector PDF 복원값이다. 원본 config와 시드별 raw 로그 전체가 없어
표의 모든 세부 설정이 완전히 같은지는 현재 보관본만으로 확정할 수 없다.
Hopper T0.05는 min-Q 또는 logstd 상한-3.5인 기존 기록이 있으며, 동일 mean-Q 5시드 완료군은 확인하지 못했다.

## 준비 경로

- 원격: `{remote}`
- 로컬: `{out}`
- `plan.json`: 준비 상태, 10개 명령, 실행 보류 플래그
- `configs_resolved/`: Hydra 설정을 해석한 최종 값
- `source/`, `source-manifest.json`: 현재 비익명 작업 트리의 코드 복사본 및 SHA-256
- `launch.py`: 기본 호출은 미리보기만 한다. `--execute`도 보류 플래그가 false이면 거부한다.
- `validate_preparation.py`: 모델 생성 없이 설정·코드 구문·패키지 버전만 검증한다.

서버의 기존 Python 환경을 읽기 전용으로 재사용하며 패키지는 변경하지 않는다.
등록된 supervisor/cron/기존 큐는 건드리지 않는다. 실제 실행 전 자원 상태를 다시 확인해야 한다.
현재 준비는 GPU smoke test나 학습 속도 검증을 포함하지 않는다.
'''
    (out/'README.md').write_text(readme)
    archive=out/'prepared-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path!=archive:tar.add(path,arcname=str(path.relative_to(out)),recursive=False)
    (ROOT/'outputs/mujoco-rebuttal/latest-crossq-standard-preparation.txt').write_text(str(out))
    print(json.dumps(dict(local=str(out),remote=str(remote),archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),jobs=10,training_started=False)))

if __name__=='__main__':main()
