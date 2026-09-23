"""Register only the five explicitly specified historical inference jobs."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = 'antmaze-dense-random-reevaluation-20260924'
OPS = Path('/home/heechan/OptiQ-ops')
SUP = ['/usr/local/bin/supervisorctl', '-c', str(OPS/'supervisor/supervisord.conf')]
LEGACY = '19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5'
OFF = '26336810f7ea4ca61210ea70c6aeeae9f7acaed0'


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--host', choices=['180', '199'], required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--only', choices=['legacy-v1','official-v1','legacy-v2','legacy-v3','legacy-v4'])
    p.add_argument('--attempt', choices=['r2'])
    a = p.parse_args()
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    campaign = Path('/home/heechan/optiq-experiments')/CAMPAIGN
    campaign.mkdir(exist_ok=True)
    (campaign/'logs').mkdir(exist_ok=True)
    jobs = [('legacy','v1',1000), ('official','v1',1000), ('legacy','v3',500)] if a.host=='180' else [('legacy','v2',500), ('legacy','v4',500)]
    records = []
    for index,(family,task,episodes) in enumerate(jobs):
        if a.only and a.only != f'{family}-{task}':
            continue
        name = f'{family}-{task}' + ('-smoke' if a.smoke else '') + (f'-{a.attempt}' if a.attempt else '')
        if not a.smoke:
            proofs = list(campaign.glob(f'{family}-{task}-smoke*/verification.json'))
            assert any(json.loads(p.read_text())['passed'] for p in proofs)
        training = LEGACY if family=='legacy' else OFF
        train_root = OPS/'sources'/training
        original = ('antmaze-dense-noveld-1m-s0-20260922' if family=='legacy' else
                    'antmaze-dense-noveld-off-main-s0-20260923')
        run = campaign.parent/original/'runs'/f'{task}-optiq-s0'
        assert (run/'config.json').exists() and train_root.exists()
        out = campaign/name
        assert not out.exists(), out
        python = '/home/heechan/.venv-optiq-antmaze/bin/python' if family=='legacy' else '/home/heechan/.venv-ddiffpg-native/bin/python'
        args = ['/usr/bin/nice','-n','10',python,'-u',str(ROOT/'antmaze_experiments/reevaluate_dense_random.py'),
                '--family',family,'--task',task,'--run',str(run),'--training-source',str(train_root),
                '--output',str(out),'--evaluation-source',source,'--episodes',str(4 if a.smoke else episodes),
                '--batch',str(4 if a.smoke else 50),'--cpu-offset',str(40+4*index)]
        env = dict(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='ignore',MPLBACKEND='Agg',WANDB_MODE='disabled',
                   USE_FLAX='0',USE_TORCH='1',D4RL_SUPPRESS_IMPORT_ERROR='1',MUJOCO_PY_FORCE_CPU='1',
                   LD_LIBRARY_PATH='/home/heechan/.mujoco/mujoco210/bin',OPTIQ_SOURCE_DIR=str(train_root))
        service = f'{CAMPAIGN}-{name}'
        conf = OPS/'supervisor/jobs'/f'{service}.conf'
        assert not conf.exists(), conf
        text = f'[program:{service}]\ncommand={shlex.join(args)}\ndirectory={train_root}\n'
        text += 'environment='+','.join(f'{k}="{v}"' for k,v in env.items())+'\n'
        text += ('autostart=false\nautorestart=false\nstartsecs=1\nstartretries=0\n'
                 'stopasgroup=true\nkillasgroup=true\nredirect_stderr=true\nstdout_logfile_maxbytes=0\n')
        text += f'stdout_logfile={campaign}/logs/{name}.log\n'
        conf.write_text(text)
        records.append(dict(name=name,service=service,command=args,environment=env,
                            output=str(out),training_source=training,evaluation_source=source))
    manifest = campaign/f'registration-{a.host}{"-smoke" if a.smoke else ""}{"-"+a.attempt if a.attempt else ""}.json'
    assert not manifest.exists()
    manifest.write_text(json.dumps(records,indent=2)+'\n')
    subprocess.run(SUP+['reread'],check=True)
    for r in records:
        subprocess.run(SUP+['update',r['service']],check=True)
        subprocess.run(SUP+['start',r['service']],check=True)
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    main()
