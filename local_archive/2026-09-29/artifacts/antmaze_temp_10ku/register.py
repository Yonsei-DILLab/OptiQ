"""Register the eight independent AntMaze v1 temperature jobs (four per host)."""
import json
import subprocess
import sys
import time
from pathlib import Path

shard = int(sys.argv[1])
assert shard in (0, 1, 2)
sha = 'a8250c4cde21c8748f99c33329b0540bf54b19a3'
name = ('antmaze-v1-optiq-temp-10ku-s0-20260923'
        if shard != 2 else 'antmaze-v1-optiq-temp-10ku-samehost180-s0-20260923')
source = Path('/home/heechan/OptiQ-ops/sources') / sha
root = Path('/home/heechan/optiq-experiments') / name
ctl = ['/usr/local/bin/supervisorctl', '-c', '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
temps = ([1.0, 0.25, 0.05, 0.01] if shard == 0 else [0.5, 0.1, 0.025, 0.005])
mode = 'offline' if shard == 1 else 'online'

assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == sha
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip()
assert not root.exists(), root
if shard != 2:
    assert not subprocess.run(['pgrep', '-af', '[a]ntmaze_experiments.run'],
        text=True, capture_output=True).stdout.strip()
else:
    old = root.parent / 'antmaze-v1-optiq-temp-10ku-s0-20260923'
    assert json.loads((old / 'manifest.json').read_text())['source_commit'] == sha

manifest = dict(
    campaign=name, shard=shard, source=str(source), source_commit=sha,
    upstream_commit='7edd06c4799abbab0f8fa534c21deb56253b018e',
    wandb_entity='OptiQ', wandb_project='gmm-trg', wandb_group=name,
    wandb_mode=mode, task='v1', method='optiq', seed=0,
    reward='unchanged original DDiffPG sparse reward', noveld_coefficient=0.01,
    num_envs=256, batch_size=4096, updates_per_vector_step=8,
    warmup_transitions=8192, learner_updates=10000,
    total_environment_transitions=328192, eval_interval=250000,
    final_eval_episodes_per_mode_per_reset=100,
    checkpoint='final only',
    jobs=[dict(id=f'v1-optiq-T{t:g}-s0', task='v1', method='optiq', temperature=t,
               steps=328192, final_eval_episodes=100) for t in temps],
)
root.mkdir()
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
conf = Path('/home/heechan/OptiQ-ops/supervisor/jobs') / (name + '.conf')
assert not conf.exists(), conf
conf.write_text(f'''[program:{name}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.controller --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="{mode}",OPTIQ_CAMPAIGN="{name}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/controller.log
stderr_logfile={root}/controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
for command in (['reread'], ['update', name], ['start', name]):
    subprocess.run(ctl + command, check=True)
(root / 'registration.json').write_text(json.dumps(dict(time=time.time(), manifest=manifest,
    supervisor=name, config=str(conf)), indent=2) + '\n')
print(json.dumps(dict(root=str(root), shard=shard, jobs=len(temps), temps=temps)))

if mode == 'offline':
    sync_name = name + '-wandb-sync'
    sync_conf = conf.parent / (sync_name + '.conf')
    assert not sync_conf.exists(), sync_conf
    sync_conf.write_text(f'''[program:{sync_name}]
command=/bin/bash -lc 'source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm; cd {source}; exec /home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.sync_wandb --root {root}'
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stdout_logfile={root}/wandb-sync-controller.log
stderr_logfile={root}/wandb-sync-controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    for command in (['reread'], ['update', sync_name], ['start', sync_name]):
        subprocess.run(ctl + command, check=True)
