"""Register only fresh, committed dense/NovelD-off probes or approved main jobs."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .settings import total_budget
from .dependencies import prepare_dependencies


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--stage',choices=['probe','main'],required=True)
    p.add_argument('--shard',type=int,choices=[0,1],required=True)
    p.add_argument('--probe-review',type=Path)
    a=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source==Path('/home/heechan/OptiQ-ops/sources')/sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    name=f'antmaze-dense-noveld-off-{a.stage}-s0-20260923'
    root=Path('/home/heechan/optiq-experiments')/name
    assert not root.exists(),root
    if a.stage=='probe':
        pairs=([('v1','dipo'),('v3','dipo'),('v1','optiq'),('v3','optiq')] if a.shard==0 else
               [('v1','sac'),('v3','sac'),('v1','mfpo'),('v3','mfpo')])
        review=None
    else:
        assert a.probe_review is not None
        review=json.loads(a.probe_review.read_text())
        assert review['source_commit']==sha and review['technical_validation_passed']
        assert review['completed_probe_count']==8 and review['trajectories_reviewed']
        # Native DIPO is slower: make both DIPO jobs eligible from the start.
        pairs=([('v1','dipo'),('v3','dipo'),('v1','optiq'),('v3','optiq'),
                ('v1','sac'),('v3','sac'),('v1','mfpo'),('v3','mfpo')] if a.shard==0 else
               [('v2','dipo'),('v4','dipo'),('v2','optiq'),('v4','optiq'),
                ('v2','sac'),('v4','sac'),('v2','mfpo'),('v4','mfpo')])
    jobs=[dict(id=f'{task}-{method}-s0',task=task,method=method,reward_profile='dense',
               noveld='off',eval_starts='upstream',steps=328192 if a.stage=='probe' else total_budget(task),
               final_eval_episodes=100) for task,method in pairs]
    manifest=dict(campaign=name,stage=a.stage,shard=a.shard,source=str(source),
                  source_commit=sha,wandb_mode='online',jobs=jobs,probe_review=review,
                  protocol='antmaze_experiments/DENSE_OFF_PROTOCOL.md',
                  num_envs=256,batch_size=4096,updates_per_vector_step=8,
                  reward_profile='dense',noveld_enabled=False,seed=0)
    manifest['evaluation_starts']='xy uniform[-2,2] per episode, all mazes; training reset settings unchanged'
    manifest['source_dependencies'] = prepare_dependencies(source, {method for _, method in pairs})
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf')
    assert not conf.exists(),conf
    conf.write_text(f'''[program:{name}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.controller --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{name}"
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
    ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    for cmd in (['reread'],['update',name],['start',name]):
        subprocess.run(ctl+cmd,check=True)
    (root/'registration.json').write_text(json.dumps(dict(time=time.time(),supervisor=name,
        manifest=manifest,config=str(conf)),indent=2)+'\n')
    print(json.dumps(dict(root=str(root),jobs=jobs,source_commit=sha)))


if __name__=='__main__':main()
