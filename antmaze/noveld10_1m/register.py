"""Register one shard without interrupting existing GPU owners."""
import argparse
import json
from pathlib import Path
import subprocess
from .campaign import NAME,PYTHON,source_sha


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--shard',type=int,choices=(0,1),required=True)
    a=p.parse_args();source,sha=source_sha();ops=Path('/home/heechan/OptiQ-ops')
    assert source==ops/'sources'/sha,'Use immutable committed worktree'
    dependencies={}
    for relative,required in (('gmm40-baseline/meow','cleanrl/cleanrl/meow_continuous_action.py'),
                              ('gmm40-baseline/MFPO','jaxrl5/agents/mean_flow_learner.py')):
        dependency=source/relative
        assert (dependency/required).is_file(),f'Initialize pinned Git submodule before registration: {relative}'
        expected=subprocess.check_output(['git','ls-tree','HEAD',relative],cwd=source,text=True).split()[2]
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=dependency,text=True).strip()
        assert actual==expected,f'Submodule revision mismatch: {relative}'
        dependencies[relative]=actual
    root=Path('/home/heechan/optiq-experiments')/NAME
    conf=ops/'supervisor/jobs'/(NAME+'.conf')
    assert not root.exists() and not conf.exists(),'Refuse duplicate registration'
    root.mkdir();(root/'logs').mkdir()
    cmd=[PYTHON,'-u','-m','antmaze.noveld10_1m.campaign','--root',str(root),'--shard',str(a.shard)]
    conf.write_text(f'''[program:{NAME}]
command={' '.join(cmd)}
directory={source}
environment=OPTIQ_SOURCE_DIR="{source}",PYTHONPATH="{source}",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",XLA_PYTHON_CLIENT_PREALLOCATE="false"
autostart=false
autorestart=false
startsecs=5
startretries=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={root}/controller.log
stdout_logfile_maxbytes=0
''')
    (root/'registration.json').write_text(json.dumps(dict(source_commit=sha,source=str(source),
        shard=a.shard,service=NAME,command=cmd,gpu_slots=[0,1,2,3],respect_existing_gpu_locks=True,dependencies=dependencies,
        autostart=False,autorestart=False),indent=2)+'\n')
    ctl=['/usr/local/bin/supervisorctl','-c',str(ops/'supervisor/supervisord.conf')]
    for tail in (['reread'],['update',NAME],['start',NAME]):subprocess.run(ctl+tail,check=True)
    print(json.dumps(dict(registered=True,source_commit=sha,shard=a.shard,root=str(root),service=NAME)))


if __name__=='__main__':main()
