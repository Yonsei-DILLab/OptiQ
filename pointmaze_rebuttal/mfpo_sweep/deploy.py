"""Create a frozen PointMaze sweep snapshot and its dedicated Supervisor entry."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from ..guards import ROOT,assert_work_branch,assert_output_boundary
from .campaign import make_plan,source_hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--maze',choices=('medium','hard'),required=True)
    parser.add_argument('--campaign',required=True)
    args=parser.parse_args()
    assert_work_branch()
    folder=ROOT/'outputs/pointmaze-rebuttal'/args.campaign/args.maze
    assert_output_boundary(folder);folder.mkdir(parents=True,exist_ok=False)
    source=folder/'source';source.mkdir()
    shutil.copytree(ROOT/'pointmaze_rebuttal',source/'pointmaze_rebuttal',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    python=ROOT/'.venv/bin/python'
    plan=make_plan(args.maze,folder,source,python)
    blocked=set()
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True)
    for value in pids.splitlines():
        if not value.strip().isdigit():continue
        try:
            affinity=set(os.sched_getaffinity(int(value)))
            if 0<len(affinity)<=16:blocked.update(affinity)
        except ProcessLookupError:pass
    allowed=set(os.sched_getaffinity(0));seen=set();groups=[]
    for cpu in sorted(allowed):
        siblings=set()
        for part in Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(','):
            if '-' in part:
                lo,hi=map(int,part.split('-'));siblings.update(range(lo,hi+1))
            else:siblings.add(int(part))
        group=tuple(sorted(siblings&allowed))
        if group not in seen and not (set(group)&blocked):seen.add(group);groups.append(group)
    lanes=[list(groups[i]+groups[i+1]) for i in range(0,len(groups)-1,2)]
    if len(lanes)<4:raise RuntimeError('Not enough free physical cores for four GPU lanes')
    plan.update(cpu_lanes=lanes,excluded_existing_gpu_cpus=sorted(blocked),
        source_hashes=source_hashes(source),
        root_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    (folder/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    program='pointmaze-'+args.campaign+'-'+args.maze
    conf=f'''[program:{program}]
command={python} -u -m pointmaze_rebuttal.mfpo_sweep.campaign --plan {folder}/plan.json
directory={source}
user=heechan
autostart=false
autorestart=unexpected
exitcodes=0
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={folder}/controller.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
environment=PYTHONPATH="{source}",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1"
'''
    subprocess.run(['sudo','-n','tee','/etc/supervisor/conf.d/'+program+'.conf'],input=conf,
        text=True,stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sudo','-n','supervisorctl','reread'],stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['sudo','-n','supervisorctl','add',program],stdout=subprocess.DEVNULL,check=True)
    print(json.dumps(dict(maze=args.maze,folder=str(folder),program=program,
        sweep_runs=len(plan['jobs']),final_runs_after_selection=5,dedicated_cpu_lanes=len(lanes))),flush=True)


if __name__=='__main__':main()
