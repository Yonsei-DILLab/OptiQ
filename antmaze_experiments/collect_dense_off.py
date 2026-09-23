"""Collect immutable dense-off metadata and raw trajectories; verify final digests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def collect(host, root, name, complete):
    destination=root/host
    destination.mkdir(parents=True,exist_ok=True)
    remote=f'/home/heechan/optiq-experiments/{name}/'
    subprocess.run(['rsync','-az','--exclude=wandb/','--exclude=checkpoint*.pt',
                    '--exclude=learner/',f'{host}:{remote}',str(destination)+'/'],check=True)
    status=json.loads((destination/'status.json').read_text())
    summary=dict(host=host,completed=status['completed'],failed=status['failed'],
                 running=[j['id'] for j in status['running']],pending=status['pending'])
    if complete:
        assert not status['failed'] and not status['running'] and not status['pending'],summary
        hashes=subprocess.check_output(['ssh',host,f'sha256sum {remote}runs/*/checkpoint-final.pt'],text=True)
        digests={Path(line.split(maxsplit=1)[1]).parent.name:line.split()[0]
                 for line in hashes.splitlines() if line}
        manifest=json.loads((destination/'manifest.json').read_text())
        for job in manifest['jobs']:
            run=destination/'runs'/job['id']
            result=json.loads((run/'result.json').read_text())
            proof=json.loads((run/'checkpoint-verification.json').read_text())
            assert result['completed'] and result['source_commit']==manifest['source_commit']
            assert result['steps']==job['steps'] and result['rnd_updates']==0
            assert proof['readback_verified'] and proof['dense_replay_verified']
            assert digests[job['id']]==proof['sha256']==result['checkpoint']['sha256']
        (destination/'archive-verification.json').write_text(json.dumps(dict(
            verified=True,checkpoint_sha256=digests,source_commit=manifest['source_commit'],
            checkpoint_location='retained on remote server; SHA256 recomputed',
            local_data='config,proofs,training xy,successes,raw evaluation rollouts,logs'),indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['probe','main'],default='probe')
    p.add_argument('--complete',action='store_true')
    a=p.parse_args()
    name=f'antmaze-dense-noveld-off-{a.stage}-s0-20260923'
    a.root.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda h:collect(h,a.root,name,a.complete),
                             ['vast-heechan-180','vast-heechan-199']))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
