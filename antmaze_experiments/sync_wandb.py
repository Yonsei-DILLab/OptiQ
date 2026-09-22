"""Upload completed offline runs after connectivity recovers, without GPU locks."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from .run import write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text())
    synced={};attempts={}
    proof_path=root/'wandb-sync-status.json'
    if proof_path.exists():synced=json.loads(proof_path.read_text()).get('synced',{})
    while True:
        waiting=[]
        for entry in manifest['jobs']:
            identifier=entry['id'];run=root/'runs'/identifier
            meta=run/'wandb.json'
            if not meta.exists() or not (run/'result.json').exists():continue
            info=json.loads(meta.read_text())
            if info.get('mode')!='offline' or identifier in synced:continue
            waiting.append(identifier)
            if time.time()-attempts.get(identifier,0)<300:continue
            attempts[identifier]=time.time()
            try:
                # Bounded subprocess protects this sidecar from stalled DNS.
                subprocess.run([sys.executable,'-c',
                    'import socket;s=socket.create_connection(("api.wandb.ai",443),5);s.close()'],
                    check=True,timeout=12,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                candidates=list((run/'wandb').glob('offline-run-*'))
                assert len(candidates)==1
                with (root/'wandb-sync.log').open('a') as log:
                    subprocess.run([sys.executable,'-m','wandb','sync','--entity','OptiQ',
                        '--project','gmm-trg',str(candidates[0])],check=True,timeout=180,
                        stdout=log,stderr=subprocess.STDOUT)
                synced[identifier]=dict(id=info['id'],time=time.time(),
                    url=f"https://wandb.ai/OptiQ/gmm-trg/runs/{info['id']}")
                write(run/'wandb-synced.json',synced[identifier])
            except (OSError,subprocess.SubprocessError,AssertionError) as error:
                with (root/'wandb-sync.log').open('a') as log:
                    log.write(f'{time.time()} {identifier}: {type(error).__name__}: {error}\n')
        status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        all_done=len(status.get('completed',[]))==len(manifest['jobs'])
        remaining=[i for i in waiting if i not in synced]
        write(proof_path,dict(time=time.time(),synced=synced,waiting=remaining,
            training_complete=all_done,completed=all_done and not remaining))
        if all_done and not remaining:return
        time.sleep(60)


if __name__=='__main__':main()
