"""SHA-check separate mu-only evaluations against their immutable training inputs."""
import argparse
import json
from pathlib import Path
from .collect_deadline import digest, sync
from .run_nway_job import atomic_json

REMOTE = 'vast-heechan-46:/home/heechan/optiq-experiments/maze-mu-posthoc-20260926/'


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args()
    destination=a.root/'posthoc_mu';destination.mkdir(exist_ok=True)
    sync(REMOTE,destination)
    status=json.loads((destination/'status.json').read_text())
    verified=[]
    for name in status['completed']:
        folder=destination/name
        proof=json.loads((folder/'proof.json').read_text())
        if proof['mode']!='mu_only' or proof['learner_updates']!=0:
            raise ValueError('unexpected evaluation/training mode')
        original=a.root/'runs'/name;step=proof['step']
        for relative,key in [('config.json','config_sha256'),
                (f'checkpoints/policy_{step:09d}.msgpack','checkpoint_sha256'),
                (f'evaluations/{step:09d}_mu_only.npz','preserved_mu_sha256')]:
            if digest(original/relative)!=proof[key]:
                raise ValueError('post-hoc policy differs from archived policy')
        for relative,sha in proof['sha256'].items():
            if digest(folder/relative)!=sha:raise ValueError('post-hoc SHA mismatch')
        verified.append(name)
    atomic_json(destination/'archive-verification.json',dict(verified=verified,expected=15,
        complete=status['status']=='complete' and len(verified)==15))
    print(json.dumps(dict(status=status['status'],verified=len(verified),expected=15)))


if __name__=='__main__':main()
