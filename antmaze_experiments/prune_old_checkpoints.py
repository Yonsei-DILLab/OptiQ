"""Remove only intermediate states of completed legacy v3 MuJoCo runs."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path('/workspace/optiq-v3-temperature-20260912/outputs')

def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
    files=[]
    for final in ROOT.rglob('actor_state_1000000.msgpack'):
        folder=final.parent
        if not (folder/'critic_state_1000000.msgpack').is_file():continue
        for path in folder.iterdir():
            match=re.fullmatch(r'(actor|critic)_state_(\d+)\.msgpack',path.name)
            if match and int(match[2]) not in (5001,500000,1000000) and path.is_file() and not path.is_symlink():
                files.append(dict(path=str(path),bytes=path.stat().st_size))
    print(json.dumps(dict(files=len(files),bytes=sum(x['bytes'] for x in files),sample=files[:4])))
    if args.apply:
        manifest=Path('/workspace/pruned-v3-intermediate-20260924.json')
        with manifest.open('x') as f:json.dump(files,f,indent=2)
        for entry in files:
            path=Path(entry['path'])
            assert path.is_relative_to(ROOT) and path.stat().st_size==entry['bytes']
            path.unlink()
        print('Deleted intermediate checkpoints; retained initial, 500K and final 1M states.')

if __name__=='__main__':main()
