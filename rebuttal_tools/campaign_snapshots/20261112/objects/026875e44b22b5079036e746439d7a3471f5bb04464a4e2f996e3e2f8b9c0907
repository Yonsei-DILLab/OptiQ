"""Check the positive replay gate and meaningful corruptions before training."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .verified_demos import resolve, sha, verify
from .run import atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();registry=json.loads(a.registry.read_text());checks=[]
    for task,spec in registry.items():
        dataset,proof=resolve(task,spec)
        checks.append(dict(task=task,test='positive_saved_replay',passed=True,trajectories=proof['trajectories']))
        def rejected(name,action):
            try:action()
            except (AssertionError,KeyError):checks.append(dict(task=task,test=name,passed=True))
            else:raise AssertionError('Gate accepted '+name)
        rejected('missing_proof',lambda:resolve(task,{}))
        rejected('wrong_manifest_hash',lambda:resolve(task,dict(spec,demonstration_manifest_sha256='0'*64)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=Path(spec['demonstration_manifest'])
            m=json.loads(original.read_text())
            shutil.copytree(original.parent,root,dirs_exist_ok=True)
            path=root/'manifest.json'
            bad=copy.deepcopy(m);bad['episodes'][1]['trajectory_sha256']=bad['episodes'][0]['trajectory_sha256'];atomic(path,bad)
            rejected('duplicate_episode',lambda:verify(path))
            bad=copy.deepcopy(m);bad['environment']['frame_skip']+=1;atomic(path,bad)
            rejected('different_physics',lambda:verify(path))
            # Forge internally consistent success labels and dataset hash. The
            # actual environment's info[success] must still reject the forgery.
            with np.load(dataset,allow_pickle=False) as z:rows={k:z[k].copy() for k in z.files}
            index=int(np.flatnonzero(~rows['success'])[0]);episode=int(rows['trajectory_ids'][index])
            rows['success'][index]=True;np.savez_compressed(root/m['dataset']['file'],**rows)
            bad=copy.deepcopy(m);bad['dataset']['sha256']=sha(root/m['dataset']['file']);bad['episodes'][episode]['success_steps']+=1
            if index==int(rows['offsets'][episode+1])-1:bad['episodes'][episode]['final_step_success']=True
            atomic(path,bad)
            rejected('forged_success_label_even_with_updated_hash',lambda:verify(path))
    atomic(a.output,dict(passed=True,checks=checks));print(json.dumps(dict(passed=True,checks=len(checks))))


if __name__=='__main__':main()
