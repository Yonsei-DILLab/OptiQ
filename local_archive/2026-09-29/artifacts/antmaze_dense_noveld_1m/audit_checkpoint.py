"""Inspect an immutable production snapshot on CPU, without changing training."""
import argparse,json,sys,datetime
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from antmaze.multimodal.dense_noveld_report import state_digest,file_hash,write,GOALS,TRAINING_SHA
p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);a=p.parse_args();folder=a.snapshot
proof=json.loads((folder/'manifest.json').read_text())
assert proof['source_commit']==TRAINING_SHA
for name,spec in proof['files'].items():
 f=folder/name;assert f.stat().st_size==spec['bytes'] and file_hash(f)==spec['sha256']
s=torch.load(folder/'state.pt',map_location='cpu',weights_only=False)
assert state_digest(s)==proof['state_digest']
step=proof['step'];updates=proof['updates'];meta=s['extra'];assert not meta['smoke']
assert step-meta['warmup']==updates==s['model']['updates']==s['intrinsic']['updates']
assert s['coverage']['total_steps']==step
assert sum(e['length'] for e in s['coverage']['episodes'])+s['coverage']['episode_length']==step
assert int(s['coverage']['counts'].sum()+s['coverage']['outside'])==step
assert state_digest(s['intrinsic']['target'])==meta['initial_rnd']['target']
assert state_digest(s['intrinsic']['predictor'])!=meta['initial_rnd']['predictor']
with np.load(folder/'replay.npz',allow_pickle=False) as z:
 data={k:z[k] for k in z.files}
 assert state_digest(data)==proof['replay_digest']
 assert all(np.isfinite(x).all() for x in data.values())
 assert len(data['rewards'])==s['replay']['size']==min(step,1000000)
 assert s['replay']['position']==step%1000000
 goals=np.asarray(GOALS[meta['task']]);dist=np.linalg.norm(data['next_observations'][:,:2,None]-goals.T[None],axis=1).min(1)
 np.testing.assert_allclose(data['rewards'],-dist,atol=1e-5,rtol=2e-6)
 assert (dist[data['dones'].astype(bool)]<=.50001).all()
 assert (dist[~data['dones'].astype(bool)]>.49999).all()
 if meta['method']=='optiq':
  assert int(s['model']['policy']['actor_state']['step'])==updates
  assert int(s['model']['policy']['qf_state']['step'])==updates
  assert s['model']['regulator']['regulator_count']>0
report=dict(passed=True,method=meta['method'],task=meta['task'],step=step,updates=updates,
 replay_rows=len(data['rewards']),dense_reward_rows_verified=len(dist),replay_terminal_goals=int(data['dones'].sum()),
 learner_and_intrinsic_updates_match=True,sha256_verified=True,full_state_digest_verified=True,
 source_commit=TRAINING_SHA,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
write(folder/'READ_ONLY_AUDIT.json',report);print(json.dumps(report,indent=2))
