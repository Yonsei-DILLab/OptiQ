"""45 DACER-off comparisons; preserve and prioritize the previous campaign."""
import argparse,json,subprocess
from pathlib import Path
from . import vast_queue as queue
from .temperature_sweep import campaign_manifest

ROOT=Path('/workspace/antmaze-dacer-off-20260924')
OLD=Path('/workspace/antmaze-temperature-20260924')
CAMPAIGN='antmaze-dacer-off-dense-sparse-20260924'
HOSTS=['vast2','vast5','vast3','vast4']

def manifest(host):
 source=Path(__file__).resolve().parents[1]
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
 m=campaign_manifest(source,sha);jobs=[]
 # Interleave reward profiles so comparisons do not wait behind a whole sweep.
 for seed in range(3):
  for reward,t in [('dense',3.),('sparse',3.),('dense',5.),('sparse',5.),('sparse',1.)]:
   for task in ('v1','v3','v4'):
    j=dict(next(j for j in m['jobs'] if j['task']==task))
    j.update(id=f'{task}-{reward}-dacerOff-T{t:g}-s{seed}',seed=seed,temperature=t,
             reward_profile=reward,noveld='on' if reward=='sparse' else 'off',dacer='off')
    jobs.append(j)
 assert len(jobs)==45 and len({j['id'] for j in jobs})==45
 shards={h:[] for h in HOSTS}
 for j in jobs:
  eligible=HOSTS[:2] if j['task']=='v4' else HOSTS[2:] if j['task']=='v1' else HOSTS
  h=min(eligible,key=lambda h:sum(x['steps'] for x in shards[h]))
  shards[h].append(j)
 m.update(campaign=CAMPAIGN,jobs=shards[host],host=host,
          protocol='antmaze_experiments/DACER_OFF_PROTOCOL.md')
 return m

def register(host):
 ROOT.mkdir(exist_ok=True)
 with (ROOT/'manifest.json').open('x') as f:json.dump(manifest(host),f,indent=2)
 for d in ('jobs','logs','preflight','runs'):(ROOT/d).mkdir(exist_ok=True)
 queue.write(ROOT/'status.json',dict(phase='waiting_previous',pending=[j['id'] for j in manifest(host)['jobs']]))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','run']);p.add_argument('--host');a=p.parse_args()
 if a.mode=='register':register(a.host);return
 previous_pending=queue.existing_pending
 def held():
  s=json.loads((OLD/'status.json').read_text())
  # Older scheduler owns first claim while it has pending jobs. Idle slots
  # may backfill once all its jobs have been dispatched (GPU checked below).
  return bool(s.get('pending') or s.get('failed')) or previous_pending()
 queue.ROOT=ROOT;queue.CAMPAIGN=CAMPAIGN;queue.existing_pending=held
 raise SystemExit(queue.run())

if __name__=='__main__':main()
