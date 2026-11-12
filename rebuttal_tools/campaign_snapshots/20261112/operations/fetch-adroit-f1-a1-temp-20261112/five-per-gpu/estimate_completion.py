"""Queue/dependency forecast, not an experimental result or a statistical confidence interval."""
from pathlib import Path
import json,collections,statistics,random,math,heapq,datetime
P=Path(__file__).resolve().parent;B=P.parent;HS=[1,3,4,5];LIMIT=5000192
before={h:json.loads((P/f'rate-start-host{h}.json').read_text()) for h in HS}
after={h:json.loads((B/f'expanded-host{h}.json').read_text()) for h in HS}
bygroup=collections.defaultdict(list);bytask=collections.defaultdict(list);measurement=[]
for h in HS:
 old={r['id']:r for r in before[h]['running']}
 for r in after[h]['running']:
  if r['id'] not in old:continue
  x=r['progress'] or {};y=old[r['id']]['progress'] or {}
  dt=(x.get('observed_unix') or 0)-(y.get('observed_unix') or 0);ds=(x.get('steps') or 0)-(y.get('steps') or 0)
  if dt<60 or ds<=0 or (y.get('steps') or 0)<12800:continue
  rate=ds/dt;bygroup[h,r['family'],r['method']].append(rate);bytask[h,r['family'],r['method'],r['task']].append(rate)
  measurement.append(dict(host=h,id=r['id'],steps_per_second=rate,window_seconds=dt))
# SAC Adroit on host3 already finished; retain its measured ten-minute pre-expansion rate
# conservatively scaled by the increase from 3 to 5 learners per GPU.
oldhealth={r['id']:r for r in json.loads((B/'heartbeat-health.json').read_text())['active']}
for h in HS:
 for j in before[h]['plan']['jobs']:
  if j['status']!='running':continue
  key=(h,j['family'],j['method'])
  if key in bygroup:continue
  r=next((r for r in before[h]['running'] if r['id']==j['id']),None)
  if not r or not r.get('progress'):continue
  x=r['progress'];elapsed=x.get('elapsed_seconds') or 0
  if elapsed>300 and x.get('steps',0)>50000:bygroup[key].append(x['steps']/elapsed*.75)
assert all((h,'fetch' if h==1 else 'adroit','ibolt') in bygroup for h in HS)
rates={key:statistics.median(v) for key,v in bygroup.items()}
# No future Adroit SAC is assigned to host3; no extrapolation is needed for it.
alljobs=[dict(j,host=h) for h in HS for j in after[h]['plan']['jobs'] if j['phase']!='smoke']
progress={r['id']:r['progress'] or {} for h in HS for r in after[h]['running']}
cap={1:20,3:20,4:20,5:15};keys=sorted({(j['family'],j['task']) for j in alljobs if j['phase']=='pilot'})

def jobrate(j,scale):
 key=(j['host'],j['family'],j['method']);specific=bytask.get(key+(j['task'],))
 return (statistics.median(specific) if specific else rates[key])*scale

def simulate(choice,scale):
 jobs=[j for j in alljobs if j['phase']!='confirm' or j['temperature']==choice[j['family'],j['task']]]
 assert len(jobs)==160
 done={j['id']:0. for j in jobs if j['status'] in ['completed','reused']};active=collections.Counter();events=[];pending=[];t=0.
 deps={(f,k):{j['id'] for j in jobs if j['phase']=='pilot' and j['family']==f and j['task']==k} for f,k in keys}
 for j in jobs:
  if j['id'] in done:continue
  if j['status']=='running':
   remain=max(0,LIMIT-progress.get(j['id'],{}).get('steps',0));finish=remain/jobrate(j,scale)+120
   heapq.heappush(events,(finish,j['id'],j));active[j['host']]+=1
  else:pending.append(j)
 while pending or events:
  for h in HS:
   while active[h]<cap[h]:
    ix=next((i for i,j in enumerate(pending) if j['host']==h and (j['phase']!='confirm' or deps[j['family'],j['task']].issubset(done))),None)
    if ix is None:break
    j=pending.pop(ix);duration=LIMIT/jobrate(j,scale)+240
    heapq.heappush(events,(t+duration,j['id'],j));active[h]+=1
  if not events:raise RuntimeError('Blocked dependency simulation')
  t,_,j=heapq.heappop(events);done[j['id']]=t;active[j['host']]-=1
 finish={f:max(done[j['id']] for j in jobs if j['family']==f)/3600 for f in ['fetch','adroit']}
 finish['all']=max(finish.values());finish['pilot']=max(done[j['id']] for j in jobs if j['phase']=='pilot')/3600;finish['sac']=max(done[j['id']] for j in jobs if j['phase']=='baseline')/3600
 return finish
rng=random.Random(20261112);temps=[.05,.1,.25,.5,1.]
choices=[{k:t for k in keys} for t in temps]+[{k:rng.choice(temps) for k in keys} for _ in range(95)]
scenarios={}
for label,scale in [('faster',1.2),('observed',1.),('slower',.8)]:
 vals=[simulate(c,scale) for c in choices];scenarios[label]={k:dict(min=min(v[k] for v in vals),median=statistics.median(v[k] for v in vals),max=max(v[k] for v in vals)) for k in vals[0]}
now=max(after[h]['at'] for h in HS);tz=datetime.timezone(datetime.timedelta(hours=9))
result=dict(as_of_unix=now,as_of_kst=datetime.datetime.fromtimestamp(now,tz).isoformat(),scientific_runs=160,done=sum(j['status'] in ['completed','reused'] for j in alljobs),capacity=cap,measurements=measurement,rates={str(k):v for k,v in rates.items()},remaining_hours=scenarios,assumptions=['Current host placement and queue order; no future work stealing','Confirmation waits for all ten pilots of its task','Unknown selected temperature changes seed-to-host placement; 100 placement scenarios','Five concurrent learners per available GPU; no foreign GPU use','Rates are recent wall-clock observations; +/-20% speed scenarios, not a confidence interval','Adds two minutes final evaluation and two minutes new-job startup overhead'])
(P/'eta.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='measurements'},indent=2))
