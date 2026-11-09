
import pathlib,json,time,subprocess
p=pathlib.Path(BASE);handoff=p/'controller-handoff.json';planpath=pathlib.Path(json.loads(handoff.read_text())['plan']) if handoff.exists() else p/'plan.json';plan=json.loads(planpath.read_text());d={'checked_unix':time.time(),'status':plan['status'],'counts':plan.get('counts'),'selections':plan.get('selections'),'mfpo_selection':plan.get('mfpo_selection'),'jobs':[]}
for j in plan['jobs']:
 if j['status']=='waiting_for_previous_task':continue
 o=pathlib.Path(j['output']);r={k:j.get(k) for k in ['id','method','seed','temperature','task','status','error','phase','target_entropy_coeff','initial_temperature']}
 if (o/'progress.json').exists():r['progress']=json.loads((o/'progress.json').read_text())
 f=o/'evaluations.jsonl'
 if f.exists():
  es=[json.loads(s) for s in f.read_text().splitlines() if s.strip()]
  r['last5']=[dict(steps=e['steps'],episodes=e[e['primary_mode']]['episodes'],success=e[e['primary_mode']]['success_rate'],any_success=e[e['primary_mode']]['any_success_rate'],new_final=e[e['primary_mode']]['new_final_success_count'],new_any=e[e['primary_mode']]['new_any_success_count']) for e in es[-5:]]
 d['jobs'].append(r)
d['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
print(json.dumps(d))
