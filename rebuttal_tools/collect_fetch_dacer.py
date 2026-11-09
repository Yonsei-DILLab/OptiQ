"""Read both DACER cohorts and compare profiles without selecting on confirmation seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import base64
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time
import zlib

QUERY=r'''
import pathlib,json,time,os,subprocess
base=pathlib.Path(BASE);plan=json.loads((base/'plan.json').read_text());rows=[]
def live(pid,start=None):
 try:
  fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return fields[0]!='Z' and (start is None or int(fields[19])==start)
 except OSError:return False
for job in plan['jobs']:
 out=pathlib.Path(job['output']);row={k:job.get(k) for k in ('id','method','task','seed','target_entropy_per_dim','noise_scale','phase','status','pid','start_ticks','external','output')}
 row['process_live']=live(job.get('pid'),job.get('start_ticks'))
 for filename,key in [('progress.json','progress'),('dacer_regulator.json','regulator'),('dacer-worker-failure.json','failure'),('scale-worker-failure.json','scale_failure')]:
  p=out/filename
  if p.exists():row[key]=json.loads(p.read_text())
 marker=out/('verified-completion.json' if job.get('external') else 'dacer-verified-completion.json')
 row['verified_complete']=marker.exists()
 row['evaluations']=[];p=out/'evaluations.jsonl'
 if p.exists():
  values={}
  for line in p.read_text().splitlines():
   if not line.strip():continue
   try:r=json.loads(line)
   except ValueError:continue
   v=r[r['primary_mode']];values[r['steps']]=dict(step=r['steps'],episodes=v['episodes'],success=v['success_rate'],new_success=v['new_final_success_count']/v['episodes'])
  row['evaluations']=[values[k] for k in sorted(values)]
 row['heights']=[];p=out/'height-evaluations.jsonl'
 if p.exists():
  values={}
  for line in p.read_text().splitlines():
   if not line.strip():continue
   try:r=json.loads(line);values[r['steps']]=dict(step=r['steps'],**r['modes']['stochastic_z'])
   except ValueError:continue
  row['heights']=[values[k] for k in sorted(values)]
 rows.append(row)
pid=plan.get('controller_pid');cmd=''
try:cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(bytes([0]),b' ').decode()
except OSError:pass
result=dict(checked_unix=time.time(),status=plan['status'],controller_pid=pid,controller_live=live(pid) and str(base/'plan.json') in cmd,
 error=plan.get('error'),selection=plan.get('selection'),jobs=rows,source_commit=plan['source_commit'],preflights_passed=plan.get('preflights_passed'))
print(json.dumps(result,separators=(',',':')))
'''


def stats(values):
    return dict(mean=statistics.mean(values),seed_sd=statistics.stdev(values) if len(values)>1 else None,values=values)


def interim_map(job):return {r['step']:r for r in job['evaluations'] if 0<r['step']<1000000 and r['episodes']==10}


def summarize_pair(jobs,cap=None):
    assert sorted(j['seed'] for j in jobs)==[0,1]
    jobs=sorted(jobs,key=lambda r:r['seed']);maps=[interim_map(j) for j in jobs]
    common=set.intersection(*(set(m) for m in maps))
    if cap is not None:common={s for s in common if s<=cap}
    result=dict(verified_complete=all(j['verified_complete'] for j in jobs),
        training_processes_live=sum(j['process_live'] for j in jobs),
        progress_steps=[j.get('progress',{}).get('steps',0) for j in jobs])
    if common:
        steps=sorted(common);late=steps[-5:]
        result.update(common_step=steps[-1],last5_steps=late,evaluation_count=len(late),
            last5_success=stats([statistics.mean(m[s]['success'] for s in late) for m in maps]),
            last5_new_success=stats([statistics.mean(m[s]['new_success'] for s in late) for m in maps]),
            auc=stats([statistics.mean(m[s]['success'] for s in steps) for m in maps]))
        heights=[{r['step']:r for r in j['heights']} for j in jobs]
        if all(set(late)<=set(h) for h in heights):
            n=sum(h[s]['air_over5cm']['episodes'] for h in heights for s in late)
            hits=sum(h[s]['air_over5cm']['final_successes'] for h in heights for s in late)
            result['air_success_rate']=hits/n if n else 0.;result['air_successes']=hits;result['air_episodes']=n
    return result


def summarize(hosts,sac):
    jobs=[dict(job,host=host['host']) for host in hosts for job in host['jobs']]
    search=[j for j in jobs if j['phase']=='search'];profiles=sorted({(j['target_entropy_per_dim'],j['noise_scale']) for j in search})
    groups={p:[j for j in search if (j['target_entropy_per_dim'],j['noise_scale'])==p] for p in profiles}
    summaries={p:summarize_pair(group) for p,group in groups.items()}
    common_budget=min((s.get('common_step',0) for s in summaries.values()),default=0)
    result=dict(checked_unix=time.time(),profiles=[],all_search_complete=all(j['verified_complete'] for j in search),
        search_runs=len(search),live_training_processes=sum(j['process_live'] for j in search),
        counts=dict(__import__('collections').Counter(j['status'] for j in search)),matched_interim_budget=common_budget,
        large_advantage_verified=False,selection_uses='Only training seeds0/1, never confirmation-seed results.',
        completion_standard='Full five-seed mean gains >=10pp on both last5 and final100, plus >=10pp final100 mean gain on additional seeds2/3/4; positive improvement on high-air goals.')
    for p,summary in summaries.items():
        entry=dict(target_entropy_per_dim=p[0],noise_scale=p[1],**summary)
        if common_budget:entry['matched_budget']=summarize_pair(groups[p],common_budget)
        result['profiles'].append(entry)
    if not result['all_search_complete']:return result
    assert all(s['evaluation_count']==5 and s['common_step']==995000 for s in summaries.values())
    chosen=max(profiles,key=lambda p:(summaries[p]['last5_success']['mean'],summaries[p]['air_success_rate'],
        summaries[p]['auc']['mean'],-abs(p[0]+.9),-abs(__import__('math').log(p[1]/.1))))
    result['selected_parameters']=dict(target_entropy_per_dim=chosen[0],noise_scale=chosen[1])
    group=sorted([j for j in jobs if (j['target_entropy_per_dim'],j['noise_scale'])==chosen],key=lambda r:r['seed'])
    result['selected_seed_count']=len(group)
    if len(group)!=5 or [j['seed'] for j in group]!=list(range(5)) or not all(j['verified_complete'] for j in group):return result
    reference=sorted(sac['seeds'],key=lambda r:r['seed']);last=[];final=[];height_hits=height_n=0
    for job in group:
        es=interim_map(job);late=sorted(es)[-5:];last.append(statistics.mean(es[s]['success'] for s in late))
        final.append(next(r['success'] for r in job['evaluations'] if r['step']==1000000 and r['episodes']==100))
        h=next(r['air_over5cm'] for r in job['heights'] if r['step']==1000000);height_hits+=h['final_successes'];height_n+=h['episodes']
    last_gap=[a-b['last5_success'] for a,b in zip(last,reference)];final_gap=[a-b['final100_success'] for a,b in zip(final,reference)]
    result.update(last5_success=stats(last),final100_success=stats(final),paired_last5_gap=stats(last_gap),
        paired_final100_gap=stats(final_gap),additional_seed_final100_gap=stats(final_gap[2:]),
        air_successes=height_hits,air_episodes=height_n)
    result['large_advantage_verified']=(statistics.mean(last_gap)>=.1-1e-12 and statistics.mean(final_gap)>=.1-1e-12
        and statistics.mean(final_gap[2:])>=.1-1e-12 and height_hits>sac['final100_height']['air_over5cm']['successes'])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    main_folder=Path(Path('/tmp/ibolt-fetch-dacer-path').read_text().strip())
    extra=Path(Path('/tmp/ibolt-fetch-dacer-strong-path').read_text().strip())
    specs=[(4,'/workspace/heechan-fetch-sigma/outputs/'+main_folder.name),
           (3,'/workspace/heechan-fetch-dacer/outputs/'+extra.name)]
    output=args.output or main_folder/'global-monitor';output.mkdir(exist_ok=True)
    def collect(spec):
        host,base=spec;blob=base64.b64encode(zlib.compress(('BASE='+repr(base)+'\n'+QUERY).encode())).decode()
        command='python3 -c '+shlex.quote('import zlib,base64;exec(zlib.decompress(base64.b64decode('+repr(blob)+')))')
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ControlMaster=no','-o','ControlPath=none',
            f'ibolt-server-{host}',command],capture_output=True,text=True,timeout=45);r.check_returncode()
        d=json.loads(r.stdout);d['host']=host;(output/f'host{host}.json').write_text(json.dumps(d,indent=2)+'\n');return d
    with ThreadPoolExecutor(max_workers=2) as pool:hosts=list(pool.map(collect,specs))
    report=summarize(hosts,json.loads((main_folder/'sac-reference.json').read_text()))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CONTROLLERS',[(h['host'],h['status'],h['controller_live'],h['controller_pid']) for h in hosts])
    print('SEARCH',report['search_runs'],'LIVE',report['live_training_processes'],'COUNTS',report['counts'],
          'MATCHED_BUDGET',report['matched_interim_budget'],'LARGE_ADVANTAGE',report['large_advantage_verified'])
    for p in report['profiles']:
        s=p.get('last5_success',{});print('H',p['target_entropy_per_dim'],'L',p['noise_scale'],'steps',p['progress_steps'],
            'last5',round(s['mean']*100,1) if s else None,'air',p.get('air_successes'),'complete',p['verified_complete'])


if __name__=='__main__':main()
