"""Continuously summarize the six fixed runs without selecting partial winners."""
import argparse
import json
from pathlib import Path
import statistics
import time


def aggregate(values):
    return dict(mean=statistics.mean(values),seed_std=statistics.stdev(values) if len(values)>1 else None)


def records(path):
    values=[]
    for line in path.read_text().splitlines():
        try:values.append(json.loads(line))
        except json.JSONDecodeError:continue  # A writer may still be flushing its last record.
    return values


def summarize(plan,reference):
    rows=[]
    for job in plan['jobs']:
        output=Path(job['output']);row={k:job[k] for k in ('variant','seed','status','output')}
        for name,key in [('progress.json','progress'),('completed.json','completed'),('failure.json','failure')]:
            f=output/name
            if f.exists():row[key]=json.loads(f.read_text())
        f=output/'evaluations.jsonl'
        if f.exists():
            evaluations=records(f)
            interim=[r for r in evaluations if r['steps']<1000192 and r['stochastic_z']['episodes']==10][-5:]
            if interim:
                row['last5_interim']=dict(steps=[r['steps'] for r in interim],
                    evaluation_count=len(interim),success=statistics.mean(r['stochastic_z']['success_rate'] for r in interim))
            if evaluations:
                row['latest_evaluation']=dict(steps=evaluations[-1]['steps'],
                    success=evaluations[-1]['stochastic_z']['success_rate'],episodes=evaluations[-1]['stochastic_z']['episodes'])
        f=output/'height-evaluations.jsonl'
        if f.exists() and records(f):row['latest_height_evaluation']=records(f)[-1]
        rows.append(row)
    complete=all(r.get('completed',{}).get('completed') for r in rows)
    groups={}
    for variant in ('original','narrow','wide'):
        group=[r for r in rows if r['variant']==variant]
        entry=dict(seeds=group,complete=all(r.get('completed',{}).get('completed') for r in group))
        if entry['complete']:
            assert len(group)==2 and all(r['completed']['steps']==1000192 for r in group)
            entry['last5_interim_success']=aggregate([r['last5_interim']['success'] for r in group])
            entry['final100_success']=aggregate([r['completed']['final_success'] for r in group])
            entry['elapsed_seconds']=aggregate([r['completed']['elapsed_seconds'] for r in group])
            strata=[r['latest_height_evaluation']['modes']['stochastic_z'] for r in group]
            entry['final_height_success']={k:dict(successes=sum(s[k]['final_successes'] for s in strata),
                episodes=sum(s[k]['episodes'] for s in strata)) for k in ('ground','low_air','air_over5cm')}
        groups[variant]=entry
    best=[]
    if complete:
        top=max(g['last5_interim_success']['mean'] for g in groups.values())
        best=[k for k,g in groups.items() if abs(g['last5_interim_success']['mean']-top)<1e-12]
        for entry in groups.values():entry['last5_delta_vs_vector_original']=entry['last5_interim_success']['mean']-groups['original']['last5_interim_success']['mean']
    return dict(checked_unix=time.time(),complete=complete,controller_status=plan['status'],
        groups=groups,previous_single_env_utd1=reference,
        best_variants_by_last5_interim=best,
        primary_rule='Mean over seeds 0/1 of each seed\'s last five 10-episode evaluations before the 100-episode final evaluation.',
        comparison_limits='Six runs share vec256 and UTD1/8; previous single-env UTD1 results also differ in warmup, sampling and update budget, hardware and GPU concurrency.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    while True:
        plan=json.loads(args.plan.read_text());reference=json.loads(args.reference.read_text())
        result=summarize(plan,reference);dest=args.plan.parent/'comparison.json';temp=dest.with_suffix('.tmp')
        temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest)
        if args.once or result['complete'] or plan['status'] in ('controller_error','stopped','completed_with_failures'):return
        time.sleep(15)


if __name__=='__main__':main()
