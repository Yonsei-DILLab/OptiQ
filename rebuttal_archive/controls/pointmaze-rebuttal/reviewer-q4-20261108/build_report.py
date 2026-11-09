from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,statistics,html,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'review-report-20261108T050131Z'
OUT=Path(__file__).resolve().parent
METHODS=('ibolt','sac','sql','td3','mfpo','meow')
COLORS=dict(zip(METHODS,['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']))
LABELS={m:('iBOLT (mu-only)' if m=='ibolt' else m.upper()) for m in METHODS}
reports={};data={};supplementary={}
for maze,K in [('medium',4),('hard',8)]:
 summary=json.loads((SOURCE/maze/'summary.json').read_text())
 entries=json.loads((SOURCE/maze/'per-seed.json').read_text())
 rows=[]
 for method in METHODS:
  entry=[e for e in entries if e['method']==method]
  s=next(r for r in summary['methods'] if r['method']==method and r['evaluation_variant']=='native')
  assert s['complete'] and s['training_completed']==5 and s['final_evaluations']==5
  training=[e['training'] for e in entry]
  all_times=[t['all_goals_discovery_at_collection_end'] for t in training if t['all_goals_discovered']]
  r={k:v for k,v in s.items() if k not in ('evaluation_curve','discovery_curve')}
  r.update(all_goal_discovery_median_conditional=statistics.median(all_times) if all_times else None,
    discovered_by_50k=sum(t['first_reward_at_collection_end'] is not None and t['first_reward_at_collection_end']<=50000 for t in training),
    discovered_by_100k=sum(t['first_reward_at_collection_end'] is not None and t['first_reward_at_collection_end']<=100000 for t in training),
    all_goals_discovered_by_100k=sum(t['all_goals_discovered'] and t['all_goals_discovery_at_collection_end']<=100000 for t in training),
    training_goal_coverage_per_seed=[t['training_goal_coverage'] for t in training],
    evaluation_role='primary',evaluation_mode='mu_only' if method=='ibolt' else 'policy',
    per_seed=[dict(seed=e['seed'],first_reward=e['training']['first_reward_at_collection_end'],first_goal_rewards=e['training']['first_goal_reward_at_collection_end'],all_goal_discovery=e['training']['all_goals_discovery_at_collection_end'],reward_events=e['training']['reward_events'],reward_events_per_100000=e['training']['reward_events_per_100000_transitions'],training_goal_coverage=e['training']['training_goal_coverage'],final=e['evaluations'][-1]['normal']) for e in entry])
  rows.append(r)
 reports[maze]=dict(source_updated_kst=datetime.fromtimestamp(summary['updated_unix'],ZoneInfo('Asia/Seoul')).isoformat(),goals=K,complete_methods=rows,dipo_status='5 runs still training; no final discovery/success estimate reported')
 data[maze]=(summary,entries)
 full=next(r for r in summary['methods'] if r['method']=='ibolt' and r['evaluation_variant']=='full_policy')
 supplementary[maze]=dict(evaluation_role='supplementary',evaluation_mode='full_policy',
  summary={k:v for k,v in full.items() if k not in ('evaluation_curve','discovery_curve')},
  per_seed=[dict(seed=e['seed'],final=e['full_policy_evaluations'][-1]['normal']) for e in entries if e['method']=='ibolt'])
result=dict(protocol=dict(seeds=list(range(5)),transitions=1000192,warmup=8192,utd=1,batch_size=256,vector_envs=256,ibolt_temperature=5,final_eval_episodes_per_seed=500,uncertainty='Sample standard deviation across 5 independent training seeds, ddof=1; episodes are not independent training seeds',first_discovery_resolution=256,success='Any goal reached within horizon; medium300/hard600',coverage='Goals reached at least once in 500 final evaluation episodes, computed within each training seed',full_policy='Fresh latent and conditional box-truncated Gaussian at every action',mean_only='Fresh latent, conditional center only; not fixed z=0',training_discovery='Actual online training rewards; complete replay history reconstructed and audited',time_to_discovery='First-reward times conditioned on discovery; undiscovered seeds right-censored at training budget',source_boundary='Private heechan-rebuttal only; never copy into anonymous branches or exports'),results=reports,limitations=['DIPO still training; no final DIPO comparison yet','Initial critic error/calibration was not measured directly. From-scratch online discovery is not itself a quantitative critic-inaccuracy measurement.','First reward is not universally earlier for iBOLT; SAC also discovers rewards quickly and SQL discovers all goals in all seeds.','Goal visitation is not uniform, particularly on Hard. Full policy covers7/8 goals for one seed in500 episodes.','Raw repeated reward counts can favor repeatedly visiting a nearby goal; they are not the same as new-goal discovery.','These goal-level statistics do not directly establish multimodality of the action distribution at a fixed state.','Shared-GPU elapsed times are not isolated efficiency benchmarks.'])
result['protocol'].update(primary_ibolt_evaluation='mu_only',
 latent_sampling='Fresh Gaussian z independently at every step',
 conditional_gaussian_noise_in_primary_evaluation=False,
 training_discovery_evaluation_independence='Training reward discovery is unchanged by evaluation-mode selection.')
result['supplementary_full_policy']=supplementary
result['limitations']=[s.replace('Goal visitation is not uniform, particularly on Hard. Full policy covers7/8 goals for one seed in500 episodes.',
 'Goal visitation is not uniform, particularly on Hard. Primary mu-only evaluation covers all goals in every seed; supplementary full-policy evaluation covers7/8 goals for one seed in500 episodes.') for s in result['limitations']]
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,4,figsize=(14,6.1),layout='constrained')
for row,(maze,K) in enumerate([('medium',4),('hard',8)]):
 summary,entries=data[maze]
 for method in METHODS:
  s=next(r for r in summary['methods'] if r['method']==method and r['evaluation_variant']=='native')
  curve=s['discovery_curve'];x=np.array([p['transitions'] for p in curve])/1000
  axes[row,0].step(x,[p['discovered_seed_fraction'] for p in curve],where='post',color=COLORS[method],label=method.upper(),lw=1.5)
  axes[row,1].step(x,[p['training_goal_coverage']['mean']/K for p in curve],where='post',color=COLORS[method],lw=1.5)
 for col in (0,1):
  axes[row,col].set(xlim=(0,100),ylim=(-.025,1.04),xlabel='Training transitions (thousands)')
  axes[row,col].grid(alpha=.2)
 axes[row,0].set_ylabel(maze.title()+'\nFraction / normalized coverage')
 selected=reports[maze]['complete_methods']
 for i,(method,r) in enumerate(zip(METHODS,selected)):
  sr=[e['final']['success']*100 for e in r['per_seed']]
  cov=[e['final']['reachable_goals'] for e in r['per_seed']]
  for col,values in ((2,sr),(3,cov)):
   ax=axes[row,col];ax.bar(i,float(np.mean(values)),color=COLORS[method],alpha=.68,width=.65)
   ax.scatter(i+np.linspace(-.16,.16,5),values,s=13,c='black',zorder=3)
 for col in (2,3):
  axes[row,col].set_xticks(range(6),['iBOLT\nmu-only','SAC','SQL','TD3','MFPO','MEOW'],rotation=35,ha='right')
  axes[row,col].grid(axis='y',alpha=.2)
 axes[row,2].set(ylim=(-2,105),ylabel='Final success (%)')
 axes[row,3].set(ylim=(-.05,K+.4),ylabel=f'Goals covered (max {K})',yticks=range(K+1))
for ax,title in zip(axes[0],['Online reward discovery (5 seeds)','Training goal discovery (seed mean)','Final success: mu-only iBOLT / baseline policy','Final goal coverage within each seed']):ax.set_title(title,fontsize=8)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=6,frameon=False)
fig.suptitle('Sparse PointMaze: online discovery and retained goal coverage\n1M transitions, UTD=1, batch=256; iBOLT evaluation: fresh z, conditional mean; five training seeds',fontsize=12)
fig.savefig(OUT/'discovery-and-coverage.png',dpi=180)
fig.savefig(OUT/'discovery-and-coverage.pdf')
plt.close(fig)
def fstat(s,scale=1,places=2):return f"{s['mean']*scale:.{places}f} ± {s['sample_std']*scale:.{places}f}"
sections=[]
for maze in ('medium','hard'):
 rows=reports[maze]['complete_methods'];K=reports[maze]['goals']
 tr=[]
 for r in rows:
  med=r['first_reward_median_conditional_on_discovery']
  values=[LABELS[r['method']],f"{r['reward_discovery_seeds']}/5",f"{r['all_training_goals_discovered_seeds']}/5",f'{med:,.0f}' if med is not None else '—',fstat(r['reward_events_per_100000_transitions']),fstat(r['success'],100),fstat(r['coverage']),fstat(r['normalized_goal_entropy_conditional_on_success'],places=3)+(f" (n={r['normalized_goal_entropy_conditional_on_success']['n']})" if r['normalized_goal_entropy_conditional_on_success']['n']<5 else '')]
  tr.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
 sections.append(f'<h2>{maze.title()} ({K} goals)</h2><p>Source updated {reports[maze]["source_updated_kst"]}. Five independent seeds per complete method;500 evaluation episodes per seed. DIPO pending.</p><table><tr><th>Method</th><th>Reward discovered</th><th>All goals found in training</th><th>First reward median transitions</th><th>Reward events /100K</th><th>Final success %</th><th>Final goal coverage</th><th>Normalized entropy</th></tr>'+''.join(tr)+'</table>')
 per=[]
 for r in rows:
  if r['method']!='ibolt':continue
  for e in r['per_seed']:
   n=e['final'];values=[str(e['seed']),str(e['first_reward']),str(e['all_goal_discovery']),f"{n['success']*100:.2f}",str(n['reachable_goals']),str(n['goals']),f"{n['normalized_goal_entropy']:.4f}"]
   per.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
 sections.append('<h3>Primary iBOLT mu-only: per seed</h3><table><tr><th>Seed</th><th>First reward</th><th>All goals discovered</th><th>Final success %</th><th>Goal coverage</th><th>Goal episode counts</th><th>Entropy</th></tr>'+''.join(per)+'</table>')
 native=next(r for r in data[maze][0]['methods'] if r['method']=='ibolt' and r['evaluation_variant']=='native')
 full=next(r for r in data[maze][0]['methods'] if r['method']=='ibolt' and r['evaluation_variant']=='full_policy')
 sections.append(f"<p>Separate obstacle diagnostic, primary mu-only: single-trial success {fstat(native['obstacle_success'],100)}%; SR5 {fstat(native['obstacle_sr5'],100)}%. These are not the base-task success rates.</p>")
 sections.append(f"<details><summary>Supplementary full-policy evaluation (not used in the main table or figure)</summary><p>Both latent and conditional Gaussian sampled: success {fstat(full['success'],100)}%, coverage {fstat(full['coverage'])}, normalized goal entropy {fstat(full['normalized_goal_entropy_conditional_on_success'],places=3)}. Obstacle single-trial success {fstat(full['obstacle_success'],100)}%; SR5 {fstat(full['obstacle_sr5'],100)}%.</p></details>")
page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>PointMaze reviewer Q4 evidence</title><style>body{font:15px system-ui;max-width:1400px;margin:35px auto;padding:0 24px;line-height:1.55}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccc;padding:7px;text-align:left}img{width:100%}.note{padding:14px;background:#eef4fa}h2{margin-top:30px}</style><h1>PointMaze reviewer Q4: reward discovery, success, and goal coverage</h1><p class="note">Read-only result audit from the timestamps below:60/70 training runs analyzed in this snapshot; DIPO10 final results pending in this snapshot. Success and coverage uncertainty are sample SD over five independently trained seeds, not over episodes. Conditional metrics identify any smaller sample. Primary iBOLT evaluation is mu-only: sample a fresh z at every step and execute its conditional mean; do not sample conditional Gaussian noise. Full-policy results are supplementary. Discovery comes from actual training rewards; goal coverage is computed within each seed.</p><p><a href="summary.json">All statistics and per-seed evidence (JSON)</a> · <a href="discovery-and-coverage.pdf">Exportable PDF figure</a></p><img src="discovery-and-coverage.png" alt="Online discovery, final success and goal coverage across five seeds">'''+''.join(sections)+'<p>TD3 first-reward medians condition on the4/5 runs that discovered reward; the undiscovered run is right-censored at the total training budget. Goal entropy is conditioned on successful episodes and includes only seeds with at least one success; n is shown when fewer than5. PointMaze evaluation: SAC/SQL/MFPO/MEOW direct policy samples; TD3 deterministic actor; iBOLT stochastic-z conditional means (mu-only).</p><h2>Interpretation limits</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in result['limitations'])+'</ul><p>Private heechan-rebuttal evidence only. Must not be copied into anonymous branches or anonymous exports.</p></html>'
(OUT/'report.html').write_text(page)
print(json.dumps({'output':str(OUT),'figure':str(OUT/'discovery-and-coverage.png'),'complete_training_runs':60,'pending_dipo_runs':10,'primary_ibolt_evaluation':'mu_only'}))
