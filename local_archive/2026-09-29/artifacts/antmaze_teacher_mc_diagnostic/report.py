"""Post-hoc verification of archived scratch arrays; no training or rollout."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
out=R/'report';out.mkdir(exist_ok=True)
all_results={}
for host in ('vast-heechan-180','vast-heechan-199'):
    folder=R/'results'/host/'diagnostic'
    result=json.loads((folder/'result.json').read_text())
    assert result['completed'] and not result['training_performed'] and not result['rollouts_performed']
    assert result['existing_M64_probe_matched'] and result['model_optimizer_rng_checkpoint_unchanged']
    assert hashlib.sha256((folder/'teacher-mc.npz').read_bytes()).hexdigest()==result['raw_sha256']
    with np.load(folder/'teacher-mc.npz') as raw:
        for count,rows in result['records'].items():
            for row in rows:
                p=f"rep{row['rep']}_M{count}_";q=f"rep{row['rep']}_reference_"
                w=raw[p+'weights'];g=raw[p+'gradient'];h=raw[q+'gradient']
                assert np.isfinite(g).all() and np.isfinite(w).all()
                np.testing.assert_allclose(w.sum(1),1,atol=2e-6)
                values=dict(ess=1/(w*w).sum(1),max_weight=w.max(1),
                    output_gradient_cosine=(g*h).sum(1)/(np.linalg.norm(g,axis=1)*np.linalg.norm(h,axis=1)),
                    output_gradient_relative_error=np.linalg.norm(g-h,axis=1)/np.linalg.norm(h,axis=1),
                    weighted_action_l2_error=np.linalg.norm(raw[p+'action_mean']-raw[q+'action_mean'],axis=1),
                    weighted_q_absolute_error=np.abs(raw[p+'q_mean']-raw[q+'q_mean']),source_q_std=raw[p+'q'].std(1))
                for k,v in values.items():np.testing.assert_allclose(v,row[k],atol=2e-6,rtol=2e-5)
            for k,stats in result['summary'][count].items():
                x=np.asarray([r[k] for r in rows]);check=[x.mean(),np.median(x),np.quantile(x,.1),np.quantile(x,.9)]
                np.testing.assert_allclose(check,[stats[k] for k in ('mean','median','p10','p90')],atol=1e-10)
    all_results[result['task']]=result
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for i,(task,r) in enumerate(all_results.items()):
    for ax,key,title in zip(axes[i],('ess','output_gradient_cosine','weighted_action_l2_error'),('Effective sample size','Output-gradient cosine','Weighted action error (L2)')):
        x=r['candidates'];s=[r['summary'][str(n)][key] for n in x]
        ax.plot(x,[v['mean'] for v in s],marker='o')
        ax.fill_between(x,[v['p10'] for v in s],[v['p90'] for v in s],alpha=.15)
        ax.set_xscale('log',base=2);ax.set_xticks(x,[str(n) for n in x]);ax.set_xlabel('Scratch candidate count')
        ax.set_title(task+' | '+title);ax.grid(alpha=.2)
fig.suptitle('Frozen teacher Monte Carlo diagnostic | 8 replay states x 8 latent clouds\nMean and 10-90% empirical range; independent finite 4096 reference; training N=M=64 unchanged',fontsize=12)
fig.savefig(out/'teacher_mc_diagnostic.png',dpi=160);plt.close(fig)
lines=['# 고정 모델 teacher 표본 진단','',
'학습이나 rollout을 추가하지 않은 진단입니다. 실제 replay에서 고른 8개 상태, 8개 latent cloud의 원시 배열과 통계를 재검산했습니다.','',
'|환경|후보 수|평균 ESS|최대 weight 평균|출력 gradient cosine|가중 action 평균 오차|',
'|---|---:|---:|---:|---:|---:|']
for task,r in all_results.items():
    for n,s in r['summary'].items():
        lines.append(f"|{task}|{n}|{s['ess']['mean']:.2f}|{s['max_weight']['mean']:.3f}|{s['output_gradient_cosine']['mean']:.3f}|{s['weighted_action_l2_error']['mean']:.3f}|")
lines+=['','![진단](teacher_mc_diagnostic.png)','',
'64개 후보에서 ESS가 평균 26–33이므로 한 후보에 거의 모든 가중치가 몰린 상황은 아닙니다. 표본에 따른 출력 gradient 변동은 관찰되지만, 이는 실제 batch4096의 네트워크 parameter gradient 분산을 측정한 결과가 아닙니다. 4096 기준도 유한한 표본이며 true Q나 정확한 타겟이 아닙니다.',
'이 결과만으로 경로 붕괴 원인을 확정하거나 N/M 변경을 정당화하지 않습니다. 실제 학습 N=M64는 유지합니다. 요청한 좌표 대신 가장 가까운 실제 replay 상태를 사용했으므로 방문하지 않은 지점의 진단으로 해석하면 안 됩니다.']
(out/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
(out/'verification.json').write_text(json.dumps(dict(verified=True,report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tasks={t:dict(raw_sha256=r['raw_sha256'],training_source=r['training_source'],reporting_source=r['reporting_source']) for t,r in all_results.items()},training_changed=False),indent=2)+'\n')
print(json.dumps(dict(report=str(out),raw_recomputed=True)))
