"""Combine completed GMM/non-GMM screens into one user-review artifact, no jobs."""
import argparse,json,shutil,html,base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..kl_diverse_targets_1d.target import reference as gmref
from .target import reference as ngref
from ..kl_diverse_targets_1d.report import load,gate

LABELS={
 't00_reference':'A. Three Gaussian modes (reference)',
 't01_two_offset':'Two Gaussian modes (offset)',
 't03_two_symmetric':'Two symmetric Gaussian modes',
 't05_unequal_mass':'Three Gaussian modes (unequal mass)',
 't06_minor_mode':'Three Gaussian modes (small left mode)',
 'n00_spike_ramp':'Narrow spike + rising shoulder',
 'n01_spike_ramp_equal':'Spike + rising shoulder (equal mass)',
 'n02_three_plateaus':'Three flat plateaus',
 'n03_skew_shelves':'Three asymmetric ramps',
 'n04_spike_two_shelves':'Sharp spike between broad plateaus',
 'n05_four_plateaus':'Four plateaus with unequal masses',
 'n06_ripple_shelf':'Five peaks from a corrugated shelf',
 'n07_spike_flat_ramp':'Spike + plateau + ramp'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--studies',type=Path,required=True);p.add_argument('--reports',type=Path,required=True);args=p.parse_args()
 out=args.reports/'20260925_kl_candidate_review';out.mkdir(parents=True,exist_ok=True)
 families=[];byid={}
 for dirname,module,ref in [('20260925_kl_diverse_targets','kl_diverse_targets_1d',gmref),('20260925_kl_nongmm_targets','kl_nongmm_targets_1d',ngref)]:
  cfg=json.loads((args.studies/dirname/'config.json').read_text());data=load(args.studies/dirname,cfg['cases']);families.append((dirname,cfg,data))
  for c in cfg['cases']:byid[c['id']]=(cfg,c,data[c['id']],ref,dirname)
 selected=[]
 # Diverse non-GMM families take precedence over duplicating Gaussian weights.
 priority=['t00_reference','n00_spike_ramp','n02_three_plateaus','n03_skew_shelves','n04_spike_two_shelves','n05_four_plateaus','n07_spike_flat_ramp','n06_ripple_shelf','n01_spike_ramp_equal','t01_two_offset','t05_unequal_mass','t06_minor_mode','t03_two_symmetric']
 for ident in priority:
  if ident=='n01_spike_ramp_equal' and 'n00_spike_ramp' in selected:continue
  if gate(byid[ident][2],range(4)) and len(selected)<6:selected.append(ident)
 plt.style.use('default');plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 lines=['# Forward KL / Reverse KL: 본 실험 전 후보 검토','',
 '**현재는 Reverse L=1024, 100K 탐색 결과다. L=2²⁰ 본 실험은 아직 제출하지 않았다.**','',
 '## 무엇을 바꿨는가','',
 'Actor는 기존의 implicit Gaussian mixture 그대로다. 목표 density만 바꿨다. 기존 GMM16종 외에, Gaussian을 사용하지 않고 직접 정의한8종을 추가했다. 처음 그림을 폭이 다른 GMM으로 해석했던 별도 실행은 사용자 정정 후 취소했고, 본 후보군에서는 제외했다.','',
 '### Non-GMM 목표의 정의','',
 r'\(S(x)=1/(1+e^{-x})\)라 하자. 다음 함수들을 직접 정규화해 목표를 구성한다.','',
 r'$$f_{\rm spike}(a)=\frac1hS\left(\frac{a-c}{h}\right)S\left(-\frac{a-c}{h}\right),$$','',
 r'$$f_{\rm shelf}(a)=S\left(\frac{a-l}{e_L}\right)S\left(\frac{r-a}{e_R}\right)\exp\left[\beta\left(a-\frac{l+r}{2}\right)\right].$$','',
 'beta=0은 평평한 plateau, beta>0은 오른쪽으로 상승하는 비대칭 density다. 좌우 경계 폭을 독립적으로 정한다. Ripple 후보는 shelf에 양수인 cosine 굴곡을 곱한다. 급격한 경계를 얇은 sigmoid로 매끄럽게 만들어 reverse action score를 정의한다. Gaussian을 target fitting에 사용하지 않는다.','',
 r'$$p^\star(a)=\sum_k\eta_k\frac{f_k(a)}{Z_k},\quad Z_k=\int_{-10}^{10}f_k(x)dx,\qquad Q(a)=0.25\log p^\star(a).$$','',
 '## 공통 설정과 판정','',
 '|항목|설정|','|---|---|','|학습|N=M128, batch32,100K updates,Adam3e-4|','|Actor|256×256 GELU,mean-head 초기화scale3,logsigma[-5,-1],초기-1|','|Action / temperature|[-10,10] / 0.25|','|Reverse density bank|L1024,독립 재표집|','|Seeds|초기0·1,통과 후보에2·3 추가|','|평가|각 seed2¹⁸ 실제action,512bin,histogram,noKDE|','',
 '그림의 검정 점선은 target, 색칠한 굵은 곡선은 네 seed histogram의 평균이다. 가는 선은 각 seed다. **평균 TV는 seed별 TV의 평균이며, 평균 density의 TV가 아니다.** 서로 다른 mode를 놓친 seed들이 평균에서는 모두 복구한 것처럼 보일 수 있으므로 개별seed그림과 누락수를 함께 본다.','',
 'Mode missing은 해당 peak/core와 basin의 질량이 모두 target의25%미만인 경우다. Non-GMM core는 plateau/peak 내부의 사전 지정 구간, basin은 target의 골짜기 경계다. Forward 후보 통과는 모든 core/basin의50%이상 복구 및 valley검사 통과,TV≤.15다. Reverse는 적어도1mode를 놓쳐야 하며 네 seed모두에서 이 조건을 확인한다. 모든 실패/반대결과를 포함한 원자료 보고서도 함께 제공한다.','',
 '이 조건들은 요청한 차이가 드러나는 예시를 찾기 위한 탐색 기준이다. 선택된 사례를 전체 환경에서의 보편적 우열로 해석하지 않는다. Mode 복구는 완벽한 density fitting이라는 뜻이 아니며 TV를 함께 제시한다.','',
 '## 목표 곡선','', '![Non-GMM targets](nongmm_targets.png)','',
 '## 네 seed에서 통과한 제안 후보','',
 '|ID|형태|Forward 평균TV|Reverse 평균TV|Reverse 누락mode수(seed0–3)|','|---|---|---:|---:|---|']
 shutil.copy2(args.reports/'20260925_kl_nongmm_targets/target_candidates.png',out/'nongmm_targets.png')
 if selected:
  fig,axes=plt.subplots(len(selected),2,figsize=(11.4,2.55*len(selected)),squeeze=False)
  for row,ident in enumerate(selected):
   cfg,c,runs,ref,dirname=byid[ident];x=np.linspace(-10,10,16385);target=ref(dict(cfg,**c),x)[0]
   ft=[runs[f'forward_s{s}']['metric']['histogram_TV'] for s in range(4)];rt=[runs[f'reverse_s{s}']['metric']['histogram_TV'] for s in range(4)];miss=[runs[f'reverse_s{s}']['metric']['missing_modes'] for s in range(4)]
   lines.append(f'|{ident}|{LABELS[ident]}|{np.mean(ft):.4f}|{np.mean(rt):.4f}|{miss}|')
   for mi,method in enumerate(['forward','reverse']):
    ax=axes[row,mi];ys=[]
    for seed in range(4):
     z=np.load(Path(runs[f'{method}_s{seed}']['folder'])/'samples_100000.npz');edges=z['edges'];y=z['histogram_mass']/np.diff(edges);ys.append(y)
     ax.stairs(y,edges,color=f'C{mi}',alpha=.25,lw=.65)
    ax.stairs(np.mean(ys,axis=0),edges,fill=True,alpha=.17,color=f'C{mi}',lw=0)
    ax.stairs(np.mean(ys,axis=0),edges,color=f'C{mi}',lw=1.2,label='Mean of 4 seeds')
    ax.plot(x,target,'k--',lw=1.1,label='Target');ax.set(xlim=(-10,10),xlabel='Action',ylabel='Density')
    tvs=ft if method=='forward' else rt
    ax.set_title(f'{LABELS[ident].removeprefix("A. ")} | {method.capitalize()} KL\nMean seed TV={np.mean(tvs):.3f}',fontsize=10)
    if method=='reverse':ax.text(.02,.96,f'Missing modes per seed: {miss}',transform=ax.transAxes,va='top',fontsize=8)
    if row==0:ax.legend(fontsize=8,loc='upper right')
   # Full seed panels remain accessible alongside each selected environment.
   shutil.copy2(args.reports/dirname/f'{ident}_all_seeds.png',out/f'{ident}_all_seeds.png')
  fig.tight_layout()
  for ext in ['png','pdf','svg']:fig.savefig(out/f'candidate_comparison.{ext}',dpi=190)
  plt.close(fig)
  lines+=['','![후보 비교](candidate_comparison.png)','']
  for ident in selected:lines+=[f'[{ident}: 네 seed 개별 histogram]({ident}_all_seeds.png)','']
 lines+=['**대칭 2-mode(t03) 주의:** 네 seed가 모두1mode만 복구해도, 서로 반대 mode를 선택하면 평균 histogram은 양쪽을 덮는다. 이 경우 평균 density가 아닌 개별seed곡선과 seed별TV/누락수를 해석해야 한다.','', '## 전체 결과와 설정','', '[기존 GMM 후보: 모든 성공·실패](../20260925_kl_diverse_targets/report.md)','', '[Non-GMM 후보: 모든 성공·실패 및 정확한 파라미터](../20260925_kl_nongmm_targets/report.md)','',
 '## 다음 단계','', '후보 검토 후 승인한 환경만 Reverse L=2²⁰,100K 본 실험으로 진행한다. 현재 결과만으로 L=2²⁰에서도 같은 양상이 유지된다고 단정하지 않는다.']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 (out/'SELECTION.json').write_text(json.dumps(dict(selected=selected,high_L_approved=False,high_L_jobs=[],all_stage_completed_runs={name:sum(map(len,data.values())) for name,cfg,data in families}),indent=2)+'\n')
 print(json.dumps(dict(selected=selected,output=str(out)),indent=2))
if __name__=='__main__':main()
