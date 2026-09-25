import argparse,base64,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm,SymLogNorm
from matplotlib.patches import Patch

COLORS=['#14876b','#d55e00','#7057a3','#a8a8a8']
LABELS=['Global optimum','Bad local minimum','Saddle','Unresolved']

def title(c):return 'Target: ('+', '.join(f'{x:g}' for x in c['centers'])+')'
def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=210,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def legend(fig):fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(COLORS,LABELS)],loc='lower center',ncol=4,frameon=False)
def cbar(fig,im,axes):
    cb=fig.colorbar(im,ax=axes,shrink=.8,pad=.02);ticks=np.array([0,.1,1,3,10,30,100]);cb.set_ticks(np.log1p(ticks),labels=[f'{x:g}' for x in ticks]);cb.set_label('Forward KL (log color scale)')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--analysis-commit',required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    cfg=json.loads(Path(__file__).with_name('config.json').read_text());cases=cfg['cases'];surf={};metrics={};basins={}
    for c in cases:
        folder=a.root/'surface'/c['id']
        if (folder/'COMPLETE.json').exists():surf[c['id']]=dict(np.load(folder/'surface.npz'));metrics[c['id']]=json.loads((folder/'COMPLETE.json').read_text())
    for cid in cfg['basin_cases']:
        if (a.root/'basin'/cid/'COMPLETE.json').exists():basins[cid]=dict(np.load(a.root/'basin'/cid/'basin.npz'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':11,'pdf.fonttype':42})
    fig,axs=plt.subplots(3,3,figsize=(13,12),layout='constrained')
    for c,ax in zip(cases,axs.flat):
        if c['id'] not in surf:ax.set_title(title(c)+' (pending)');continue
        s=surf[c['id']];zidx=np.argmin(abs(s['third_mean']-c['centers'][2]));z=s['cube'][zidx]
        im=ax.pcolormesh(s['axis'],s['axis'],np.log1p(np.maximum(z,0)),vmin=0,vmax=np.log1p(100),cmap='viridis',shading='auto',rasterized=True)
        ax.plot(c['centers'][0],c['centers'][1],marker='*',color='white',mec='black',ms=12)
        ax.set_title(title(c)+rf' | $\mu_3={c["centers"][2]:g}$');ax.set_xlabel(r'$\mu_1$');ax.set_ylabel(r'$\mu_2$');ax.set_aspect('equal')
    cbar(fig,im,axs);fig.suptitle('Forward-KL loss sections | Three fitted means, equal fixed weights, fixed sigma = 0.5',fontsize=13)
    save(fig,a.out,'loss_sections_all')
    cmap=ListedColormap(COLORS);norm=BoundaryNorm(np.arange(-.5,4.5),4)
    fig,axs=plt.subplots(2,2,figsize=(11,10))
    basin_counts={}
    for cid,ax in zip(cfg['basin_cases'],axs.flat):
        c=next(x for x in cases if x['id']==cid)
        if cid not in basins:ax.set_title(title(c)+' (pending)');continue
        b=basins[cid];z=np.full((len(b['axis']),len(b['axis'])),np.nan);z[b['jj'],b['ii']]=b['label']
        ax.pcolormesh(b['axis'],b['axis'],np.ma.masked_invalid(z),cmap=cmap,norm=norm,shading='auto');ax.set_aspect('equal')
        ax.set_xlabel(r'Initial $\mu_1$');ax.set_ylabel(r'Initial $\mu_2$');ax.set_title(title(c)+rf' | Initial $\mu_3={c["centers"][2]:g}$')
        basin_counts[cid]=[int((b['label']==i).sum()) for i in range(4)]
        ax.text(.96,.04,' / '.join(str(v) for v in basin_counts[cid])+'\nGlobal / bad / saddle / unresolved',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
    fig.suptitle('Initialization outcomes | All three means are free during gradient descent',fontsize=13)
    legend(fig);fig.tight_layout(rect=(0,.055,1,.965));save(fig,a.out,'basin_maps')
    cid=cfg['loggap_case'];gapfile=a.root/'loggap'/cid/'basin.npz';gap_counts=None
    if (a.root/'loggap'/cid/'COMPLETE.json').exists():
        b=dict(np.load(gapfile));z=b['label'].reshape(len(b['log10_gap']),len(b['low_mean']));fig,ax=plt.subplots(figsize=(8.3,5.5))
        ax.pcolormesh(b['low_mean'],b['log10_gap'],z,cmap=cmap,norm=norm,shading='auto');ax.set_xlabel(r'Initial $\mu_1=a$');ax.set_ylabel(r'$\log_{10}\delta$');ax.set_title(r'Thin attraction regions inside $[5,6]^3$'+'\n'+r'Initial means $(a,a+\delta,6)$; target $(-1.5,1.5,6)$')
        legend(fig);fig.tight_layout(rect=(0,.06,1,1));save(fig,a.out,'near_tie_basins');gap_counts=[int((b['label']==i).sum()) for i in range(4)]
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for cid,ax in zip(cfg['basin_cases'],axs.flat):
        c=next(x for x in cases if x['id']==cid)
        if cid not in surf:continue
        s=surf[cid];im=ax.pcolormesh(s['u'],s['v'],np.log1p(np.maximum(s['plane_kl'],0)),vmin=0,vmax=np.log1p(100),cmap='viridis',shading='auto',rasterized=True)
        ax.plot([0,1],[0,0],'w--',lw=1);ax.plot(1,0,'*',color='white',mec='black',ms=13)
        ax.plot(0,0,'o',color='#e74c3c' if c['anchor_is_bad_minimum'] else '#bbbbbb',mec='white',ms=7)
        ax.set_xlabel('u: reference (0) to true means (1)');ax.set_ylabel('v: transverse displacement');ax.set_title(title(c)+('\nReference = bad minimum' if c['anchor_is_bad_minimum'] else '\nReference = initial configuration'))
    fig.suptitle(r'A plane in $(\mu_1,\mu_2,\mu_3)$ containing both reference and global solution',fontsize=13);cbar(fig,im,axs);save(fig,a.out,'connecting_planes')
    if 'R1.5_D6' in surf:
        s=surf['R1.5_D6'];fig,axs=plt.subplots(1,2,figsize=(12,4.5))
        ax=axs[0];delta=s['zoom_kl']-s['anchor_kl'];xx,yy=np.meshgrid(s['zoom_u'],s['zoom_v'])
        im=ax.pcolormesh(xx,yy,delta,cmap='RdBu_r',norm=SymLogNorm(linthresh=1e-4,vmin=-.1,vmax=.1),shading='auto',rasterized=True)
        levels=[0,.0001,.001,.005,.01,.03];cs=ax.contour(xx,yy,delta,levels=levels,colors='black',linewidths=.7);ax.clabel(cs,fmt='%g',fontsize=8)
        ax.plot(0,0,'ko',ms=5);ax.set_title('A local well around the bad minimum');ax.set_xlabel('u');ax.set_ylabel('v');fig.colorbar(im,ax=ax,label='KL minus KL at bad minimum')
        ax=axs[1];mask=s['path_t']<=.4;ax.plot(s['path_t'][mask],s['path_kl'][mask],color='#1f77b4',lw=2);ax.axhline(s['anchor_kl'],color='gray',ls='--',lw=1)
        imax=int(np.argmax(s['path_kl']));bar=float(s['path_kl'][imax]-s['anchor_kl']);ax.plot(s['path_t'][imax],s['path_kl'][imax],'ro',ms=5)
        ax.set_title(f'Straight-path barrier: {bar:.5f} KL');ax.set_xlabel('u along the line to the global solution');ax.set_ylabel('Forward KL');ax.grid(alpha=.2)
        fig.tight_layout();save(fig,a.out,'local_well_and_barrier')
        fig=plt.figure(figsize=(10,6));ax=fig.add_subplot(111,projection='3d');xx,yy=np.meshgrid(s['u'],s['v'])
        ax.plot_surface(xx,yy,s['plane_kl'],cmap='viridis',rstride=3,cstride=3,linewidth=0,alpha=.9,rasterized=True)
        ax.plot(s['path_t'],np.zeros(len(s['path_t'])),s['path_kl'],color='black',lw=2);ax.scatter([0,1],[0,0],[s['anchor_kl'],0],c=['#d55e00','#14876b'],s=50,depthshade=False)
        ax.set_xlabel('u');ax.set_ylabel('v');ax.set_zlabel('Forward KL');ax.set_title('Target (-1.5,1.5,6): a 2D parameter plane, shown as a loss surface');ax.view_init(elev=28,azim=-60);fig.tight_layout();save(fig,a.out,'loss_surface_3d')
    # Pack all conditional sections for an offline, dependency-free viewer.
    packed=[]
    for c in cases:
        if c['id'] not in surf:continue
        s=surf[c['id']];packed.append(dict(id=c['id'],centers=c['centers'],axis=s['axis'].tolist(),slices=s['third_mean'].tolist(),values=np.round(s['cube'],7).tolist(),anchor=s['anchor'].tolist(),isBad=c['anchor_is_bad_minimum']))
    colors=(plt.get_cmap('viridis')(np.linspace(0,1,256))[:,:3]*255).astype(int).tolist()
    rows='\n'.join('| '+title(next(c for c in cases if c['id']==cid))+' | '+' | '.join(map(str,counts))+' |' for cid,counts in basin_counts.items())
    barriers='\n'.join(f"| {title(c)} | {metrics[c['id']]['anchor_kl']:.6f} | {metrics[c['id']]['straight_line_barrier']:.6f} | {'verified bad point' if c['anchor_is_bad_minimum'] else 'initial reference'} |" for c in cases if c['id'] in metrics)
    total_surfaces=len(surf);total_basins=len(basins);total_gd=sum(sum(v) for v in basin_counts.values())+(sum(gap_counts) if gap_counts else 0)
    report=REPORT.replace('SURF_COUNT',str(total_surfaces)).replace('BASIN_COUNT',str(total_basins)).replace('TRAJ_COUNT',str(total_gd)).replace('BASIN_ROWS',rows).replace('BARRIER_ROWS',barriers).replace('ANALYSIS_COMMIT',a.analysis_commit)
    if gap_counts:report=report.replace('GAP_COUNTS',' / '.join(map(str,gap_counts)))
    else:report=report.replace('GAP_COUNTS','pending')
    (a.out/'report.md').write_text(report)
    image_names=['basin_maps','near_tie_basins','loss_sections_all','connecting_planes','local_well_and_barrier','loss_surface_3d']
    figures=''.join('<figure><img src="data:image/png;base64,'+base64.b64encode((a.out/(n+'.png')).read_bytes()).decode()+'"><figcaption>'+n.replace('_',' ')+'</figcaption></figure>' for n in image_names if (a.out/(n+'.png')).exists())
    template=Path(__file__).with_name('viewer.html').read_text();template=template.replace('__DATA__',json.dumps(packed,separators=(',',':'))).replace('__COLORS__',json.dumps(colors)).replace('__FIGURES__',figures).replace('__ROWS__',''.join('<tr><td>'+cid+'</td>'+''.join('<td>'+str(x)+'</td>' for x in counts)+'</tr>' for cid,counts in basin_counts.items()))
    (a.out/'report.html').write_text(template)
    summary=dict(surfaces=total_surfaces,basins=total_basins,gd_trajectories=total_gd,basin_counts=basin_counts,near_tie_counts=gap_counts,barriers=metrics,analysis_commit=a.analysis_commit)
    (a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

REPORT=r'''# K=3 mean-only Forward KL: loss landscape와 초기화별 수렴 지도

세 mean (mu1,mu2,mu3)은 3차원 파라미터이고 loss까지 포함하면 네 값이므로, 일반적인 3D surface 하나로 전체를 표시할 수 없다. 여기서는 **조건부 단면**, **두 해를 포함하는 파라미터 평면**, **실제 GD의 초기화별 수렴 결과**를 함께 본다. [인터랙티브 단면 보기](report.html)는 인터넷 없이 열린다. 환경을 선택하고 mu3 슬라이더를 움직일 수 있다.

## 실험 설정

\[
p^*(x)=\frac13\sum_{k=1}^3\mathcal N(x;c_k,0.5^2),\quad
q_\mu(x)=\frac13\sum_{i=1}^3\mathcal N(x;\mu_i,0.5^2).
\]
\[
F(\mu)=D_{\mathrm{KL}}(p^*\Vert q_\mu),\qquad
\mu\leftarrow\mu-0.01\nabla F(\mu).
\]

학습하는 것은 mean 세 개뿐이다. Weight=1/3과 sigma=0.5는 고정하고, 실수 전체의 ordinary Gaussian을 쓴다. Target expectation을 float64 Simpson 적분 4,097점으로 계산하고 8,193점과 비교했다. SNIS/actor/proposal/SGD noise가 개입하지 않는다. 수렴 지도에서는 gradient norm<=1e-7까지, 최대 100K updates를 수행했다. 종료점의 gradient와 Hessian은 3차원 전체에 대해 검사했다.

현재 loss landscape **SURF_COUNT/9개**, 일반 수렴 지도 **BASIN_COUNT/4개**가 포함되어 있고, 총 **TRAJ_COUNT개 초기값**의 결과를 표시한다. 임의로 실패한 seed만 선택한 것이 아니라 정해둔 모든 격자점을 보존했다.

## 1. 어느 초기화에서 실패하는가?

![초기화별 수렴](basin_maps.png)

각 panel의 x,y축은 **초기** mu1,mu2다. 초기 mu3만 오른쪽 target mean으로 두었다. **학습 중에는 세 mean 모두 자유롭게 업데이트했다.** 동일 weight의 component permutation을 중복 계산하지 않기 위해 mu1<=mu2<=mu3인 17-point 격자의 153개 배치를 검사했다. 비어 있는 삼각형은 검사하지 않은 중복 영역이다.

- 초록: target을 복구한 global optimum (KL<1e-6).
- 주황: gradient가 충분히 작고 Hessian 고유값이 모두 양수인 bad local minimum.
- 보라: gradient는 작지만 Hessian에 음의 고유값이 있는 saddle.
- 회색: 시간 내 미수렴 또는 곡률이 너무 작아 분류하지 않은 경우.

| Target | Global | Bad minimum | Saddle | Unresolved |
|---|---:|---:|---:|---:|
BASIN_ROWS

이 격자에서의 개수는 특정 초기화 분포 아래의 실패 확률이 아니다. 또한 지도에서 같은 색으로 보이는 유한 격자 사이의 모든 점이 같은 결과를 갖는다는 보장도 아니다.

## 2. 보통 격자가 놓치는 얇은 성공 영역

![거의 겹치는 초기 mean](near_tie_basins.png)

Target=(-1.5,1.5,6), 초기 mean=(a,a+delta,6)이다. 가로축 a는 [5,5.75], 세로축은 두 mean 사이 간격의 log10이다. 모든 초기 mean은 [5,6] 안에 있다. 11x17=187개 초기값을 검사했다. Global / bad / saddle / unresolved: **GAP_COUNTS**.

이 그림은 "[5,6]에서 random seed가 전부 실패했으니 그 구간 전체가 실패 영역이다"라는 결론을 검토하기 위한 것이다. 정확히 같은 mean의 대칭 saddle뿐 아니라, 간격이 0이 아닌 초기값에서의 탈출도 구분해서 표시한다. 작은 간격에서는 유한 정밀도 영향에도 주의해야 한다.

## 3. 세 번째 mean을 고정한 loss 단면

![9개 target의 조건부 loss](loss_sections_all.png)

mu3를 각 target의 오른쪽 중심에 고정한 그림이다. 축은 mu1,mu2, 색은 Forward KL이며 낮을수록 좋다. 색에는 log(1+KL) 변환을 사용하고 colorbar 숫자는 원래 KL 단위다. KL>=100은 같은 최상위 색으로 표시한다. 흰 별은 정답 mean 배치다.

**단면에서 보이는 minimum이 3차원 전체의 local minimum인 것은 아니다.** 고정했던 mu3 방향으로 탈출할 수 있기 때문이다. 전체 Hessian 검사와 실제 세 mean을 모두 학습한 수렴 지도를 함께 봐야 한다. 동일 mixture를 나타내는 component 순열 때문에 대칭으로 반복되는 골짜기가 생긴다.

## 4. 정답과 나쁜 minimum을 동시에 지나는 평면

![두 해를 지나는 평면](connecting_planes.png)

\[
\mu(u,v)=b+u(c-b)+v d,\qquad d^\top(c-b)=0,\quad\|d\|=1.
\]

c는 true mean vector, b는 앞서 확인한 bad minimum이다. Bad minimum을 찾지 못했던 target은 b를 초기 배치로 사용하며 **initial configuration**이라고 따로 표시했다. (u,v)=(0,0)은 b, (1,0)은 정답이다. d는 (0,-1,1)을 c-b에 수직으로 투영한 방향이다. 따라서 이 평면에서도 세 mean이 함께 움직인다.

![국소 골짜기와 직선 경로의 장벽](local_well_and_barrier.png)

대표 target (-1.5,1.5,6)의 왼쪽 그림은 bad minimum 대비 KL 증가량을 확대했다. 오른쪽은 정답으로 향하는 **직선 경로**의 초반을 확대했다. 경로에서 처음에 loss가 올라가는 것은 순수한 descent가 바로 정답 방향으로 이동하지 못하는 이유를 보여준다. 이 장벽 값은 모든 탈출 경로 중 최소 장벽을 계산한 값이 아니다.

| Target | KL at reference | 직선 경로의 최대 상승량 | Reference 종류 |
|---|---:|---:|---|
BARRIER_ROWS

![3D loss surface](loss_surface_3d.png)

이 3D surface의 축은 (u,v,KL)이다. 세 mean 자체를 세 축으로 놓고 loss를 높이로 놓은 그림은 아니다. 두 해를 포함하는 2D 파라미터 평면 위의 loss를 높이로 표현한 것이다.

## 해석의 범위

이번 그림은 finite 3-GMM의 population Forward-KL optimization geometry를 관찰한다. Semi-implicit actor의 성공 여부를 새로 실험한 것은 아니다. 특정 초기화 영역 전체가 bad minimum으로 수렴한다고 보장하려면, 유한 격자 그림에 추가로 경계·불변성 또는 contraction 조건을 검증해야 한다.

## 재현·보관

- Numerical source: `026c2944868204d775ed8bdce7cfef0f71e4d59c`, 실행 전에 heejoon에 commit/push.
- Figure/report source: `ANALYSIS_COMMIT`.
- Source/config: `experiments/gmm3_landscapes`.
- login4: `/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/gmm3_landscapes_20260925`.
- dildata: `/data1/heejoonorm/OptiQ/studies/20260925_gmm3_landscapes`.
- Original surfaces, trajectories, termination gradients/Hessians, checkpoints and job IDs are preserved. PNG와 PDF를 함께 제공한다.
'''
if __name__=='__main__':main()
