"""CPU-only partial/final report; all actor density panels are sample histograms."""
import argparse,base64,html,json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;out=root/'report';out.mkdir(exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
    runs=sorted((root/'runs').glob('N*_s*'));rows=[]
    for r in runs:
        histories=[json.loads(l) for l in (r/'history.jsonl').read_text().splitlines()] if (r/'history.jsonl').exists() else []
        if histories:rows.append((r,histories[-1]))
    lines=['# Direct GMM: 3-mode gradient 간섭과 latent 분업','',f'완료 {sum((r/"COMPLETE.json").exists() for r in runs)}/8. 아래는 현재 저장된 결과이며, 학습 중 결과를 최종 성능으로 해석하지 않는다.','',
        '## 실험 설정과 그림 읽는 법','',
        '기존 0917 toy의 세 mode를 그대로 사용한다. 중심 −0.6, 0, 0.6 / 폭0.1 / 동일 질량, action [−1,1], Q=0.25 log f, temperature0.25. Q는 20K updates 동안 고정한다.','',
        '| 항목 | 값 |','|---|---|','| 크기 / seeds | N=64,2048 / M=4096 고정 / seed0–3 |','| Actor | 기존 v5 256×2 GELU, latent1D IID Normal, 초기 sigma0.5, log sigma [−5,1] |','| Update | 원본 Direct GMM marginal NLL, Adam3e−4, batch1, clipping/OT/gradient 수정 없음 |','| Proposal | 원본 conditional squashed Gaussian mixture, teacher-only sigma floor0.05, 정확한 mixture density correction |','| Density | 새 latent·noise로 뽑은 32,768개 action의 histogram, 256 bins, smoothing 없음 |','',
        'Mode는 target 중심 사이 경계 −0.3,0.3으로 나눈 basin이다. Mode별 NLL은 원래 w를 그대로 분할하며, 각 mode 질량으로 재정규화하지 않는다. 세 gradient의 합은 원래 gradient다. Histogram TV=1/2 Σ_bin|actor mass−target mass|.','',
        '고정 latent 2048개의 conditional Gaussian이 특정 basin에 확률0.8 이상을 두면 그 mode의 specialist로 분류한다. 나머지는 broad/ambiguous다. 이 확률만 Gaussian CDF로 계산하며 actor density plot은 histogram이다. Tanh(mu)는 대표 위치이며 action 평균이 아니다.','',
        '진단 시 actor·Adam 상태를 복사해 full Adam, zero-gradient momentum control, mode별 Adam, full Adam과 같은 parameter 이동 norm의 mode별 SGD를 각각 한 번 적용한다. 모두 폐기하며 실제 학습 및 RNG는 바꾸지 않는다. Mode별 Adam에서 다른 mode가 악화돼도 이것만으로 실제 full update의 실패라고 결론내리지 않는다.','',
        'NLL 변화는 같은 teacher/latent와, 별도 정답 quadrature(1024 bins)·고정 평가 latent에서 모두 측정한다. 후자는 샘플 teacher 과적합과 새 latent 일반화를 구분하기 위한 보조 likelihood 평가이다.','',
        '| Run | Step | Histogram TV | Specialist fraction | 완료 |','|---|---:|---:|---:|---|']
    for r,h in rows:lines.append(f'| {r.name} | {h["step"]} | {h["histogram_tv"]:.4f} | {h["specialist_fraction"]:.3f} | {(r/"COMPLETE.json").exists()} |')
    x=np.linspace(-1,1,2001);c=np.array([-.6,0,.6]);norm=(ndtr((1-c)/.1)-ndtr((-1-c)/.1)).mean()
    density=np.exp(-.5*((x[:,None]-c)/.1)**2).mean(1)/(.1*np.sqrt(2*np.pi)*norm)
    figure,axes=plt.subplots(1,2,figsize=(13,4),sharex=True)
    for ax,n in zip(axes,[64,2048]):
        ax.plot(x,density,'k--',label='Exact target',lw=2)
        for r,h in rows:
            if not r.name.startswith(f'N{n}_'):continue
            d=np.load(r/f'eval_{h["step"]:06d}.npz');ax.stairs(d['histogram']/np.diff(d['edges']),d['edges'],label=f'{r.name.split("_")[-1]}, step {h["step"]}',alpha=.7)
        ax.set(title=f'N={n}, M=4096',xlabel='Action',ylabel='Density');ax.legend(fontsize=8)
    figure.tight_layout();figure.savefig(fig/'densities.png',dpi=160);plt.close(figure)
    lines+=['','## 현재 density','', 'Seed별 최신 step을 범례에 표시한다. 서로 다른 step의 결과를 평균내지 않는다.','', '![실제 action histogram](figures/densities.png)']
    for n in [64,2048]:
        r=root/'runs'/f'N{n}_M4096_s0';paths=sorted(r.glob('diagnostic_*.npz'))
        if not paths:continue
        d=np.load(paths[-1]);step=int(paths[-1].stem.split('_')[1]);tag=f'N{n}_s0_{step}'
        figure,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,name in zip(axes,['gradient_cosine','cosine_trunk','cosine_mu_head']):
            im=ax.imshow(d[name],vmin=-1,vmax=1,cmap='coolwarm');ax.set(title=name,xticks=range(3),yticks=range(3),xticklabels=['L','C','R'],yticklabels=['L','C','R'])
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{d[name][i,j]:.2f}',ha='center',va='center',fontsize=9)
        figure.colorbar(im,ax=axes,shrink=.7);figure.suptitle(f'N={n}, seed0, step={step}');figure.savefig(fig/f'{tag}_cosine.png',dpi=150,bbox_inches='tight');plt.close(figure)
        figure,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,name,title in zip(axes,['delta_mode_nll','delta_reference_nll','cross_mode_own_mass_change'],['Same-teacher NLL change','Held-out reference NLL change','Old specialist own-basin mass change']):
            mat=d[name][2:5,:3];lim=max(np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 0,1e-8)
            im=ax.imshow(np.ma.masked_invalid(mat),vmin=-lim,vmax=lim,cmap='coolwarm');figure.colorbar(im,ax=ax,shrink=.7)
            ax.set(title=title,xticks=range(3),yticks=range(3),xticklabels=['L','C','R'],yticklabels=['L update','C update','R update'])
        figure.tight_layout();figure.savefig(fig/f'{tag}_effect.png',dpi=150);plt.close(figure)
        figure,axes=plt.subplots(1,3,figsize=(14,4));order=np.argsort(d['z'][:,0]);prob=d['basin_prob'][order]
        axes[0].imshow(prob.T,aspect='auto',origin='lower',vmin=0,vmax=1);axes[0].set(title='Fixed latent conditional basin probability',xlabel='Fixed latent (sorted by z)',yticks=range(3),yticklabels=['L','C','R'])
        axes[1].plot(d['z'][order,0],np.tanh(d['mu'][order,0]));axes[1].set(title='Representative tanh(mu)',xlabel='Fixed z',ylabel='Action')
        axes[2].plot(d['z'][order,0],np.exp(d['log_sigma'][order,0]));axes[2].set(title='Conditional pre-tanh sigma',xlabel='Fixed z')
        figure.tight_layout();figure.savefig(fig/f'{tag}_latents.png',dpi=150);plt.close(figure)
        hist=[np.load(p) for p in paths];figure,axes=plt.subplots(1,2,figsize=(12,4));steps=[int(p.stem.split('_')[1]) for p in paths]
        ix=order[np.linspace(0,len(order)-1,64,dtype=int)]
        axes[0].plot(steps,np.stack([np.tanh(h['mu'][ix,0]) for h in hist]),alpha=.35,lw=.7);axes[0].set(title='Same fixed latent trajectories',xlabel='Training update',ylabel='tanh(mu)')
        axes[1].plot(steps,np.stack([h['specialist_counts']/2048 for h in hist]),label=['Left','Center','Right','Ambiguous']);axes[1].legend();axes[1].set(title='Specialist fractions',xlabel='Training update')
        figure.tight_layout();figure.savefig(fig/f'{tag}_trajectory.png',dpi=150);plt.close(figure)
        lines+=['',f'## N={n}, seed0, step={step} 진단','',
            'Cosine은 parameter gradient 간의 각도다. 음수는 해당 단독 SGD 방향이 다른 mode loss를 1차 근사에서 증가시킴을 뜻한다. 실제 Adam의 전체 update 결과와 구별한다.','',
            f'![Gradient cosine](figures/{tag}_cosine.png)','',
            '아래 행은 update에 사용한 mode, 열은 영향을 받은 mode다. NLL 변화의 양수는 악화, own-basin mass 변화의 음수는 기존 specialist가 담당 mode에서 이탈함을 뜻한다. 빈칸은 해당 specialist가 없어서 측정 불가한 경우다.','',
            f'![한 mode update의 다른 mode 영향](figures/{tag}_effect.png)','',
            f'![고정 latent 분업](figures/{tag}_latents.png)','',f'![동일 latent 시간 추적](figures/{tag}_trajectory.png)','',
            f'실제 full Adam의 같은-teacher mode NLL 변화: `{d["delta_mode_nll"][0].tolist()}`. 정답 reference 변화: `{d["delta_reference_nll"][0].tolist()}`.']
    lines+=['','## 해석 시 확인할 것','',
        '- Teacher부터 mode 질량이 충분한가? Teacher ESS/wmax 및 원본 가중 후보를 함께 확인한다.',
        '- 음의 cosine이 실제 full Adam에서도 mode loss 악화로 이어지는가?',
        '- 이미 specialist가 있는가, 아니면 대부분 broad/ambiguous인가?',
        '- 다른 mode update 이후 기존 specialist의 위치·sigma·담당 확률이 함께 바뀌는가?',
        '- Momentum control과 norm-matched SGD에서도 같은 현상이 있는가?',
        '- Seed0 그림만으로 결론내리지 않고 4 seed 및 같은 training step끼리 비교한다.',
        '', '원본 진단 NPZ에는 원본 teacher 후보/weight/Q/log q, training z, 2048 fixed z, mean/sigma, responsibility mode mass, 전체·trunk·mu/sigma head gradient 행렬, 8개 branch의 실제/1차 NLL 변화와 출력 변화가 저장된다.']
    md='\n'.join(lines)+'\n';(out/'report.md').write_text(md)
    try:
        import markdown
        body=markdown.markdown(md,extensions=['tables','fenced_code'])
    except ImportError:body='<pre>'+html.escape(md)+'</pre>'
    # Embed pictures for self-contained offline sharing.
    for pth in fig.glob('*.png'):
        encoded='data:image/png;base64,'+base64.b64encode(pth.read_bytes()).decode()
        body=body.replace('figures/'+pth.name,encoded)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;line-height:1.65}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px}code{overflow-wrap:anywhere}</style>'+body)
    print(f'Report: {len(rows)} runs with data, {sum((r/"COMPLETE.json").exists() for r in runs)}/8 complete')

if __name__=='__main__':main()
