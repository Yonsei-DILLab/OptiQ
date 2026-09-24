"""Same-sample bin-resolution audit and unadorned Matplotlib figure."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def target_cdf(x, centers, std, bound):
    norm=np.mean(ndtr((bound-centers)/std)-ndtr((-bound-centers)/std))
    return np.mean(ndtr((np.asarray(x)[...,None]-centers)/std)-ndtr((-bound-centers)/std),axis=-1)/norm


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    bin_counts=[512,1024,2048,4096];raw={};input_metadata={};prior={}
    for method in ['forward','reverse']:
        for seed in range(4):
            name=f'{method}_s{seed}';folder=args.results/name;path=folder/'actions_1048576.npz'
            raw[name]=np.load(path)['actions'];assert raw[name].shape==(2**20,)
            assert np.isfinite(raw[name]).all() and (np.abs(raw[name])<=10).all()
            run=json.loads((folder/'RUN.json').read_text());prior[name]=json.loads((folder/'METRICS.json').read_text())
            cfg=run['parent_config']
            input_metadata[name]={'actions_sha256':sha(path),'checkpoint_sha256':run['checkpoint_sha256'],
                                  'sampling_commit':run['analysis_commit'],'training_commit':run['parent_commit']}
    centers=np.array(cfg['target_centers']);std=cfg['target_width'];bound=cfg['action_bound']
    bounds=np.r_[-bound,(centers[:-1]+centers[1:])/2,bound]
    target_basin=np.diff(target_cdf(bounds,centers,std,bound))
    basins={name:float(.5*np.abs(np.histogram(a,bounds)[0]/len(a)-target_basin).sum()) for name,a in raw.items()}
    n=2**20;ranks=(np.arange(n,dtype=np.float64)+.5)/n
    lower=np.full(n,-bound);upper=np.full(n,bound)
    for _ in range(42):
        mid=(lower+upper)/2;less=target_cdf(mid,centers,std,bound)<ranks
        lower=np.where(less,mid,lower);upper=np.where(less,upper,mid)
    quantiles=(lower+upper)/2
    inverse_residual=float(np.max(np.abs(target_cdf(quantiles,centers,std,bound)-ranks)))
    assert inverse_residual<1e-10
    w1={name:float(np.mean(np.abs(np.sort(a).astype(np.float64)-quantiles))) for name,a in raw.items()}
    null_rng_w=np.random.default_rng(202609243);null_w1=[]
    lo=ndtr((-bound-centers)/std);hi=ndtr((bound-centers)/std);component_mass=hi-lo
    for _ in range(16):
        comp=null_rng_w.choice(len(centers),n,p=component_mass/component_mass.sum())
        u=lo[comp]+component_mass[comp]*null_rng_w.random(n)
        action=centers[comp]+std*ndtri(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps))
        null_w1.append(float(np.mean(np.abs(np.sort(action)-quantiles))))
    wasserstein={'per_seed':w1,'perfect_sampler_values':null_w1,
                 'perfect_sampler_mean':float(np.mean(null_w1)),
                 'perfect_sampler_q05':float(np.quantile(null_w1,.05)),
                 'perfect_sampler_q95':float(np.quantile(null_w1,.95)),
                 'inverse_cdf_max_residual':inverse_residual,'midpoint_quadrature_bound':2*bound/n}
    for method in ['forward','reverse']:
        values=np.array([w1[f'{method}_s{s}'] for s in range(4)])
        wasserstein[method]={'mean':float(values.mean()),'SD':float(values.std(ddof=1))}
    all_metrics={};matrices={};null_counts={};null_rng=np.random.default_rng(202609242)
    for bins in bin_counts:
        edges=np.linspace(-bound,bound,bins+1)
        target=np.diff(target_cdf(edges,centers,std,bound))
        assert np.min(target)>=-1e-15
        target=np.maximum(target,0);assert abs(target.sum()-1)<1e-12
        hist={};per_seed={}
        for name,a in raw.items():
            counts=np.histogram(a,edges)[0];assert counts.sum()==2**20
            hist[name]=counts/(2**20)
            tv=float(.5*np.abs(hist[name]-target).sum());per_seed[name]=tv
            if bins==512:assert abs(tv-prior[name]['histogram_TV'])<1e-12
            else:assert tv>=all_metrics[str(bins//2)]['per_seed'][name]-1e-12
        null=null_rng.multinomial(2**20,target/target.sum(),size=128)
        null_tv=.5*np.abs(null/(2**20)-target).sum(1);null_counts[bins]=null_tv
        metrics={'per_seed':per_seed,'perfect_sampler_TV_mean':float(null_tv.mean()),
                 'perfect_sampler_TV_q05':float(np.quantile(null_tv,.05)),
                 'perfect_sampler_TV_q95':float(np.quantile(null_tv,.95))}
        for method in ['forward','reverse']:
            tv=np.array([per_seed[f'{method}_s{s}'] for s in range(4)])
            mm=np.stack([hist[f'{method}_s{s}'] for s in range(4)])
            metrics[method]={'TV_mean':float(tv.mean()),'TV_SD':float(tv.std(ddof=1)),
                             'TV_seed_mean_histogram':float(.5*np.abs(mm.mean(0)-target).sum())}
        all_metrics[str(bins)]=metrics;matrices[bins]=(edges,target,hist)
    payload={'bins':bin_counts,'sample_count_per_seed':2**20,'seeds':[0,1,2,3],'rebin_commit':args.commit,
             'inputs':input_metadata,'metrics':all_metrics,'basin_TV':basins,
             'perfect_sampler_repeats':128,'perfect_sampler_rng_seed':202609242,'wasserstein1':wasserstein}
    (args.output/'SUMMARY.json').write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(args.output/'perfect_sampler_tv.npz',**{f'bins_{b}':v for b,v in null_counts.items()})
    for bins,(edges,target,hist) in matrices.items():
        np.savez_compressed(args.output/f'histograms_{bins}.npz',edges=edges,target_mass=target,**hist)

    plt.style.use('default')
    plt.rcParams.update({'font.size':12,'axes.titlesize':16,'axes.labelsize':13,
                         'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    edges,_,hist=matrices[4096];width=np.diff(edges)
    x=np.linspace(-bound,bound,8193)
    norm=np.mean(ndtr((bound-centers)/std)-ndtr((-bound-centers)/std))
    target_pdf=np.exp(-.5*((x[:,None]-centers)/std)**2).mean(1)/(std*np.sqrt(2*np.pi)*norm)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharex=True,sharey=True)
    for k,(ax,method) in enumerate(zip(axes,['forward','reverse'])):
        density=np.stack([hist[f'{method}_s{s}']/width for s in range(4)])
        mean=density.mean(0);sd=density.std(0,ddof=1)
        ax.stairs(mean,edges,color=f'C{k}',lw=1.2,label='Policy (mean)')
        ax.fill_between(edges,np.r_[np.maximum(mean-sd,0),max(mean[-1]-sd[-1],0)],np.r_[mean+sd,mean[-1]+sd[-1]],
                        color=f'C{k}',alpha=.2,step='post',linewidth=0,label=r'$\pm$1 SD across seeds')
        ax.plot(x,target_pdf,'k--',lw=1.8,label='Target')
        ax.set_title(f'{method.capitalize()} KL')
        tv=all_metrics['4096'][method]
        ax.text(.035,.93,f'Mean TV = {tv["TV_mean"]:.4f}\nMean W1 = {wasserstein[method]["mean"]:.4f}',transform=ax.transAxes,ha='left',va='top')
        ax.set(xlim=(-10,10),ylim=(0,.9),xlabel='Action')
        ax.legend(loc='upper right',frameon=True)
        ax.set_xticks([-10,-5,0,5,10])
    axes[0].set_ylabel('Probability density')
    fig.tight_layout(pad=1.4,w_pad=2.)
    for ext in ['png','pdf','svg']:fig.savefig(args.output/f'final_density_4096.{ext}',dpi=300)
    plt.close(fig)
    caption=r'''**Forward versus reverse KL on a three-mode target.** The learned density in each panel is the mean histogram across four seeds after 100,000 updates. Each seed contributes the same $2^{20}$ saved action samples; 4,096 equal-width bins cover $[-10,10]$. Shading denotes one sample standard deviation across seeds, and the dashed line is the exact target density. No KDE smoothing is applied. TV is calculated against exact target bin probabilities; one-dimensional Wasserstein-1 (W1) is computed independently of the histogram using sorted actions and target quantiles. Annotations give the mean of per-seed metrics, not metrics of the averaged density. Both methods use $N=M=128$; reverse KL uses $L=2^{20}$. The target has equally weighted means $(-4.25,0,4.25)$ and standard deviation $0.5$. This is the previously selected illustrative toy setting.'''
    (args.output/'caption.md').write_text(caption+'\n')
    lines=['# 最終 KL 비교: 표준 Matplotlib 스타일과 4096-bin TV'.replace('最終','최종'),'',
           '![4096-bin seed 평균 histogram](final_density_4096.png)','',
           '기존의 실제 action 표본을 그대로 사용했다. actor당 2²⁰개, 네 seed를 같은 비중으로 평균했다. Matplotlib 기본 팔레트·축·글꼴을 사용하고 캔버스를 12×4.6인치로 확대했다. 음영은 seed 간 ±1 SD이고, KDE smoothing은 없다.','',
           '## Bin 수에 따른 TV','',
           '|Bin 수|Forward 평균 TV|Reverse 평균 TV|완벽한 target sampler의 표본 TV 평균|','|---:|---:|---:|---:|']
    for bins in bin_counts:
        m=all_metrics[str(bins)];lines.append(f'|{bins}|{m["forward"]["TV_mean"]:.8f}|{m["reverse"]["TV_mean"]:.8f}|{m["perfect_sampler_TV_mean"]:.8f}|')
    lines+=['','## 왜 bin을 늘려도 TV가 줄지 않는가','',
            '같은 표본에서 한 bin을 여러 sub-bin으로 나누면, 이전에는 상쇄되던 양·음의 질량 오차가 드러난다. 다음 삼각부등식 때문에 이번처럼 경계가 서로 포함되는 bin 분할에서는 histogram TV가 감소하지 않는다. 실제로 모든 seed에서 이를 확인했다.','',
            r'$$\left|\sum_{j\in I}(\widehat p_j-p_j^\star)\right|\le\sum_{j\in I}|\widehat p_j-p_j^\star|.$$', '',
            '**512개 bin이 TV를 과도하게 높여서 0.04가 나온 것은 아니다.** 세밀한 bin은 좁은 구간의 차이를 더 드러내고, 각 bin의 표본 수를 줄여 sampling noise도 늘린다. 표본 수를 늘리는 것은 MC 오차를 줄이는 조치이고, bin을 늘리는 것은 공간 해상도를 높이는 조치다. 둘은 같은 효과가 아니다.','',
            '## 샘플링 잡음과 실제 분포 오차','',
            '완벽하게 target에서 샘플링하는 경우도 유한 표본의 histogram TV는 0이 아니다. 이를 확인하기 위해 exact target bin 확률로부터 128회의 multinomial count를 생성했다. 이는 target에서 2²⁰개 action을 뽑아 bin별 개수만 세는 것과 정확히 같은 확률분포다. 이 수치는 관측 TV에서 빼는 보정값이나 보편적 오차 하한이 아니다.','',
            '|Bin 수|Perfect sampler 평균|128회 반복의 5–95% 구간|','|---:|---:|---:|']
    for bins in bin_counts:
        m=all_metrics[str(bins)];lines.append(f'|{bins}|{m["perfect_sampler_TV_mean"]:.8f}|{m["perfect_sampler_TV_q05"]:.8f}–{m["perfect_sampler_TV_q95"]:.8f}|')
    f512=all_metrics['512']['forward'];f4096=all_metrics['4096']['forward']
    bmean=np.mean([basins[f'forward_s{s}'] for s in range(4)])
    lines+=['',f'Forward의 512-bin TV {f512["TV_mean"]:.5f}는 완벽한 target sampler의 표본 TV {all_metrics["512"]["perfect_sampler_TV_mean"]:.5f}보다 크다. 따라서 표본 잡음만으로 설명되지는 않는다. 반면 세 mode의 basin 질량만 비교한 평균 TV는 {bmean:.5f}이다. **세 mode의 총 질량을 잘 맞춰도, mode 안의 위치·폭·모양과 mode 사이의 작은 질량 차이가 전체 TV에 남을 수 있다.** 이 측정만으로 어느 shape 오차가 원인인지까지 분리한 것은 아니다.','',
            f'그림은 seed 평균이므로 개별 seed의 오차 일부가 상쇄된다. 4096-bin Forward 평균 histogram 자체의 TV는 {f4096["TV_seed_mean_histogram"]:.5f}지만, 보고하는 per-seed TV 평균은 {f4096["TV_mean"]:.5f}다.','',
            '## 4096-bin 최종 지표','',
            '|방법|Seed 평균 TV|Seed 간 SD|','|---|---:|---:|']
    for method in ['forward','reverse']:
        m=all_metrics['4096'][method];lines.append(f'|{method.capitalize()} KL|{m["TV_mean"]:.8f}|{m["TV_SD"]:.8f}|')
    lines+=['','|Seed|Forward TV|Reverse TV|','|---:|---:|---:|']
    for seed in range(4):lines.append(f'|{seed}|{all_metrics["4096"]["per_seed"][f"forward_s{seed}"]:.8f}|{all_metrics["4096"]["per_seed"][f"reverse_s{seed}"]:.8f}|')
    lines+=['','## Bin에 의존하지 않는 Wasserstein-1','',
            r'$$W_1(\widehat\pi,p^\star)=\int_0^1|F_{\widehat\pi}^{-1}(u)-F_{p^\star}^{-1}(u)|\,du\ \approx\ \frac1n\sum_{i=1}^{n}\left|a_{(i)}-F_{p^\star}^{-1}\!\left(\frac{i-1/2}{n}\right)\right|.$$', '',
            'action 표본을 정렬한 뒤 exact target CDF를 역으로 푼 분위수와 비교했다. 별도의 target MC 표본이나 histogram을 기준으로 사용하지 않았다. W1은 확률 질량을 옮기는 평균 거리이며, 이 실험에서는 action 좌표 단위다. TV와 수치 크기 자체를 직접 비교하는 지표는 아니다.','',
            '|방법|W1 평균|Seed 간 SD|','|---|---:|---:|']
    for method in ['forward','reverse']:
        w=wasserstein[method];lines.append(f'|{method.capitalize()} KL|{w["mean"]:.8f}|{w["SD"]:.8f}|')
    lines+=['','|Seed|Forward W1|Reverse W1|','|---:|---:|---:|']
    for seed in range(4):lines.append(f'|{seed}|{w1[f"forward_s{seed}"]:.8f}|{w1[f"reverse_s{seed}"]:.8f}|')
    lines+=['',f'Perfect target sampler 16회의 W1 평균은 {wasserstein["perfect_sampler_mean"]:.6f}, 5–95% 구간은 {wasserstein["perfect_sampler_q05"]:.6f}–{wasserstein["perfect_sampler_q95"]:.6f}였다. W1도 유한 표본에서는 0이 아니지만, bin 개수의 영향은 없다.', '',
            'Target quantile은 [-10,10]에서 42번 bisection으로 계산했다. 경험분포와 target 사이의 W1 적분에 대한 midpoint 근사의 보수적 오차 상한은 action 범위/n = 20/2²⁰ ≈ 0.0000191이다(역 CDF 부동소수점 오차 제외). 이 수치 적분 오차는 정책으로부터 유한 표본만 얻는 sampling error와 별개다.','']
    lines+=['','## 논문용 caption','',caption,'',
            '## 재현','',f'- Rebin/plot source commit: `{args.commit}`.',
            '- 원본 action 생성 source: `8b35c3ecae257846ee453e099af2432064e894f0`.',
            '- 새 actor sample 생성·재학습 없음. 원본 action SHA256을 SUMMARY.json에 기록했다.',
            '- 원본 표본의 512-bin TV가 기존 값과 1e-12 이내에서 일치함을 확인했다.',
            '- TV는 연속 density의 정확한 TV가 아닌 binned empirical TV다.',
            '- 보관: `dildata:/data1/heejoonorm/OptiQ/studies/20260924_selected_tv1m/rebin_4096/`.','']
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'metrics':all_metrics,'basin_TV':basins},indent=2))


if __name__=='__main__':main()
