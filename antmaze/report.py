"""Four-seed final report. Never substitutes partial runs for final results."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .controller import METHODS
from .evaluation import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    a=p.parse_args();root=a.root;out=root/"report";out.mkdir(exist_ok=True)
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["smoke"]: raise ValueError("Cannot report smoke as final experiment")
    fig,axes=plt.subplots(1,3,figsize=(16,4.7));stats={}
    records={}
    for method in METHODS:
        all_runs=[];per_seed=[]
        for seed in range(4):
            d=root/"runs"/f"{method}-s{seed}"
            result=json.loads((d/"result.json").read_text())
            cfg=json.loads((d/"config.json").read_text())
            assert result["completed"] and result["steps"]==1000000 and not cfg["smoke"]
            assert result["source_commit"]==manifest["source_commit"]==cfg["source_commit"]
            assert result["updates"]==1000000-cfg["warmup"]
            z=np.load(d/f"evaluations_{result['primary_mode']}.npz",allow_pickle=False)
            last=(z["timesteps"]>900000)&(z["timesteps"]<=1000000)
            assert last.sum()==20 and z["results"].shape==(201,10)
            row=dict(seed=seed)
            for metric in ("successes","results","min_distances"):
                assert np.isfinite(z[metric]).all()
                row[metric]=float(z[metric][last].mean())
            per_seed.append(row);all_runs.append({k:z[k] for k in z.files})
        records[method]=all_runs
        stats[method]=dict(seeds=per_seed,evaluation_mode="stochastic_z mu-only" if method=="optiq" else "native")
        for ax,metric,label in zip(axes,("successes","results","min_distances"),
                ("Success rate","Episode return","Minimum goal distance")):
            matrix=np.stack([r[metric].mean(axis=1) for r in all_runs])
            x=all_runs[0]["timesteps"];mean=matrix.mean(axis=0);sd=matrix.std(axis=0,ddof=1)
            ax.plot(x,mean,label=method);ax.fill_between(x,mean-sd,mean+sd,alpha=.10)
            ax.set(xlabel="Environment steps",ylabel=label);ax.grid(alpha=.2)
            vals=np.array([r[metric] for r in per_seed])
            stats[method][metric]=dict(mean=float(vals.mean()),sample_sd=float(vals.std(ddof=1)))
    axes[0].set_ylim(-.05,1.05);axes[1].set_ylim(-.05,1.05)
    axes[0].legend(ncol=2);fig.suptitle("AntMaze UMaze sparse · 4 seeds · mean ± sample SD")
    fig.tight_layout();fig.savefig(out/"learning_curves.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    x=np.arange(len(METHODS))
    for ax,metric,label in zip(axes,("successes","min_distances"),("Success rate","Minimum goal distance")):
        ax.bar(x,[stats[m][metric]["mean"] for m in METHODS],
               yerr=[stats[m][metric]["sample_sd"] for m in METHODS],capsize=4)
        for i,m in enumerate(METHODS):ax.scatter([i]*4,[r[metric] for r in stats[m]["seeds"]],color="black",s=16,zorder=3)
        ax.set_xticks(x,METHODS);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Last 100k (900k < step ≤ 1M) · seed mean ± sample SD")
    fig.tight_layout();fig.savefig(out/"last100k.png",dpi=180);plt.close(fig)
    atomic_json(out/"results.json",dict(source_commit=manifest["source_commit"],algorithms=stats))
    lines=["# AntMaze UMaze sparse 결과", "", "각 알고리즘 seed0–3, 1M 환경 스텝. 마지막100k의20평가×10episode를 seed내 평균 후 4seed 평균±표본표준편차.","",
        "| 알고리즘 | 성공률 | 최소 목표 거리 | 평가 |","|---|---:|---:|---|"]
    for m in METHODS:
        s=stats[m];v=s["successes"];d=s["min_distances"]
        lines.append(f"| {m} | {v['mean']:.3f} ± {v['sample_sd']:.3f} | {d['mean']:.3f} ± {d['sample_sd']:.3f} | {s['evaluation_mode']} |")
    lines += ["", "Sparse episodic return은 성공률과 같습니다. OptiQ는 conditional sigma/DACER 평가 잡음을 제거한 stochastic_z가 주 지표입니다.",
        "기본 baseline 구조·학습률·평가 방식은 native 설정이며 동일 환경 스텝이 동일 연산량을 뜻하지 않습니다. MFPO는 warmup10k, 다른 방법은5k입니다.",
        "HER·reward shaping·offline 데이터가 없는 온라인 탐색 실험이며 D4RL offline normalized score와 직접 비교하지 않습니다.","",
        "![Learning curves](learning_curves.png)","![Last100k](last100k.png)"]
    (out/"REPORT_KO.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
