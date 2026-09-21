"""Per-policy trajectories and aggregate metrics; preserve seed separation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from antmaze.evaluation import atomic_json
from .env import geometry

METHODS=("optiq","sac","meow","sql","mfpo","dipo")
COLORS={0:"#aaaaaa",1:"#2580b4",2:"#e58437"}


def maze(ax,task):
    g=geometry(task)
    for x,y in g["walls"]:ax.add_patch(Rectangle((x-2,y-2),4,4,color="#46484d",zorder=2))
    for i,(x,y) in enumerate(g["goals"],1):
        ax.add_patch(Circle((x,y),.5,color=COLORS[i],zorder=4))
        ax.text(x,y+.85,f"G{i}",ha="center",fontsize=8,color=COLORS[i],weight="bold",zorder=5)
    ax.scatter(0,0,s=30,c="black",marker="*",zorder=5)
    w=np.asarray(g["walls"])
    ax.set(xlim=(w[:,0].min()-2,w[:,0].max()+2),ylim=(w[:,1].min()-2,w[:,1].max()+2),aspect="equal")
    ax.set_xticks([]);ax.set_yticks([])


def draw_paths(ax,folder,task,label):
    maze(ax,task)
    p=folder/"rollouts"/f"100000-{label}.npz"
    if not p.exists():ax.text(.5,.5,"Pending",transform=ax.transAxes,ha="center");return
    z=np.load(p,allow_pickle=False)
    for path,n,goal in zip(z["xy"],z["lengths"],z["goal_ids"]):
        xy=path[:int(n)+1]
        ax.plot(xy[:,0],xy[:,1],color=COLORS[int(goal)],alpha=.16 if goal else .065,lw=.65,zorder=3)
    s=json.loads(p.with_suffix(".json").read_text())
    ax.set_title(f"Success {s['success_rate']:.0%} · routes {s['successful_routes']['observed_modes']}\n"
                 f"dominant {s['successful_routes']['dominant_fraction']:.0%}",fontsize=8)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--partial",action="store_true");p.add_argument("--maps-only",action="store_true")
    a=p.parse_args();root=a.root;out=root/"report";out.mkdir(parents=True,exist_ok=True)
    if a.maps_only:
        fig,axes=plt.subplots(1,3,figsize=(11,4))
        for ax,task in zip(axes,("v1","v3","v4")):maze(ax,task);ax.set_title(task)
        fig.suptitle("DDiffPG maze layouts · start ★ · goals G1/G2 · scale4")
        fig.tight_layout();fig.savefig(out/"maze_layouts.png",dpi=180);plt.close(fig);return
    m=json.loads((root/"manifest.json").read_text())
    if m["smoke"]:raise ValueError("Preflight is not a learned-policy comparison")
    from .verify import verify_campaign
    validation=verify_campaign(root,partial=a.partial)
    atomic_json(out/"validation.json",validation)
    complete={};missing=[]
    for task in m["tasks"]:
        for method in METHODS:
            for seed in range(4):
                job=f"{task}-{method}-s{seed}";d=root/"runs"/job;p=d/"result.json"
                if not p.exists():missing.append(job);continue
                r=json.loads(p.read_text());c=json.loads((d/"config.json").read_text())
                assert r["completed"] and r["steps"]==100000 and not c["smoke"]
                assert r["source_commit"]==m["source_commit"]==c["source_commit"]
                assert r["updates"]==100000-c["warmup"]
                for label,s in r["summaries"].items():
                    assert s["episodes"]==100
                    z=np.load(d/"rollouts"/f"100000-{label}.npz",allow_pickle=False)
                    assert len(z["xy"])==100 and np.isfinite(z["returns"]).all()
                    if label.endswith("fixed"):
                        state=z["initial_simulator_state"];np.testing.assert_array_equal(state,np.broadcast_to(state[0],state.shape))
                complete[job]=r
    if missing and not a.partial:raise RuntimeError(f"Missing completed jobs: {missing}")
    for task in m["tasks"]:
        for label in ("policy-natural","policy-fixed"):
            fig,axes=plt.subplots(6,4,figsize=(13,18))
            for row,method in enumerate(METHODS):
                for seed in range(4):
                    ax=axes[row,seed];job=f"{task}-{method}-s{seed}"
                    if job in complete:draw_paths(ax,root/"runs"/job,task,label)
                    else:maze(ax,task);ax.text(.5,.5,"Pending",transform=ax.transAxes,ha="center")
                    ax.set_xlabel(f"{method} · seed {seed}",fontsize=9)
            fig.suptitle(f"{task} · 100k interactions · {label} · 100 rollouts per INDIVIDUAL policy",fontsize=13)
            fig.tight_layout();fig.savefig(out/f"trajectories-{task}-{label}.png",dpi=160);plt.close(fig)
        fig,axes=plt.subplots(6,4,figsize=(13,18))
        coverage_max=max([float(np.load(root/"runs"/j/"training_coverage.npz")["counts"].max())
                          for j in complete if j.startswith(task+"-")]+[1.])
        for row,method in enumerate(METHODS):
            for seed in range(4):
                ax=axes[row,seed];job=f"{task}-{method}-s{seed}";maze(ax,task)
                if job in complete:
                    z=np.load(root/"runs"/job/"training_coverage.npz",allow_pickle=False)
                    counts=z["counts"].astype(float);counts[counts==0]=np.nan
                    ax.imshow(np.log1p(counts.T),origin="lower",extent=[z["low"][0],z["high"][0],z["low"][1],z["high"][1]],
                        cmap="YlOrRd",zorder=1,interpolation="nearest",vmin=0,vmax=np.log1p(coverage_max))
                ax.set_title(f"{method} · seed {seed}",fontsize=9)
        fig.suptitle(f"{task} · training xy visits · common log color scale 0–{coverage_max:.0f}; evaluation excluded")
        fig.tight_layout();fig.savefig(out/f"coverage-{task}.png",dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(13,4))
        for seed,ax in enumerate(axes):
            job=f"{task}-optiq-s{seed}"
            if job in complete:draw_paths(ax,root/"runs"/job,task,"mu_only-fixed")
            else:maze(ax,task)
            ax.set_xlabel(f"OptiQ seed{seed}")
        fig.suptitle(f"{task} · supplement only: random z, conditional sigma removed, fixed start")
        fig.tight_layout();fig.savefig(out/f"optiq-mu-only-{task}.png",dpi=160);plt.close(fig)
        # One bar per individual trained policy. Fractions include failures.
        fig,axes=plt.subplots(2,1,figsize=(15,8))
        available=[f"{task}-{method}-s{seed}" for method in METHODS for seed in range(4)
                   if f"{task}-{method}-s{seed}" in complete]
        if available:
            labels=[j.removeprefix(task+"-") for j in available]
            records=[complete[j]["summaries"]["policy-fixed"] for j in available]
            routes=sorted({k for r in records for k in r["successful_routes"]["counts"]})
            for ax,kind,keys in ((axes[0],"goals",list(range(1,len(geometry(task)["goals"])+1))),
                                 (axes[1],"routes",routes)):
                bottom=np.zeros(len(records))
                for idx,key in enumerate(keys):
                    values=np.array([r["goal_fractions"].get(str(key),0) if kind=="goals" else
                        r["successful_routes"]["counts"].get(key,0)/r["episodes"] for r in records])
                    color=COLORS[key] if kind=="goals" else plt.get_cmap("tab20")(idx%20)
                    ax.bar(labels,values,bottom=bottom,color=color,label=f"G{key}" if kind=="goals" else key)
                    bottom+=values
                if kind=="routes":
                    unknown=np.array([r["unclassified_successes"]/r["episodes"] for r in records])
                    ax.bar(labels,unknown,bottom=bottom,color="#9b75b8",label="Unclassified success");bottom+=unknown
                ax.bar(labels,1-bottom,bottom=bottom,color="#dddddd",label="Failure")
                ax.set(ylim=(0,1),ylabel="Fraction of all 100 rollouts",title=f"Successful {kind} and failures")
                ax.tick_params(axis="x",rotation=60);ax.legend(fontsize=7,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.3))
        fig.suptitle(f"{task} · same full initial state · each bar is ONE policy")
        fig.tight_layout();fig.savefig(out/f"goal-route-fractions-{task}.png",dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for method in METHODS:
            histories=[]
            for seed in range(4):
                path=root/"runs"/f"{task}-{method}-s{seed}"/"history-policy-natural.json"
                if path.exists():histories.append(json.loads(path.read_text()))
            if not histories:continue
            common=sorted(set.intersection(*[{r["step"] for r in h} for h in histories]))
            for ax,key in zip(axes,("success_rate","mean_return","mean_min_distance")):
                vals=np.array([[next(r[key] for r in h if r["step"]==s) for s in common] for h in histories])
                mean=vals.mean(axis=0);sd=vals.std(axis=0,ddof=1) if len(vals)>1 else np.zeros_like(mean)
                line,=ax.plot(common,mean,label=f"{method} (n={len(histories)})")
                ax.fill_between(common,mean-sd,mean+sd,color=line.get_color(),alpha=.12)
                ax.set(xlabel="Environment interactions",title=key);ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{task} · periodic direct-policy evaluation · training seed mean ± sample SD")
        fig.tight_layout();fig.savefig(out/f"learning-curves-{task}.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(len(m["tasks"]),6,figsize=(18,4*len(m["tasks"])),squeeze=False)
    for row,task in enumerate(m["tasks"]):
        for col,method in enumerate(METHODS):
            ax=axes[row,col];job=f"{task}-{method}-s0"
            if job in complete:draw_paths(ax,root/"runs"/job,task,"policy-fixed")
            else:maze(ax,task);ax.text(.5,.5,"Pending",transform=ax.transAxes,ha="center")
            ax.set_xlabel(f"{task} · {method} · seed0")
    fig.suptitle("100 stochastic rollouts from the same full initial state · seed0 policy only")
    fig.tight_layout();fig.savefig(out/"seed0-overview.png",dpi=180);plt.close(fig)
    aggregates={};rows=[]
    for task in m["tasks"]:
        for method in METHODS:
            records=[complete[f"{task}-{method}-s{s}"]["summaries"]["policy-fixed"] for s in range(4) if f"{task}-{method}-s{s}" in complete]
            if not records:continue
            row=dict(task=task,method=method,n=len(records),final=len(records)==4)
            for name,values in dict(success_rate=[r["success_rate"] for r in records],
                effective_routes=[r["successful_routes"]["effective_modes"] for r in records],
                dominant_route_fraction=[r["successful_routes"]["dominant_fraction"] for r in records]).items():
                row[name]=dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
            rows.append(row)
    atomic_json(out/"results.json",dict(source_commit=m["source_commit"],completed=len(complete),missing=missing,aggregate=rows,per_policy=complete))
    lines=["# AntMaze 다중 목표·경로 비교", "",f"완료 {len(complete)}/{len(m['jobs'])}. 각 정책은100k online interaction 이후 평가했습니다.","",
        "주 그림은 동일한 전체 시뮬레이터 초기 상태에서 정책을 직접 샘플링한100회 rollout입니다. 각 칸은 한 training seed의 정책이며 시드를 섞지 않았습니다.","",
        "| 미로 | 방법 | 완료seed | 성공률 | 유효 경로 수 | 최다 경로 비중 |","|---|---|---:|---:|---:|---:|"]
    for r in rows:
        def fmt(k):
            v=r[k];return f"{v['mean']:.3f}"+(f" ± {v['sample_sd']:.3f}" if v["sample_sd"] is not None else "")
        lines.append(f"| {r['task']} | {r['method']} | {r['n']}/4 | {fmt('success_rate')} | {fmt('effective_routes')} | {fmt('dominant_route_fraction')} |")
    lines += ["","유효 경로 수는 성공 rollout의 경로 범주 엔트로피를 exp한 값입니다. 성공이 없으면0으로 표시하며 mode collapse라고 단정하지 않습니다.",
        "경로는 사전에 정의한 통로 횡단으로 분류했습니다. 모든 가능한 homotopy class나 action 분포의 multimodality를 증명하는 값은 아닙니다.",
        "v4의 원래 초기 상태 분포는 고정되어 natural/fixed 평가 조건이 같습니다. 두 결과를 독립된200회로 합치지 않습니다.",
        "OptiQ 주 그림은 learned sigma를 포함한 정책 샘플입니다. DACER 외부 행동잡음은 제외하며 μ-only 그림은 별도 보조 자료입니다.",
        "원본 저속 Ant/미로를 MuJoCo3로 이식하고 명시적인 최근접목표 거리 보상을 적용했습니다. 미공개 MFPO AntMaze 설정의 정확한 재현으로 해석하지 않습니다.",
        "![Seed0](seed0-overview.png)"]
    (out/"REPORT_KO.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":main()
