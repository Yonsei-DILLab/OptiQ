"""Read-only audit/report for the 16-run campaign; no simulator/GPU dependency."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .analysis import summarize

TRAINING_SHA="19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5"
METHODS=("optiq","sac","meow","mfpo")
TASKS=("v1","v2","v3","v4")
GOALS={"v1":[[-8.,0.]],"v2":[[8.,0.],[-8.,8.]],"v3":[[-12.,12.],[12.,-12.]],"v4":[[-16.,4.],[-16.,-4.]]}
COLORS={0:"#89939c",1:"#2282b5",2:"#ec8c34"}
LINES={"optiq":"#168882","sac":"#c26330","meow":"#9270af","mfpo":"#4261a2"}


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def state_digest(value):
    """Snapshot-v1 canonical digest, available without Flax/JAX/MuJoCo."""
    import torch
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        elif isinstance(x,torch.Tensor):visit(x.detach().cpu().numpy())
        elif hasattr(x,"shape") and hasattr(x,"dtype"):
            a=np.asarray(x);h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
        elif isinstance(x,bytes):h.update(x)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def equivalent(expected,actual,path="result"):
    if isinstance(expected,dict):
        assert set(expected)<=set(actual),path
        for k,v in expected.items():equivalent(v,actual[k],path+"."+k)
    elif isinstance(expected,list):
        assert len(expected)==len(actual),path
        for i,(x,y) in enumerate(zip(expected,actual)):equivalent(x,y,f"{path}[{i}]")
    elif isinstance(expected,(float,int,np.number)):
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6,err_msg=path)
    else:assert expected==actual,path


def verify_run(folder,allow_smoke=False):
    """Check actual replay rewards/checkpoints and recompute rollout outcomes."""
    import torch
    digest=state_digest
    folder=Path(folder);c=read(folder/"config.json");r=read(folder/"result.json")
    assert c["source_commit"]==r["source_commit"]==TRAINING_SHA
    assert not c["smoke"] or allow_smoke,"Smoke data cannot be reported as production"
    steps=c["steps"];episodes=2 if c["smoke"] else 100
    assert r["completed"] and r["steps"]==steps
    if not c["smoke"]:assert steps==1000000
    assert c["profile"]=="dense-noveld-1m" and c["seed"]==0
    assert c["task"] in TASKS and c["method"] in METHODS
    for k in ("task","method","seed"):assert r[k]==c[k]
    expected_warmup=256 if c["smoke"] else (10000 if c["method"]=="mfpo" else 5000)
    assert c["warmup"]==expected_warmup and r["updates"]==steps-expected_warmup
    assert c["batch_size"]==256 and c["utd"]==1 and c["num_envs"]==1
    assert c["intrinsic"]["type"]=="noveld" and c["intrinsic"]["coefficient"]==.01
    assert c["intrinsic"]["novelty_discount"]==.5 and not c["intrinsic"]["normalize"]
    assert c["reward"]=="environment=-nearest goal distance; learner adds NovelD; evaluation excludes NovelD"
    goals=np.asarray(GOALS[c["task"]]);np.testing.assert_array_equal(c["environment"]["goals"],goals)
    horizon=500 if c["task"] in ("v1","v2") else 700
    assert c["environment"]["horizon"]==horizon
    audit=read(folder/"parameter-audit.json")
    for k,initial in audit["initial"].items():
        final=audit["final"][k]
        assert final["parameters"]==initial["parameters"]>0 and final["sha256"]!=initial["sha256"]
    intrinsic=read(folder/"intrinsic-audit.json")
    assert intrinsic["passed"] and intrinsic["updates"]==r["updates"]
    assert intrinsic["initial"]["target"]==intrinsic["final"]["target"]
    assert intrinsic["initial"]["predictor"]!=intrinsic["final"]["predictor"]
    checkpoint=folder/"resume"/f"step_{steps:010d}"
    proof=read(checkpoint/"manifest.json");equivalent(proof,read(folder/"checkpoint.json"))
    assert proof["step"]==steps and proof["updates"]==r["updates"] and proof["source_commit"]==TRAINING_SHA
    hashes={}
    for name,spec in proof["files"].items():
        p=checkpoint/name;assert p.stat().st_size==spec["bytes"] and file_hash(p)==spec["sha256"]
        hashes[str(p.relative_to(folder))]=spec["sha256"]
    state=torch.load(checkpoint/"state.pt",map_location="cpu",weights_only=False)
    assert digest(state)==proof["state_digest"]
    assert state["model"]["updates"]==state["intrinsic"]["updates"]==r["updates"]
    assert state["extra"]["source_commit"]==TRAINING_SHA
    assert state["extra"]["method"]==c["method"] and state["extra"]["task"]==c["task"]
    assert digest(state["intrinsic"]["target"])==intrinsic["final"]["target"]
    assert digest(state["intrinsic"]["predictor"])==intrinsic["final"]["predictor"]
    with np.load(checkpoint/"replay.npz",allow_pickle=False) as replay:
        data={k:replay[k] for k in replay.files}
        assert digest(data)==proof["replay_digest"]
        assert proof["replay_size"]==state["replay"]["size"]==len(data["rewards"])==min(steps,1000000)
        assert state["replay"]["position"]==steps%1000000 and state["replay"]["capacity"]==1000000
        assert all(np.isfinite(v).all() for v in data.values())
        # All stored transitions, including terminal transitions: no intrinsic
        # reward, simulator reset observations, or locomotion reward in replay.
        distances=np.linalg.norm(data["next_observations"][:,:2,None]-goals.T[None,:,:],axis=1)
        closest=distances.min(axis=1)
        np.testing.assert_allclose(data["rewards"],-closest,rtol=2e-6,atol=1e-5)
        terminal=data["dones"].astype(bool)
        assert (closest[terminal]<=.50001).all() and (closest[~terminal]>.49999).all()
    cov=np.load(folder/"training_coverage.npz",allow_pickle=False)
    assert int(cov["counts"].sum()+cov["outside"])==steps
    np.testing.assert_array_equal(cov["counts"],state["coverage"]["counts"])
    training=read(folder/"training_episodes.json")
    assert training["training_steps"]==state["coverage"]["total_steps"]==steps
    assert sum(e["length"] for e in training["episodes"])+state["coverage"]["episode_length"]==steps
    assert sum(e["success"] for e in training["episodes"])==training["training_successes"]
    successful=[e["end_step"] for e in training["episodes"] if e["success"]]
    assert training["first_success_step"]==(successful[0] if successful else None)
    assert r["training"]["first_success_step"]==training["first_success_step"]
    labels={f"{mode}-{reset}" for mode in (["native","policy","zero_z"] if c["method"]=="optiq" else ["native","policy"])
            for reset in ("natural","fixed")}
    assert set(r["summaries"])==labels
    for label in sorted(labels):
        path=folder/"rollouts"/f"{steps}-{label}.npz"
        s=read(path.with_suffix(".json"))
        with np.load(path,allow_pickle=False) as z:
            assert z["xy"].shape==(episodes,horizon+1,2)
            assert np.isfinite(z["returns"]).all() and np.isfinite(z["initial_simulator_state"]).all()
            assert ((z["lengths"]>=1)&(z["lengths"]<=horizon)).all()
            assert ((z["goal_ids"]>=0)&(z["goal_ids"]<=len(goals))).all()
            np.testing.assert_array_equal(z["initial_state"][:,:2],z["xy"][:,0])
            if label.endswith("fixed"):
                init=z["initial_simulator_state"];np.testing.assert_array_equal(init,np.broadcast_to(init[0],init.shape))
            for i,(path_xy,n,gid) in enumerate(zip(z["xy"],z["lengths"],z["goal_ids"])):
                xy=path_xy[:int(n)+1];assert np.isfinite(xy).all()
                assert np.isnan(path_xy[int(n)+1:]).all()
                ds=np.linalg.norm(xy[:,None,:]-goals[None,:,:],axis=-1);dsmin=ds.min(axis=1)
                np.testing.assert_allclose(z["returns"][i],-dsmin[1:].sum(),rtol=2e-6,atol=.02)
                np.testing.assert_allclose(z["min_distance"][i],dsmin.min(),rtol=2e-6,atol=1e-5)
                np.testing.assert_allclose(z["final_distance"][i],dsmin[-1],rtol=2e-6,atol=1e-5)
                if gid:assert ds[-1].argmin()+1==gid and dsmin[-1]<=.50001
                else:assert n==horizon and (dsmin[1:]>.49999).all()
                assert (dsmin[1:-1]>.49999).all(),"Rollout continued after goal reach"
            fresh=summarize(c["task"],z["xy"],z["lengths"],z["goal_ids"],z["returns"])
            equivalent(fresh,s,label);equivalent(s,r["summaries"][label],label)
        assert s["episodes"]==episodes and s["step"]==steps and s["training_seed"]==0
        assert s["evaluation_mode"]==label.rsplit("-",1)[0]
        hashes[str(path.relative_to(folder))]=file_hash(path)
    for mode in ("native","policy"):
        h=read(folder/f"history-{mode}-natural.json")
        if not c["smoke"]:assert [v["step"] for v in h]==list(range(0,steps+1,25000))
        assert all(v["episodes"]==(2 if c["smoke"] else 10) and np.isfinite(v["mean_return"]) for v in h)
    return dict(passed=True,smoke=c["smoke"],steps=steps,updates=r["updates"],
        training_source_commit=TRAINING_SHA,checkpoint_complete=True,replay_rewards_verified=len(data["rewards"]),
        full_state_digest=proof["state_digest"],sha256=hashes)


def maze(ax,config):
    from matplotlib.patches import Rectangle,Circle
    g=config["environment"]
    for x,y in g["walls"]:ax.add_patch(Rectangle((x-2,y-2),4,4,color="#44484e",zorder=2))
    for i,(x,y) in enumerate(g["goals"],1):
        ax.add_patch(Circle((x,y),.5,color=COLORS[i],zorder=5))
        ax.text(x,y+.8,f"G{i}",ha="center",fontsize=7,color=COLORS[i],zorder=6)
    ax.scatter(0,0,s=25,c="black",marker="*",zorder=5)
    w=np.asarray(g["walls"]);ax.set(xlim=(w[:,0].min()-2,w[:,0].max()+2),
        ylim=(w[:,1].min()-2,w[:,1].max()+2),aspect="equal",xticks=[],yticks=[])


def render(root,complete,allow_smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=root/"report";out.mkdir(exist_ok=True)
    suffix="SMOKE VALIDATION ONLY" if allow_smoke else "1M interactions · one training seed (0) per policy"
    configs={j:read(root/"runs"/j/"config.json") for j in complete}
    rows=[]
    for reset in ("fixed","natural"):
        for mode in ("native","policy"):
            fig,axes=plt.subplots(4,4,figsize=(13,13.5))
            for ri,task in enumerate(TASKS):
                for ci,method in enumerate(METHODS):
                    ax=axes[ri,ci];job=f"{task}-{method}-s0";label=mode+"-"+reset
                    if job not in complete:
                        ax.axis("off");ax.text(.5,.5,f"{task} {method}\nPending",ha="center",transform=ax.transAxes);continue
                    c=configs[job];r=complete[job];s=r["summaries"][label];maze(ax,c)
                    z=np.load(root/"runs"/job/"rollouts"/f"{r['steps']}-{label}.npz")
                    for xy,n,gid in zip(z["xy"],z["lengths"],z["goal_ids"]):
                        xy=xy[:int(n)+1];ax.plot(xy[:,0],xy[:,1],c=COLORS[int(gid)],alpha=.2,lw=.65,zorder=3)
                    dom=s["successful_routes"]["dominant_fraction"] if s["successful_routes"]["counts"] else None
                    ax.set_title(f"{task} · {method.upper()}\nSuccess {s['success_rate']:.0%} · routes {s['successful_routes']['observed_modes']}",fontsize=9)
                    if method=="optiq":ax.set_xlabel("Random z: mu-only" if mode=="native" else "Random z + conditional sigma",fontsize=8)
                    rows.append(dict(task=task,method=method,training_seed=0,mode=mode,reset=reset,
                        episodes=s["episodes"],success_rate=s["success_rate"],mean_return=s["mean_return"],
                        observed_routes=s["successful_routes"]["observed_modes"],effective_routes=s["successful_routes"]["effective_modes"],
                        dominant_route_fraction=dom,goal1=s["goal_fractions"]["1"],goal2=s["goal_fractions"].get("2"),
                        training_successes=r["training"]["training_successes"],first_training_success=r["training"]["first_success_step"]))
            meaning="Direct stochastic policy draws" if mode=="policy" else "Native evaluation: SAC mean / MFPO Q-best-of10 / MEOW prior center / OptiQ random-z mean"
            fig.suptitle(f"{meaning}\n{suffix} · {reset} start",fontsize=11)
            fig.text(.5,.013,"Gray: failed rollout · blue/orange: reached G1/G2 · no extra DACER noise or NovelD reward in evaluation",ha="center",fontsize=8)
            fig.tight_layout(rect=(0,.032,1,.952));fig.savefig(out/f"trajectories-{mode}-{reset}.png",dpi=180)
            fig.savefig(out/f"trajectories-{mode}-{reset}.pdf");plt.close(fig)
    for mode in ("native","policy"):
        fig,axes=plt.subplots(4,3,figsize=(13,12))
        for ri,task in enumerate(TASKS):
            for method in METHODS:
                job=f"{task}-{method}-s0";path=root/"runs"/job/f"history-{mode}-natural.json"
                if not path.exists():continue
                h=read(path);x=np.asarray([v["step"] for v in h])/1000000
                for ci,key in enumerate(("success_rate","mean_return","mean_min_distance")):
                    ax=axes[ri,ci];ax.plot(x,[v[key] for v in h],color=LINES[method],label=method.upper())
                    ax.set(title=f"{task} · {key}",xlabel="Environment interactions (M)");ax.grid(alpha=.2)
                    if ci==0:ax.set_ylim(-.02,1.02)
            if axes[ri,0].get_lines():axes[ri,0].legend(fontsize=7)
        fig.suptitle(f"{mode} evaluation · 10 episodes per point · single seed, no across-seed error band\n{suffix}")
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/f"learning-curves-{mode}.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(13,13))
    for ri,task in enumerate(TASKS):
        maximum=max([float(np.load(root/"runs"/j/"training_coverage.npz")["counts"].max()) for j in complete if j.startswith(task+"-")]+[1])
        for ci,method in enumerate(METHODS):
            ax=axes[ri,ci];job=f"{task}-{method}-s0"
            if job not in complete:ax.axis("off");continue
            z=np.load(root/"runs"/job/"training_coverage.npz");maze(ax,configs[job])
            counts=z["counts"].astype(float);counts[counts==0]=np.nan
            ax.imshow(np.log1p(counts.T),origin="lower",extent=[z["low"][0],z["high"][0],z["low"][1],z["high"][1]],
                vmin=0,vmax=np.log1p(maximum),cmap="YlOrRd",interpolation="nearest",zorder=1)
            ax.set_title(f"{task} · {method.upper()}\nshared row scale: 0–{maximum:.0f} visits",fontsize=8)
    fig.suptitle(f"Training xy coverage · evaluation visits excluded\n{suffix}")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/"training-coverage.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(13,14))
    for ri,task in enumerate(TASKS):
        names=[m for m in METHODS if f"{task}-{m}-s0" in complete]
        records=[complete[f"{task}-{m}-s0"]["summaries"]["policy-fixed"] for m in names]
        if not records:
            for ax in axes[ri]:ax.axis("off")
            continue
        route_keys=sorted({k for s in records for k in s["successful_routes"]["counts"]})
        for ci,kind in enumerate(("goal","route")):
            ax=axes[ri,ci];bottom=np.zeros(len(records))
            keys=list(range(1,len(GOALS[task])+1)) if kind=="goal" else route_keys
            for ki,key in enumerate(keys):
                values=np.asarray([s["goal_fractions"].get(str(key),0) if kind=="goal" else
                    s["successful_routes"]["counts"].get(key,0)/s["episodes"] for s in records])
                color=COLORS[key] if kind=="goal" else plt.get_cmap("tab20")(ki%20)
                ax.bar([m.upper() for m in names],values,bottom=bottom,color=color,
                       label=f"G{key}" if kind=="goal" else key);bottom+=values
            if kind=="route":
                unknown=np.asarray([s["unclassified_successes"]/s["episodes"] for s in records])
                if unknown.any():ax.bar([m.upper() for m in names],unknown,bottom=bottom,color="#b489ba",label="Unclassified success")
                bottom+=unknown
            ax.bar([m.upper() for m in names],1-bottom,bottom=bottom,color="#d2d6d9",label="Failure")
            ax.set(ylim=(0,1),ylabel="Fraction of ALL rollouts",title=f"{task} · {kind} choice")
            ax.legend(fontsize=6,ncol=2,loc="upper center",bbox_to_anchor=(.5,-.13))
    fig.suptitle(f"Direct stochastic policy · identical full initial state · failures retained\n{suffix}")
    fig.tight_layout(rect=(0,.02,1,.95));fig.savefig(out/"goal-route-fractions.png",dpi=180);plt.close(fig)
    with (out/"results.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/"results.json",dict(training_source_commit=TRAINING_SHA,completed=len(complete),total=16,
        smoke_validation=allow_smoke,rows=rows,per_policy=complete))
    lines=["# AntMaze dense + NovelD", "",f"1M 완료 {len(complete)}/16. 각 환경·방법당 학습 seed0 하나입니다." if not allow_smoke else "사전검사 시각화입니다. 학습 성능 결과가 아닙니다.","",
        "주 경로 그림은 정책에서 직접 샘플링한 stochastic rollout입니다. OptiQ는 random latent와 conditional sigma를 포함하며, 외부 DACER 잡음은 제외합니다. Native 성능 평가는 별도 그림·표로 구분합니다.","",
        "| 미로 | 방법 | 직접샘플 성공률 | Native 성공률 | 성공 경로 수 | 최다 경로 비율 | 학습 중 최초 성공 |","|---|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for method in METHODS:
            job=f"{task}-{method}-s0"
            if job not in complete:continue
            r=complete[job];s=r["summaries"]["policy-fixed"];n=r["summaries"]["native-fixed"]
            dom=f"{s['successful_routes']['dominant_fraction']:.1%}" if s["successful_routes"]["counts"] else "N/A"
            first=r["training"]["first_success_step"]
            lines.append(f"| {task} | {method} | {s['success_rate']:.1%} | {n['success_rate']:.1%} | {s['successful_routes']['observed_modes']} | {dom} | {first if first is not None else '없음'} |")
    lines.extend(["","성공률은 동일한 전체 초기 상태에서 최종 정책을100회 평가한 값입니다. 실패 episode도 분모에 포함합니다. 성공이 없으면 경로 수는0이며, 이를 mode collapse의 증거로 단정하지 않습니다.",
        "Native: SAC tanh(mu), MFPO Q-best-of10, MEOW prior-center, OptiQ random-z mu-only. v2/v3/v4의 원래 시작 상태가 고정되어 natural/fixed를 독립된200회처럼 합산하지 않습니다.",
        "v2는 주로 두 목표 선택을 비교합니다. 경로 범주는 미리 정의한 통로 횡단 기준이며 모든 homotopy class 또는 action distribution의 multimodality 증명은 아닙니다.",
        "각 정책은 단일 training seed이므로 seed 간 평균·표준편차나 알고리즘 일반 성능 우열을 주장하지 않습니다. MaxEntDP의 dense 보상 설명에 DDiffPG NovelD를 추가한 사용자 지정 조건입니다.",
        "모델/최적화 설정은 각 방법의 기존 기본값을 유지했습니다. OptiQ/SAC는256x2, MFPO는256x3, MEOW는 native flow 구조입니다. NovelD는 공통이며 OptiQ는 승인된 DACER 행동 탐색도 사용합니다.",
        "![직접 정책 궤적](trajectories-policy-fixed.png)","![기본 평가 궤적](trajectories-native-fixed.png)"])
    (out/"REPORT_KO.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--partial",action="store_true");p.add_argument("--allow-smoke",action="store_true")
    a=p.parse_args();proofs={};complete={};missing=[]
    for task in TASKS:
        for method in METHODS:
            job=f"{task}-{method}-s0";folder=a.root/"runs"/job
            if not (folder/"result.json").exists():missing.append(job);continue
            proofs[job]=verify_run(folder,a.allow_smoke);complete[job]=read(folder/"result.json")
    if not a.partial:assert not missing,f"Missing completed runs: {missing}"
    assert complete,"No completed policies to report"
    out=a.root/"report";out.mkdir(exist_ok=True)
    write(out/"validation.json",dict(complete=not missing and not a.allow_smoke,verified=proofs,missing=missing,
        training_source_commit=TRAINING_SHA,reporting_file_sha256=file_hash(__file__),smoke_validation=a.allow_smoke))
    render(a.root,complete,a.allow_smoke)


if __name__=="__main__":main()
