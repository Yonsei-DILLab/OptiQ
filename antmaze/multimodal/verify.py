"""Read-only audit of completed policies against their raw rollout/checkpoint files.

This reporting tool may be newer than frozen training code. Its own Git SHA is
recorded separately. Missing/running jobs are not accepted as completed results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from antmaze.evaluation import atomic_json
from .analysis import summarize
from .env import geometry

METHODS=("optiq","sac","meow","sql","mfpo","dipo")


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def equivalent(expected,actual,where):
    if isinstance(expected,dict):
        assert set(expected)<=set(actual),where
        for key,value in expected.items():equivalent(value,actual[key],where+"."+key)
    elif isinstance(expected,list):
        assert len(expected)==len(actual),where
        for i,(x,y) in enumerate(zip(expected,actual)):equivalent(x,y,f"{where}[{i}]")
    elif isinstance(expected,(int,float,np.number)):
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6,err_msg=where)
    else:assert expected==actual,where


def verify_run(folder,job,source_commit,smoke=False):
    r=json.loads((folder/"result.json").read_text())
    c=json.loads((folder/"config.json").read_text())
    steps=272 if smoke else job.get("steps",100000)
    warmup=256 if smoke else (10000 if job["method"]=="mfpo" else 5000)
    noveld=c.get("profile")=="ddiffpg-dense-noveld"
    routefast=c.get("profile")=="ddiffpg-routefast"
    if noveld:
        steps=8704 if smoke else 100000;warmup=8192
    if routefast:
        steps=512 if smoke else 100000;warmup=256 if smoke else 8192
    episodes=2 if smoke else 100
    assert r["completed"] and c["smoke"]==smoke
    assert r["source_commit"]==c["source_commit"]==source_commit
    assert c["steps"]==r["steps"]==steps and c["warmup"]==warmup
    if noveld or routefast:
        count=8 if routefast else 256
        expected_updates=int(np.ceil((steps-warmup)/count))*8
        assert r["updates"]==c["expected_updates"]==expected_updates
        assert c["batch_size"]==(256 if routefast else 4096) and c["num_envs"]==count and c["updates_per_round"]==8
        assert c["utd"]==8/count and c["intrinsic"]["coefficient"]==.01
        if routefast:
            assert c["assistance"]["curriculum_boundaries"]==[60000,80000]
            assert c["assistance"]["success_bonus"]==10 and c["assistance"]["wall_clearance"]==.6
        rnd=json.loads((folder/"intrinsic-audit.json").read_text())
        assert rnd["passed"] and rnd["updates"]==expected_updates
        assert rnd["initial"]["target"]==rnd["final"]["target"]
        assert rnd["initial"]["predictor"]!=rnd["final"]["predictor"]
    else:assert r["updates"]==steps-warmup and c["batch_size"]==256 and c["utd"]==1
    for key in ("task","method","seed"):assert c[key]==r[key]==job[key]
    expected_labels={"policy-natural","policy-fixed"}
    if job["method"]=="optiq":expected_labels|={"mu_only-natural","mu_only-fixed"}
    assert set(r["summaries"])==expected_labels
    g=geometry(job["task"]);goals=np.asarray(g["goals"])
    equivalent(g,c["environment"],"environment")
    audit=json.loads((folder/"parameter-audit.json").read_text())
    assert audit["initial"] and audit["initial"].keys()==audit["final"].keys()
    for key,initial in audit["initial"].items():
        final=audit["final"][key]
        assert final["parameters"]==initial["parameters"]>0
        assert final["sha256"]!=initial["sha256"]
    checkpoint=json.loads((folder/"checkpoint.json").read_text())
    assert checkpoint["step"]==steps and checkpoint["parameter_audit"]==audit["final"]
    hashes={}
    for file in checkpoint["files"]:
        path=folder/"checkpoints"/file["name"]
        assert path.stat().st_size==file["size"] and digest(path)==file["sha256"]
        hashes[str(path.relative_to(folder))]=file["sha256"]
    assert hashes,"Final checkpoint is required"
    coverage=np.load(folder/"training_coverage.npz",allow_pickle=False)
    assert int(coverage["counts"].sum()+coverage["outside"])==steps
    for label in sorted(expected_labels):
        path=folder/"rollouts"/f"{steps}-{label}.npz"
        z=np.load(path,allow_pickle=False);s=json.loads(path.with_suffix(".json").read_text())
        assert z["xy"].shape==(episodes,g["horizon"]+1,2)
        assert z["initial_state"].shape==(episodes,29)
        assert z["lengths"].shape==z["returns"].shape==z["goal_ids"].shape==(episodes,)
        assert np.isfinite(z["returns"]).all() and np.isfinite(z["initial_simulator_state"]).all()
        assert ((z["lengths"]>=1)&(z["lengths"]<=g["horizon"])).all()
        assert ((z["goal_ids"]>=0)&(z["goal_ids"]<=len(goals))).all()
        np.testing.assert_array_equal(z["initial_state"][:,:2],z["xy"][:,0])
        if label.endswith("fixed"):
            initial=z["initial_simulator_state"]
            np.testing.assert_array_equal(initial,np.broadcast_to(initial[0],initial.shape))
        for i,(xy,n,goal) in enumerate(zip(z["xy"],z["lengths"],z["goal_ids"])):
            xy=xy[:int(n)+1];assert np.isfinite(xy).all()
            assert np.isnan(z["xy"][i,int(n)+1:]).all()
            distances=np.linalg.norm(xy[:,None,:]-goals[None,:,:],axis=-1)
            closest=distances.min(axis=1)
            # XY archives are float32, while simulator rewards use float64.
            np.testing.assert_allclose(z["returns"][i],-closest[1:].sum(),rtol=2e-6,atol=.02)
            np.testing.assert_allclose(z["final_distance"][i],closest[-1],atol=1e-5,rtol=2e-6)
            np.testing.assert_allclose(z["min_distance"][i],closest.min(),atol=1e-5,rtol=2e-6)
            if goal:
                assert int(distances[-1].argmin())+1==goal and closest[-1]<=.50001
                assert (closest[1:-1]>.49999).all(),"Trajectory continued after reaching a goal"
            else:
                assert n==g["horizon"] and closest[-1]>.49999
                assert (closest[1:]>.49999).all(),"Unrecorded success"
        fresh=summarize(job["task"],z["xy"],z["lengths"],z["goal_ids"],z["returns"])
        equivalent(fresh,s,label);equivalent(s,r["summaries"][label],"result."+label)
        assert s["episodes"]==episodes and s["step"]==steps
        assert s["training_seed"]==job["seed"] and s["task"]==job["task"]
        assert s["evaluation_mode"]==("mu_only" if label.startswith("mu_only") else "policy")
        assert s["reset_mode"]==("identical_full_state" if label.endswith("fixed") else "training_reset_distribution")
        hashes[str(path.relative_to(folder))]=digest(path)
        hashes[str(path.with_suffix(".json").relative_to(folder))]=digest(path.with_suffix(".json"))
    for name in ("config.json","result.json","checkpoint.json","parameter-audit.json","training_coverage.npz","history-policy-natural.json"):
        hashes[name]=digest(folder/name)
    if noveld or routefast:hashes["intrinsic-audit.json"]=digest(folder/"intrinsic-audit.json")
    return dict(passed=True,steps=steps,updates=r["updates"],episode_count_per_reset=episodes,sha256=hashes)


def verify_campaign(root,partial=False,preflight=False):
    m=json.loads((root/"manifest.json").read_text())
    assert bool(m["smoke"])==preflight
    methods=m.get("methods",METHODS);seeds=m.get("seeds",[0] if preflight else list(range(4)))
    expected={(task,method,seed) for task in m["tasks"] for method in methods for seed in seeds}
    jobs=m["jobs"]
    if m.get("shard_count",1)>1:
        all_jobs=m["all_jobs"]
        assert len(all_jobs)==len(expected) and {(j["task"],j["method"],j["seed"]) for j in all_jobs}==expected
        expected={(j["task"],j["method"],j["seed"]) for i,j in enumerate(all_jobs) if i%m["shard_count"]==m["shard_index"]}
    assert len(jobs)==len(expected) and {(j["task"],j["method"],j["seed"]) for j in jobs}==expected
    status=json.loads((root/"status.json").read_text())
    states={j["id"]:j["status"] for j in status["jobs"]}
    report={"training_source_commit":m["source_commit"],"preflight":preflight,
            "total":len(jobs),"verified":{},"pending":[]}
    report["reporting_source_commit"]=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    report["reporting_files_sha256"]={name:digest(Path(__file__).parent/name)
        for name in ("verify.py","analysis.py","report.py","env.py")}
    for j in jobs:
        folder=root/"runs"/j["id"]
        if states.get(j["id"])!="completed" or not (folder/"result.json").exists():
            report["pending"].append(j["id"]);continue
        report["verified"][j["id"]]=verify_run(folder,j,m["source_commit"],preflight)
    report["complete"]=not report["pending"]
    if not partial:assert report["complete"],f"Incomplete: {report['pending']}"
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--partial",action="store_true");p.add_argument("--preflight",action="store_true")
    a=p.parse_args();report=verify_campaign(a.root,a.partial,a.preflight)
    (a.root/"report").mkdir(exist_ok=True)
    atomic_json(a.root/"report"/"validation.json",report)
    print(json.dumps({"verified":len(report["verified"]),"total":report["total"],"complete":report["complete"]}))


if __name__=="__main__":main()
