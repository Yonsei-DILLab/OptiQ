"""Read preserved OptiQ checkpoints; fill missing mu-only probes/obstacle rollouts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import flax.serialization
import numpy as np

from .agents import OptiQ
from .run import evaluate, atomic_json
from .visualize_4way import probe_policy, render as render_four
from .visualize_nway import render as render_nway


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(folder, output, reporting_source):
    if output.exists(): raise FileExistsError("preserve previous post-hoc attempt")
    cfg = json.loads((folder / "config.json").read_text())
    if cfg["method"] != "optiq": raise ValueError("OptiQ only")
    step = cfg["steps"]
    checkpoint = folder / "checkpoints" / f"policy_{step:09d}.msgpack"
    checkpoint_sha = sha(checkpoint)
    saved_mu = folder / "evaluations" / f"{step:09d}_mu_only.npz"
    saved_summary = json.loads((folder / "evaluations" / f"{step:09d}_summary.json").read_text())
    with np.load(saved_mu) as data:
        ids = data["goal_ids"]
        expected = saved_summary["mu_only"]
        if str(data["mode"]) != "mu_only" or np.bincount(ids[ids>=0],minlength=len(expected["goals"])).tolist() != expected["goals"]:
            raise ValueError("invalid preserved mu-only evaluation")
    output.mkdir(parents=True)
    with TemporaryDirectory(prefix="maze-mu-posthoc-") as temporary:
        agent = OptiQ(cfg["seed"], Path(temporary), step,
                      4 if cfg["task"].startswith("pm_") else 2,
                      cfg["batch_size"], cfg["temperature"])
        if cfg["agent"]["alg"] != agent.config["alg"]:
            raise ValueError("restoration profile differs from recorded learning config")
        policy = agent.model.policy
        template = dict(actor=policy.actor_state, critic=policy.qf_state,
                        target_actor=policy.target_actor_state,
                        target_critic_params=policy.qf_state.target_params)
        restored = flax.serialization.from_bytes(template, checkpoint.read_bytes())
        policy.actor_state, policy.qf_state = restored["actor"], restored["critic"]
        policy.target_actor_state = restored["target_actor"]
        before = flax.serialization.to_bytes((policy.actor_state, policy.qf_state, policy.target_actor_state))
        record = dict(training_source=cfg["source_commit"], reporting_source=reporting_source,
                      task=cfg["task"], temperature=cfg["temperature"], source_folder=str(folder),
                      checkpoint_sha256=checkpoint_sha, config_sha256=sha(folder/"config.json"),
                      preserved_mu_sha256=sha(saved_mu), step=step, mode="mu_only", learner_updates=0)
        if cfg["task"].startswith("pm_"):
            dest = output / f"{step:09d}_obstacle_mu_only.npz"
            record["obstacle_mu_only"] = evaluate(agent, cfg["task"],
                cfg["seed"]+27000+step, 500, "mu_only", dest, obstacle=True)
        else:
            probe = output / f"{step:09d}_probe_mu_only.npz"
            probe_policy(agent, probe, mode="mu_only")
            if cfg["task"] == "4way":
                render_four([(f"OptiQ T={cfg['temperature']:g} · random-z mu-only", saved_mu, probe)],
                            output / "policy_mu_only_and_q.png")
            else:
                render_nway(cfg["task"],saved_mu,probe,output/"policy_mu_only_and_q.png",
                            title=f"{cfg['task']} · OptiQ T={cfg['temperature']:g} · random-z mu-only")
        after = flax.serialization.to_bytes((policy.actor_state, policy.qf_state, policy.target_actor_state))
        if before != after or sha(checkpoint) != checkpoint_sha or agent.updates != 0:
            raise ValueError("evaluation mutated model/checkpoint or trained")
    record["sha256"] = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    atomic_json(output/"proof.json",record)
    return record


def main():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reporting-source",required=True)
    a=parser.parse_args()
    if subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()!=a.reporting_source:
        raise ValueError("reporting source mismatch")
    r=run(a.run,a.output,a.reporting_source)
    print(json.dumps(r,indent=2))


if __name__=="__main__":main()
