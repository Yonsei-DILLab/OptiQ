"""Generate sweep commands/JSONL; never launches jobs (standard library only)."""
import argparse
import itertools
import json
from pathlib import Path
import shlex
import sys

DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "configs/sweeps/parameter_sweep.json"


def build_runs(spec, tasks=None, seeds=None):
    selected_tasks = list(spec["tasks"]) if tasks is None else tasks
    selected_seeds = spec["seeds"] if seeds is None else seeds
    if not set(selected_tasks) <= set(spec["tasks"]):
        raise ValueError("Unknown task; choose from " + ", ".join(spec["tasks"]))
    axes = spec["axes"]
    if set(axes) != {"sigma", "temperature", "beta", "anchor"}:
        raise ValueError("Expected sigma, temperature, beta, anchor axes")
    seen = set()
    for task, seed, sigma, temperature, beta, anchor in itertools.product(
        selected_tasks, selected_seeds, axes["sigma"], axes["temperature"],
        axes["beta"], axes["anchor"],
    ):
        if sigma <= 0 or temperature <= 0 or not 0 <= beta <= 1:
            raise ValueError("Require sigma,T > 0 and beta in [0,1]")
        if type(anchor) is not bool or type(seed) is not int or seed < 0:
            raise ValueError("anchor must be boolean; seed must be nonnegative integer")
        env = spec["tasks"][task]
        token = lambda v: format(v, "g").replace(".", "p")
        name = (f"{task}_s{token(sigma)}_t{token(temperature)}_b{token(beta)}"
                f"_a{int(anchor)}_seed{seed}")
        if name in seen:
            raise ValueError(f"Duplicate run: {name}")
        seen.add(name)
        overrides = {
            "task": task, "env_name": env["env_name"], "seed": seed,
            "run_name": name, "wandb.group": task,
            "alg.critic.v_min": env["v_min"], "alg.critic.v_max": env["v_max"],
            "alg.actor.proposal_std": sigma,
            "alg.actor.proposal_clip": 2.5 * sigma,
            "alg.actor.temperature": temperature,
            "alg.actor.density_beta": beta,
            "alg.actor.include_anchor": anchor,
            "alg.actor.proposals_per_policy_sample": 5 if anchor else 4,
        }
        serialize = lambda v: str(v).lower() if isinstance(v, bool) else str(v)
        argv = ["python", "run_optiq_dime.py", "--config-name=parameter_sweep"]
        argv += [f"{k}={serialize(v)}" for k, v in overrides.items()]
        yield {"run_name": name, "task": task, "seed": seed,
               "candidate_count": 80 if anchor else 64,
               "validation": env["validation"], "overrides": overrides,
               "argv": argv, "command": shlex.join(argv)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--format", choices=["summary", "commands", "jsonl"], default="summary")
    args = parser.parse_args()
    try:
        runs = list(build_runs(json.loads(args.spec.read_text()), args.tasks, args.seeds))
    except ValueError as exc:
        parser.error(str(exc))
    if args.format == "summary":
        print(f"{len(runs)} planned runs; NOTHING launched.")
        for task in dict.fromkeys(run["task"] for run in runs):
            subset = [run for run in runs if run["task"] == task]
            print(f"{task}: {len(subset)} runs — {subset[0]['validation']}")
    else:
        print("Dry-run manifest only; validate tasks before execution.", file=sys.stderr)
        for run in runs:
            print(run["command"] if args.format == "commands" else json.dumps(run))


if __name__ == "__main__":
    main()
