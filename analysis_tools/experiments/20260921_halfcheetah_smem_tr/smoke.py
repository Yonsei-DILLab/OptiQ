"""Real HalfCheetah B256 5K warmup plus 200-update GPU validation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import jax
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import train


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


class FiniteCheck(BaseCallback):
    def _on_step(self):
        if self.num_timesteps > 5000:
            leaves = jax.tree.leaves((
                self.model.policy.actor_state.params,
                self.model.policy.qf_state.params,
            ))
            if not all(np.isfinite(value).all() for value in leaves):
                raise FloatingPointError(f"nonfinite parameter at {self.num_timesteps}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=train.METHOD_OVERRIDES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cfg = train.compose_config([
        "benchmark=halfcheetah", "seed=0", "alg.buffer_size=10000",
        "checkpoint_interval=0", "diagnostic_interval=0",
        f"output_root={args.output}", "require_gpu=true",
    ], args.method)
    model, callbacks = train.runner.create_algorithm(cfg)
    model.model_save_path = None
    model.set_logger(configure(str(args.output/"logs"), ["stdout", "csv"]))
    started = time.monotonic()
    record = {
        "method": args.method,
        "seed": 0,
        "environment": cfg.env_name,
        "command": [str(Path(__file__).resolve()), *(__import__('sys').argv[1:])],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip()),
        "slurm_job_id": __import__('os').environ.get("SLURM_JOB_ID"),
        "jax_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        model.learn(total_timesteps=5200, callback=FiniteCheck())
        assert model._n_updates == 200
        assert int(model.policy.actor_state.step) == 200
        record.update(
            passed=True,
            timesteps=int(model.num_timesteps),
            updates=int(model._n_updates),
            actor_steps=int(model.policy.actor_state.step),
            elapsed_seconds=time.monotonic()-started,
        )
    except BaseException as exc:
        record.update(
            passed=False,
            failure=repr(exc),
            elapsed_seconds=time.monotonic()-started,
        )
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output/"result.json").write_text(json.dumps(record, indent=2)+"\n")
        callbacks.callbacks[0].eval_env.close()
        model.get_env().close()
        model.logger.close()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
