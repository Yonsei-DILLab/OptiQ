#!/usr/bin/env bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "$utils/logging.sh" ""
. "$utils/environment.sh"
set -u
temperatures=(0.1 0.25 0.5 1.0)
index="$1"
temperature="${temperatures[$index]}"
exec /workspace/OptiQ-v2/scripts/run_v2.sh 0 \
  "alg.actor.temperature=$temperature" alg.behavior_uniform_probability=0.0 total_steps=20000 \
  num_eval_episodes=3 diagnostic_interval=1000 checkpoint_interval=20000 \
  progress_bar=false \
  "run_name=humanoid-v2-calibration-behavior0-T${temperature}-s0" \
  output_root=outputs/v2_calibration \
  wandb.project=optiq_mujoco_v2_calibration \
  wandb.group=humanoid-v2-temperature-calibration-behavior0 \
  wandb.job_type=calibration
