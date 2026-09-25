#!/usr/bin/env bash
set -euo pipefail
export OPTIQ_CAMPAIGN=antmaze-v1-single-utd1-100k-20260925
export OPTIQ_WANDB_PROJECT=jaehun-antmaze
root=/home/heechan/optiq-experiments/antmaze-v1-single-utd1-100k-20260925
args=(--method optiq --task v1 --temperature 1 --budget-steps 100000
      --reward-profile progress100_euclidean_no_step_no_bonus
      --noveld off --dacer off --eval-starts upstream
      --collection-profile single-update1 --optiq-config-profile basic
      --eval-interval 10000 --interim-eval-episodes 20 --final-eval-episodes 100
      --save-intermediate-policy --video-interval 5000)
bash antmaze_experiments/launch.sh "${args[@]}" --preflight --output "$root/preflight"
exec bash antmaze_experiments/launch.sh "${args[@]}" --output "$root/run"
