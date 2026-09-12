# OptiQ v4

Training follows [v3](../v3/PSEUDOCODE.md), with default actor and critic hidden
widths `[256, 256]`. `G(s,z)=(mu, log_sigma)` in code; sigma is exp(log_sigma).
Initial sigma is .5. Collection, TD, teacher construction, and OT conditional
NLL continue using the conditional Gaussian. Teacher T defaults to .1.

At step 1 and every 5000 steps, evaluate the same checkpoint in both modes:

```python
for mode in ["zero_z", "stochastic_z"]:
    for episode in range(num_eval_episodes):  # default 10 EACH, total 20
        reset_environment(paired_env_seed[episode])
        reset_evaluation_rng(paired_policy_seed[episode])
        while not done:
            z = zeros(D) if mode == "zero_z" else normal(D)
            mu, log_sigma = G(s, z)
            a = tanh(mu)  # epsilon=0 for both; sigma is unused
            s, reward, done = environment.step(unscale(a))
```

Stochastic z means a fresh normal sample per action, not a nonzero constant or
a fixed episode latent. Evaluation does not mutate collection RNG or actor
parameters. Episodes use paired environment-reset seeds, with termination and
time limits handled normally. Compare both curves; do not select the better
mode after seeing the results.

W&B records `eval/zero_z/{mean_reward,std_reward,mean_ep_length}` and
`eval/stochastic_z/{mean_reward,std_reward,mean_ep_length}`. Success rates are
also recorded where available. `eval/mean_reward`, `eval/mean_ep_length`, and
`final_eval_return` are compatibility aliases for **zero_z**. Both final returns
are saved as `final_eval_return_zero_z` and `final_eval_return_stochastic_z`.
Episode rewards, lengths, reset seeds, and policy seeds are saved separately to
`evaluations_zero_z.npz` and `evaluations_stochastic_z.npz` and included in the
final evaluation artifact.

```bash
OPTIQ_PYTHON=/path/to/venv/bin/python bash scripts/run_v4.sh 0 --check benchmark=hopper
OPTIQ_PYTHON=/path/to/venv/bin/python bash scripts/run_v4.sh 0 benchmark=hopper
# Equivalent direct entry (requires the usual W&B credentials):
python run_optiq_dime.py --config-name=mujoco_v4 benchmark=hopper seed=0
```

Default project: `OptiQ/v4_test`. Outputs are separate from v3. Existing v3
configs and launcher retain their previous network/evaluation defaults.
