# AntMaze training-budget audit

Checked 2026-09-21 against upstream source and papers. This note does not change the running 100k campaign.

## Upstream DDiffPG environment repository

At commit `7edd06c4799abbab0f8fa534c21deb56253b018e`, `preprocess_cfg` sets the main environment-interaction budget to:

| Task | Main budget |
|---|---:|
| antmaze-v1 | 3,000,000 |
| antmaze-v2 | 3,000,000 |
| antmaze-v3 | 4,000,000 |
| antmaze-v4 | 5,000,000 |

Both `scripts/ddiffpg_main.py` and `scripts/baselines_main.py` call this preprocessing. The interaction counter accumulates `timesteps * num_envs`; these are aggregate environment interactions, not optimizer-update counts or per-vector steps. Warmup is performed before the counted main loop; collection batches can slightly overshoot its stopping threshold.

Sources:
- https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/utils/common.py#L36-L60
- https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/scripts/ddiffpg_main.py
- https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/scripts/baselines_main.py
- https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/ddiffpg.py#L131-L174

DDiffPG's default random warmup is 500 vector steps with 256 environments, or 128,000 interactions. Its sparse reward, intrinsic exploration, parallel collection and optimization settings differ from this campaign; its budget is not proof of a minimum required budget for our dense-reward setup.

## MaxEntDP and MFPO

- MaxEntDP Appendix D.2 and Figure 8 compare AntMaze trajectories after **1,000,000 environment interactions**, using a dense penalty for distance to the nearest goal: https://arxiv.org/html/2502.11612v3#A4.SS2
- MFPO Figure 10 explicitly says **100,000 environment interactions**: https://arxiv.org/pdf/2604.14698#page=24
- Public MFPO `train_online.py` defaults to `max_steps=1e6`, but this generic default must not be presented as a verified AntMaze-specific setting. The checked public repository does not expose the AntMaze-specific environment/experiment configuration: https://github.com/dongxiaoyi-xyz/MFPO/blob/master/train_online.py#L20-L37

## Interpretation for the current campaign

The current 100k runs are early-budget comparisons. Their budget is 1/30 of DDiffPG v1's main budget, 1/50 of v4's, and 1/10 of MaxEntDP's AntMaze comparison budget. No successful trajectories at 100k do not establish mode collapse or inability to learn the maze. A longer controlled run is needed to distinguish insufficient training from persistent exploration or implementation problems. No success at 100k is also not a universal property of AntMaze: MFPO reports successes at that point under its own experimental conditions.

The active campaign remains 48 runs at 100k. Any longer campaign must be recorded separately with unchanged comparison settings and explicit provenance; a generic 1M default is not an exact reproduction of MFPO Figure 10.
