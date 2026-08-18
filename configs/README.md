# Experiment Configs

Configs are flat YAML mappings whose keys match `optiq-train` argument names.
Pass `--config` more than once to merge a common method config with an
environment-specific override. Later files win, and explicit command-line
options override every file.

Unknown keys are rejected instead of being silently ignored.

The initial MuJoCo protocol uses five seeds and the following training budgets:

| Environment | Training steps |
| --- | ---: |
| Hopper-v4 | 1,000,000 |
| Walker2d-v4 | 1,500,000 |
| HalfCheetah-v4 | 3,000,000 |
| Ant-v4 | 2,000,000 |
| Humanoid-v4 | 5,000,000 |
