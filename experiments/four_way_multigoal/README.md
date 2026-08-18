# Four-Way Multi-Goal

Controlled validation of multimodal policy extraction in the four-way
multi-goal task.

The suite contains the local environment definition and a controlled
finite-particle policy-extraction experiment. The versioned assets additionally
include the online policy rollouts and proposal-correction mechanism used in the
paper draft.

- `environment.py`: four-way continuous-control task.
- `policy_extraction.py`: categorical OT versus independent assignment.
- `train.py`: end-to-end online OptiQ training and rollout visualization.

Generated results belong in `outputs/four_way_multigoal/`; selected figures are
versioned in `assets/figures/four_way_multigoal/`.
