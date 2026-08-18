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
- `plot_density_correction_ablation.py`: matched-format rollout comparison for
  trained policies with and without proposal-density correction.

Run the controlled correction ablation from the repository root:

```bash
python -m experiments.four_way_multigoal.train \
  --density-correction \
  --output-dir outputs/four_way_multigoal/corrected
python -m experiments.four_way_multigoal.train \
  --no-density-correction \
  --output-dir outputs/four_way_multigoal/uncorrected
python -m experiments.four_way_multigoal.plot_density_correction_ablation \
  --uncorrected-run outputs/four_way_multigoal/uncorrected/seed_1 \
  --corrected-run outputs/four_way_multigoal/corrected/seed_1 \
  --output outputs/four_way_multigoal/density_correction_ablation.png
```

Generated results belong in `outputs/four_way_multigoal/`; selected figures are
versioned in `assets/figures/four_way_multigoal/`.
