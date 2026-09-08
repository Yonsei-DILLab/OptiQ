"""Environment-step schedules for candidate generation only."""

import math


def validate_proposal_schedule(actor, total_steps):
    schedule = actor.get("proposal_schedule")
    if schedule is None:
        return
    initial = float(schedule.initial_temperature)
    hold = schedule.hold_steps
    end = schedule.end_steps
    if not math.isfinite(initial) or initial < 1:
        raise ValueError("proposal_schedule.initial_temperature must be finite and >= 1")
    if not isinstance(hold, int) or not isinstance(end, int) or not 0 <= hold < end <= total_steps:
        raise ValueError("proposal_schedule requires integer 0 <= hold_steps < end_steps <= total_steps")


def proposal_parameters(actor, env_steps):
    """Scale kernel std and local support together; leave Q temperature alone.

    A factor of one returns the exact baseline values. The factor is held,
    then linearly annealed to one. Action-space truncation still applies in
    TruncatedGaussianKDE, including when the local clip exceeds two.
    """
    schedule = actor.get("proposal_schedule")
    if schedule is None:
        return actor.proposal_std, actor.proposal_clip, 1.0
    fraction = min(1.0, max(0.0, (env_steps - schedule.hold_steps) /
                            (schedule.end_steps - schedule.hold_steps)))
    temperature = 1.0 + (float(schedule.initial_temperature) - 1.0) * (1.0 - fraction)
    scale = math.sqrt(temperature)
    return actor.proposal_std * scale, actor.proposal_clip * scale, temperature
