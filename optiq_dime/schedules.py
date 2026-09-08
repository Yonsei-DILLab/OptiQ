"""Separate environment-step schedules for KDE width and Q target temperature."""

import math


def validate_temperature_schedule(actor, total_steps):
    schedule = actor.get("temperature_schedule")
    if schedule is None:
        return
    initial = float(schedule.initial_temperature)
    final = float(actor.temperature)
    end = schedule.end_steps
    if not math.isfinite(initial) or not math.isfinite(final) or not 0 < final <= initial:
        raise ValueError("temperature_schedule requires finite initial_temperature >= temperature > 0")
    if isinstance(end, bool) or not isinstance(end, int) or not 0 < end <= total_steps:
        raise ValueError("temperature_schedule.end_steps must be an integer in (0, total_steps]")


def target_temperature(actor, env_steps):
    """Exponentially decay Q temperature to actor.temperature, then hold it.

    The clock starts at environment step zero, including random-action warmup.
    This changes Q / T in candidate weights; it does not scale KDE or TD noise.
    """
    schedule = actor.get("temperature_schedule")
    if schedule is None or env_steps >= schedule.end_steps:
        return actor.temperature
    initial = float(schedule.initial_temperature)
    fraction = max(0.0, env_steps / schedule.end_steps)
    return initial * math.exp(math.log(float(actor.temperature) / initial) * fraction)


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
