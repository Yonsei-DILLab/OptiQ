"""Optional fixed teacher-temperature schedule measured in environment steps."""

import math
from collections.abc import Mapping
from typing import NamedTuple


class TemperatureSchedule(NamedTuple):
    final_temperature: float
    anneal_steps: int
    decay: str = "log_linear"


def parse_temperature_schedule(actor, backup_mode):
    config = actor.get("temperature_schedule")
    if config is None:
        return None
    if not isinstance(config, Mapping) or not isinstance(config.get("enabled"), bool):
        raise ValueError("temperature_schedule requires a mapping and boolean enabled")
    if not config["enabled"]:
        return None
    if backup_mode != "td":
        raise ValueError("Teacher temperature annealing requires plain TD backup")
    initial = float(actor["temperature"])
    final = float(config.get("final_temperature", float("nan")))
    if not math.isfinite(initial) or not math.isfinite(final) or not 0 < final <= initial:
        raise ValueError("temperature_schedule requires finite initial >= final > 0")
    steps = config.get("anneal_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("temperature_schedule.anneal_steps must be a positive integer")
    decay = config.get("decay", "log_linear")
    if decay not in ("linear", "log_linear"):
        raise ValueError("temperature_schedule.decay must be linear or log_linear")
    return TemperatureSchedule(final, steps, decay)


def scheduled_temperature(initial, schedule, env_steps, warmup_steps):
    """Decay after warmup; preserve log-linear behavior for existing profiles."""
    progress = min(max((env_steps - warmup_steps) / schedule.anneal_steps, 0.0), 1.0)
    if progress == 0.0:
        temperature = float(initial)
    elif progress == 1.0:
        temperature = schedule.final_temperature
    elif schedule.decay == "linear":
        temperature = (1 - progress) * float(initial) + progress * schedule.final_temperature
    else:
        temperature = math.exp((1 - progress) * math.log(initial)
                               + progress * math.log(schedule.final_temperature))
    return temperature, progress
