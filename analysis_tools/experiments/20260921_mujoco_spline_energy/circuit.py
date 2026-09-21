"""Compatibility imports for the committed MuJoCo experiment harness."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.spline_energy.model import (  # noqa: E402,F401
    ConditionalSplineCircuit,
    coarse_leaf_probabilities,
    initial_leaf_bias,
    log_integrals,
    log_prob_from_output,
    q_from_output,
    sample_action,
    sample_from_output,
)
