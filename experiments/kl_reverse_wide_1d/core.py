"""Enable the previously validated reverse path on the unchanged width-1 actor."""
from ..kl_forward_wide_1d.core import Experiment as ForwardExperiment
from ..kl_forward_wide_1d.core import CENTERS, WIDTH, q_value, log_f, target_score, dense_score


class Experiment(ForwardExperiment):
    def __init__(self, cfg, method, L, seed):
        assert method == 'reverse' and int(L) > 0
        assert int(L) % min(cfg['density_chunk'], int(L)) == 0
        # Preserve actor, optimizer, and RNG initialization exactly. JIT methods
        # are first traced after __init__, with the reverse condition below.
        super().__init__(cfg, 'forward', 0, seed)
        self.method, self.L = method, int(L)

