"""Wall-clock accounting only; no sampling or optimization dependencies."""
from contextlib import contextmanager
import time


class TrainingClock:
    """Measure a training stage, excluding explicitly timed evaluations.

    The caller must synchronize pending training work before evaluation starts.
    Compilation, training logging and checkpoint writes remain training costs.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.evaluation_seconds = 0.
        self.evaluations = 0

    @contextmanager
    def evaluation(self):
        started = self.clock()
        try:
            yield
        finally:
            self.evaluation_seconds += self.clock() - started
            self.evaluations += 1

    def snapshot(self):
        elapsed = self.clock() - self.started
        return dict(stage_wall_seconds=elapsed,
                    evaluation_seconds=self.evaluation_seconds,
                    measured_training_seconds=elapsed-self.evaluation_seconds,
                    timed_evaluations=self.evaluations)
