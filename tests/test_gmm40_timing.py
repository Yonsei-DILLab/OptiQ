"""Evaluation pauses and unused future steps must not inflate sampler timing."""
import pytest

from benchmarks.gmm40.timing_audit import segment_timing
from benchmarks.gmm40.training_clock import TrainingClock


def test_training_clock_counts_variable_work_and_excludes_evaluation():
    now = [0.]
    timer = TrainingClock(clock=lambda: now[0])
    now[0] = 3.  # Compilation and training are included.
    with timer.evaluation():
        now[0] += 100.
    now[0] += 7.  # Later updates may take a different time.
    first = timer.snapshot()
    assert first['stage_wall_seconds'] == 110.
    assert first['measured_training_seconds'] == 10.
    now[0] += 40.
    assert timer.snapshot()['measured_training_seconds'] == 50.
    assert first['measured_training_seconds'] == 10.  # Saved checkpoints are fixed.


def test_training_clock_records_interrupted_evaluation():
    now = [0.]
    timer = TrainingClock(clock=lambda: now[0])
    now[0] = 2.
    with pytest.raises(RuntimeError), timer.evaluation():
        now[0] += 3.
        raise RuntimeError('evaluation interrupted')
    assert timer.snapshot() == dict(stage_wall_seconds=5., evaluation_seconds=3.,
                                   measured_training_seconds=2., timed_evaluations=1)


def test_timing_excludes_evaluation_pause_and_later_steps():
    rows=[dict(update=1000,elapsed_seconds=7.,**{'eval/modes_covered':40})]
    for step in range(1100,2100,100):
        elapsed=7.+(step-1000)*.01+(100 if step>1500 else 0)
        rows.append(dict(update=step,elapsed_seconds=elapsed,actor_loss=1.))
        if step==1500:
            rows.append(dict(update=step,elapsed_seconds=elapsed+100,**{'eval/modes_covered':40}))
    # This is an intermediate checkpoint, before the requested final budget.
    result=segment_timing(rows,1000,1800)
    assert result['updates']==800
    assert result['estimated_training_seconds']==pytest.approx(8.)
    assert result['observed_segment_span_seconds']==pytest.approx(108.)


def test_timing_rejects_insufficient_history():
    with pytest.raises(ValueError):
        segment_timing([dict(update=0,elapsed_seconds=1.,actor_loss=1.)],0,0)
