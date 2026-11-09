"""Small evaluation summaries, independent of optional reporting modules."""
import math
import numpy as np


def goal_metrics(result):
    counts=np.asarray(result['goals'],dtype=np.int64)
    if counts.sum()+result['failure']!=result['episodes']:
        raise ValueError('Goal/success episode counts disagree')
    if int(np.count_nonzero(counts))!=result['reachable_goals']:
        raise ValueError('Goal coverage disagrees with goal counts')
    p=counts/counts.sum() if counts.sum() else np.zeros(len(counts))
    entropy=-float(np.sum(p[p>0]*np.log(p[p>0]))) if counts.sum() else None
    return dict(result,goal_proportions_of_successes=p.tolist(),
        normalized_goal_entropy=entropy/math.log(len(counts)) if entropy is not None else None,
        effective_goals=math.exp(entropy) if entropy is not None else None)
