"""Online, Laplace-smoothed XY occupancy surprisal; not full-state entropy."""
import numpy as np


class XYEntropy:
    def __init__(self, bounds, width=.5, coefficient=.1):
        self.bounds = np.asarray(bounds, dtype=float)
        self.width = width
        self.coefficient = coefficient
        shape = np.ceil((self.bounds[2:] - self.bounds[:2]) / width).astype(int)
        self.counts = np.ones(tuple(shape), dtype=np.int64)

    def observe(self, xy):
        index = np.floor((np.asarray(xy)-self.bounds[:2])/self.width).astype(int)
        index = np.clip(index, 0, np.asarray(self.counts.shape)-1)
        index = tuple(index)
        # Predictive probability BEFORE counting this transition. Warmup counts too.
        bonus = self.coefficient * np.log(self.counts.sum()/self.counts[index])
        self.counts[index] += 1
        return float(bonus)


def plot_trace(folder, step, task='v1'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from .progress_reward import maze_geometry
    destination = folder/'evaluations'/f'{step:010d}'/'native-fixed'
    # Native random-z mu-only labels may be expanded by latent_profile.
    matches = list((folder/'evaluations'/f'{step:010d}').glob('*fixed/rollouts.npz'))
    path = next(p for p in matches if 'policy-' not in p.parent.name and 'zero' not in p.parent.name)
    data = np.load(path)
    np.testing.assert_array_equal(data['initial_full_state'][:, :2], 0)
    fig, ax = plt.subplots(figsize=(7, 7))
    walls, goals, bounds = maze_geometry(task)
    for x0, y0, x1, y1 in walls:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='.75'))
    for track in data['xy']:
        ax.plot(track[:,0],track[:,1],lw=.7,alpha=.3,color='tab:blue')
    ax.scatter(*goals.T,marker='*',s=180,color='tab:green')
    ax.scatter([0],[0],marker='+',s=140,color='red')
    ax.set(xlim=bounds[[0,2]],ylim=bounds[[1,3]],xlabel='x (m)',ylabel='y (m)',
           title=f'{step:,} steps | {len(data["xy"])} fixed-origin rollouts | random z, mu-only')
    ax.set_aspect('equal'); fig.tight_layout()
    output=path.parent/'trajectories.png'; fig.savefig(output,dpi=160); plt.close(fig)
    return output
