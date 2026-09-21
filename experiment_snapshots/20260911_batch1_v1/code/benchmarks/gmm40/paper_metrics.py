"""Sample metrics following iDEM's GMM definitions, in original coordinates.

The 2D TV protocol uses 200 bins per axis. Explicit out-of-range mass is
retained in an overflow bin. It is never replaced by energy/occupancy TV.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


@torch.no_grad()
def sample_w2(first, second):
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError('W2 evaluation requires equal point matrices')
    costs = torch.cdist(first.float(),second.float()).square().cpu().numpy()
    rows, cols = linear_sum_assignment(costs)
    return float(np.sqrt(costs[rows,cols].astype(np.float64).mean()))


def spatial_tv(first, second, bins=200, bounds=None):
    first = first.detach().cpu().numpy() if isinstance(first, torch.Tensor) else np.asarray(first)
    second = second.detach().cpu().numpy() if isinstance(second, torch.Tensor) else np.asarray(second)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != 2 or second.shape[1] != 2:
        raise ValueError('GMM spatial TV requires 2D points')
    reference, xedges, yedges = np.histogram2d(second[:,0],second[:,1],bins=bins,range=bounds)
    generated, _, _ = np.histogram2d(first[:,0],first[:,1],bins=(xedges,yedges))
    ref_mass = np.r_[reference.reshape(-1),len(second)-reference.sum()] / len(second)
    gen_mass = np.r_[generated.reshape(-1),len(first)-generated.sum()] / len(first)
    return {'tvd':float(.5*np.abs(ref_mass-gen_mass).sum()),
            'generated_overflow':float(gen_mass[-1]), 'reference_overflow':float(ref_mass[-1]),
            'bounds':[[float(xedges[0]),float(xedges[-1])],[float(yedges[0]),float(yedges[-1])]],
            'bins_per_axis':bins}


@torch.no_grad()
def repeated_sample_metrics(samples, target, seed, count=1000, repeats=10):
    if len(samples)<count:
        raise ValueError('Insufficient generated samples for the paper protocol')
    rng=np.random.default_rng(seed)
    rows=[]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for index in range(repeats):
            reference=target.sample((count,))
            generated=samples[rng.choice(len(samples),count,replace=False)]
            tv=spatial_tv(generated,reference)
            # Paper does not fully specify histogram limits: disclose both the
            # reference-derived range and the official GMM plotting range.
            tv_fixed=spatial_tv(generated,reference,bounds=[[-56,56],[-56,56]])
            rows.append({'repeat':index,'w2':sample_w2(generated,reference),
                         'spatial_tv':tv['tvd'],'spatial_tv_fixed_bounds':tv_fixed['tvd'],
                         'spatial_tv_generated_overflow':tv['generated_overflow'],
                         'reference_mean_log_p':float(target.log_prob(reference).mean())})
    return rows
