"""Predeclared geometric route categories and per-policy diversity metrics."""
from collections import Counter
import numpy as np


def crossing(path,x,side=-1,last=False):
    old,new=path[:-1],path[1:]
    mask=(old[:,0]>x)&(new[:,0]<=x) if side<0 else (old[:,0]<x)&(new[:,0]>=x)
    ix=np.flatnonzero(mask)
    if not len(ix):return None
    i=ix[-1] if last else ix[0]
    t=(x-old[i,0])/(new[i,0]-old[i,0])
    return float(old[i,1]+t*(new[i,1]-old[i,1]))


def route_label(task,path,goal_id):
    if not goal_id:return "failure"
    if task=="v2":
        # This maze compares two goals; do not invent multiple homotopy routes.
        y=crossing(path,4 if goal_id==1 else -4,side=1 if goal_id==1 else -1)
        return f"G{goal_id}/"+("central-corridor" if y is not None and abs(y)<2 else "unclassified")
    if task=="v1":
        y=crossing(path,-4,last=True)
        return f"G{goal_id}/"+("upper" if y is not None and y>2 else "lower" if y is not None and y< -2 else "unclassified")
    if task=="v4":
        entry=crossing(path,-4);late=crossing(path,-12,last=True)
        first="upper" if entry is not None and entry>2 else "lower" if entry is not None and entry< -2 else "unclassified"
        final="upper-outer" if late is not None and late>6 else "lower-outer" if late is not None and late< -6 else "middle" if late is not None and abs(late)<2 else "unclassified"
        return f"G{goal_id}/{first}-entry/{final}"
    y=crossing(path,-8 if goal_id==1 else 8,side=-1 if goal_id==1 else 1,last=True)
    return f"G{goal_id}/passage-y{round(y/4)*4:+d}" if y is not None else f"G{goal_id}/unclassified"


def diversity(labels):
    count=Counter(labels);n=sum(count.values())
    if not n:return dict(counts={},entropy=0.,effective_modes=0.,dominant_fraction=0.,observed_modes=0)
    p=np.array(list(count.values()),float)/n;h=float(-np.sum(p*np.log(p)))
    return dict(counts=dict(count),entropy=h,effective_modes=float(np.exp(h)),
                dominant_fraction=float(p.max()),observed_modes=len(count))


def summarize(task,paths,lengths,goals,returns):
    routes=[route_label(task,p[:int(n)+1],int(g)) for p,n,g in zip(paths,lengths,goals)]
    good=np.asarray(goals)>0
    path_lengths=np.array([np.linalg.norm(np.diff(p[:int(n)+1],axis=0),axis=1).sum() for p,n in zip(paths,lengths)])
    success_lengths=path_lengths[good]
    return dict(episodes=len(goals),success_rate=float(good.mean()),mean_return=float(np.mean(returns)),
        goal_fractions={str(i):float(np.mean(np.asarray(goals)==i)) for i in range(1,3 if task!="v1" else 2)},
        failure_fraction=float((~good).mean()),routes=routes,
        successful_goals=diversity([str(g) for g in np.asarray(goals)[good]]),
        successful_routes=diversity([r for r,g in zip(routes,good) if g and "unclassified" not in r]),
        unclassified_successes=int(sum(g and "unclassified" in r for r,g in zip(routes,good))),
        successful_path_length_mean=float(success_lengths.mean()) if len(success_lengths) else None,
        successful_path_length_std=float(success_lengths.std()) if len(success_lengths) else None)
