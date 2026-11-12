"""Saved-result paired audit; no rerunning or filtering selected states."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();rows=[]
    cases=json.loads(Path(__file__).with_name('phase_a.json').read_text())['cases']
    for case in cases:
        stem=f'{case["env"]}_s{case["seed"]}_state{case["state"]}_n32768'
        old=next(a.runs.glob(f'validation-ca3f7e1-gpu*/{stem}_legacy'))
        new=next(a.runs.glob(f'validation-final-gpu*/{stem}_verified'))
        before=json.loads((old/'diagnosis.json').read_text());after=json.loads((new/'result.json').read_text())
        mode=before['diagnostics'][1];graph=before['diagnostics'][2];fixed=after['events'][-1]
        row=dict(case,old_far=mode['far'],new_far=fixed['far'],old_modes=mode['modes'],new_modes=fixed['modes'],
                 old_negative=graph['negative_pairs'],new_negative=fixed['negative_pairs'],
                 old_min_persistence=graph['minimum_persistence'],new_min_persistence=fixed['minimum_persistence'],
                 old_endpoint_above_birth=mode['endpoint_above_assigned_mode_count'],
                 old_edge_above_birth=graph['edges_above_incident_births'],
                 old_seconds=graph['seconds'],new_seconds=fixed['seconds'],new_counts=fixed['counts'])
        with np.load(old/'legacy_graph.npz') as x,np.load(new/'verified_graph.npz') as y:
            distance=np.max(abs(x['modes'][:,None]-y['modes'][None]),axis=-1)
            mapping=distance.argmin(1);row['max_old_mode_to_new_distance']=float(distance.min(1).max())
            row['changed_sample_labels']=int(np.sum(mapping[x['labels']]!=y['labels']))
            row['old_max_assignment_distance']=float(x['assignment_distance'].max())
            row['new_max_assignment_distance']=fixed['max_assignment_distance']
        rows.append(row);print(json.dumps(row),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(cases=rows),indent=2)+'\n')


if __name__=='__main__':main()
