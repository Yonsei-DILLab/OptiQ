"""Prepare target metadata and a committed, finite 4-seed queue (no training)."""
import argparse,json,os,hashlib
from pathlib import Path
from .evaluate import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('campaign.json').read_text())
    os.environ['GMM40_ACTION_SCALE']=str(cfg['scale']);os.environ['GMM40_RESULTS_ROOT']=str(a.output)
    from gmm40.target import initialize_target
    target=initialize_target()
    jobs=[dict(condition=c['name'],seed=s,status='pending') for c in cfg['conditions'] for s in cfg['seeds']]
    if (a.output/'queue.json').exists():raise RuntimeError('Queue already exists; do not overwrite')
    files={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('experiments/gmm40_bandit','gmm40','common','models','benchmarks/gmm40') for p in Path(folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    atomic_json(a.output/'provenance.json',dict(source_commit=a.commit,source_sha256=files,campaign=cfg,target_outside_mass=target.metadata['outside_mass']))
    atomic_json(a.output/'queue.json',jobs)
    atomic_json(a.output/'preflight_queue.json',[dict(condition=c['name'],seed=0,status='pending') for c in cfg['conditions']])
    from .evaluate import sliced_w2
    reference=target.sample(cfg['eval_samples'],20260922,bounded=False)
    independent=target.sample(cfg['eval_samples'],20260923,bounded=False)
    atomic_json(a.output/'reference_floor.json',dict(sliced_w2=sliced_w2(reference,independent),n=cfg['eval_samples']))
    print(json.dumps(dict(jobs=len(jobs),preflight=len(cfg['conditions']),outside_mass=target.metadata['outside_mass'])))
if __name__=='__main__':main()
