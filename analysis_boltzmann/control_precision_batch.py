"""Audit all pilot control grids without modifying their results."""
import argparse,json
from pathlib import Path
import jax
from .control_precision import audit
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    results=[]
    for run in sorted((Path(a.campaign)/'runs').glob('control*')):
        if not (run/'COMPLETE').exists():continue
        results.append(audit(run,root/run.name))
        (root/'summary.json').write_text(json.dumps(results,indent=2));jax.clear_caches()
    (root/'COMPLETE').write_text('audit complete; consult summary for convergence\n')
