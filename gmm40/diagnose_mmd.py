"""Evaluation sampling variability from five disjoint blocks of saved samples.

This does not estimate variability across independently trained seeds and does
not replace the original 2,048-sample primary metric.
"""
import json
import numpy as np
from .target import RESULTS,Target
from .evaluation import atomic_json,mmd2


def main():
    reference=Target().sample(10000,20260917,bounded=True)
    rows=[]
    for job in json.loads((RESULTS/'queue.json').read_text())['jobs']:
        if job['steps']!=100000 or '--navigation' in job.get('args',[]):continue
        folder=RESULTS/job['name']
        if not (folder/'status.json').exists():continue
        if json.loads((folder/'status.json').read_text()).get('status')!='completed':continue
        data=np.load(folder/'evaluations/step_0100000/samples.npy')
        if len(data)<10000:continue
        scores=[];bands=[]
        for start in range(0,10000,2000):
            score,band=mmd2(data[start:start+2000],reference[start:start+2000])
            scores.append(score);bands.append(band)
        rows.append(dict(name=job['name'],block_scores=scores,block_mean=float(np.mean(scores)),
                         block_std=float(np.std(scores,ddof=1)),
                         sampling_standard_error=float(np.std(scores,ddof=1)/np.sqrt(5)),
                         mean_by_bandwidth={key:float(np.mean([b[key] for b in bands])) for key in bands[0]}))
        print(job['name'],rows[-1]['block_mean'],flush=True)
    out=RESULTS/'diagnostics/mmd_sampling'
    out.mkdir(exist_ok=True)
    atomic_json(out/'results.json',dict(blocks=5,samples_per_block=2000,reference_seed=20260917,
                updates=100000,training_performed=False,rows=rows,
                caveat='Evaluation sampling variability conditional on trained seed-0 models; not training-seed uncertainty. Primary metrics remain unchanged.'))
    lines=['# MMD evaluation sampling check','',
           'Five disjoint 2,000-sample blocks from each completed 100K model; common target reference blocks.',
           'The original 2,048-sample primary metric is unchanged. This quantifies sampling variability of fixed models, not training-seed variability.',
           '', '| Run | Block mean MMD² | Block SD | SE of block mean | h=1 | h=2 | h=5 | h=10 | h=20 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values=[row[key] for key in ('block_mean','block_std','sampling_standard_error')]+list(row['mean_by_bandwidth'].values())
        lines.append('| '+row['name']+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
