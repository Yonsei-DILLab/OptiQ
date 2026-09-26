"""Plot actual saved histograms; do not interpolate missing training states."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..kl_nongmm_targets_1d.figure_three import CASES
from ..kl_paper_highL.render import style, save, BLUE, ORANGE


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(workspace, steps):
    parent = workspace / 'reports/20260925_kl_paper_row/tight/PROVENANCE.json'
    original = read(parent)
    saved = {(r['case'], r['method'], r['seed']): r for r in original['runs']}
    data, inputs, metrics = {}, [], []
    for pair, (case_id, _, study, reference) in enumerate(CASES[:2]):
        root = workspace / 'studies' / study
        config = read(root/'config.json')
        config = dict(config, **next(c for c in config['cases'] if c['id'] == case_id))
        x = np.linspace(-10, 10, 16385)
        target = reference(config, x)[0]
        for col, method in enumerate(['forward', 'reverse']):
            for step in steps:
                densities = []
                for seed in range(4):
                    original_file = Path(saved[case_id, method, seed]['file'])
                    folder = original_file.parent
                    sample = folder/f'samples_{step:06d}.npz'
                    metric_path = folder/f'metrics_{step:06d}.json'
                    metric = read(metric_path)
                    digest = sha(sample)
                    if step == 100000:
                        assert digest == saved[case_id, method, seed]['sha256']
                    with np.load(sample) as z:
                        edges, mass = z['edges'].copy(), z['histogram_mass'].copy()
                        count = len(z['actions'])
                        assert count == metric['sample_count']
                        assert len(mass) == 512
                        assert np.allclose(np.histogram(z['actions'], edges)[0] / count, mass)
                        assert np.isclose(.5*np.abs(mass-z['target_mass']).sum(), metric['histogram_TV'])
                        densities.append(mass/np.diff(edges))
                    inputs.append(dict(case=case_id, method=method, seed=seed, step=step,
                                       samples=count, file=str(sample), sha256=digest))
                    metrics.append(dict(case=case_id, method=method, seed=seed, step=step,
                                        samples=count, histogram_TV=metric['histogram_TV']))
                data[step, 2*pair+col] = dict(values=np.array(densities), edges=edges, x=x, target=target)
    return data, inputs, metrics, sha(parent)


def draw(data, steps, out, name):
    # Match the tight 1x4 paper figure's plot area and typography.
    panel_w, panel_h = 18.8*(.995-.048)/(4+3*.30), 2.6
    gap_x, gap_y = .50, .65
    left, right, bottom, top = 1.25, .15, .65, .68
    width = left+4*panel_w+3*gap_x+right
    height = bottom+len(steps)*panel_h+(len(steps)-1)*gap_y+top
    fig = plt.figure(figsize=(width, height))
    # Column limits are fixed across time, including the faint seed curves.
    upper = [1.05*max(max(float(data[s,c]['values'].max()),float(data[s,c]['target'].max()))
                     for s in steps) for c in range(4)]
    for row, step in enumerate(steps):
        y = bottom+(len(steps)-1-row)*(panel_h+gap_y)
        fig.text(.06/width, (y+panel_h/2)/height, f'{step//1000}K' if step else '0',
                 ha='left', va='center', fontsize=18)
        for col in range(4):
            x = left+col*(panel_w+gap_x)
            ax = fig.add_axes([x/width,y/height,panel_w/width,panel_h/height])
            d = data[step,col]; color = BLUE if col%2 == 0 else ORANGE
            for v in d['values']:
                ax.stairs(v,d['edges'],color=color,alpha=.18,lw=.65)
            mean = d['values'].mean(0)
            ax.stairs(mean,d['edges'],fill=True,color=color,alpha=.20,lw=0)
            ax.stairs(mean,d['edges'],color=color,lw=1.55)
            ax.plot(d['x'],d['target'],'k--',lw=1.45)
            ax.set(xlim=(-10,10),ylim=(0,upper[col]))
            ax.set_xticks([-10,-5,0,5,10])
            ax.tick_params(direction='out',length=4,width=.8)
            if col == 0: ax.set_ylabel('Density')
            if row == len(steps)-1: ax.set_xlabel(r'$a$')
            if row == 0:
                ax.set_title(f'({"abcd"[col]}) {"Forward" if col%2 == 0 else "Reverse"} KL',pad=12)
    save(fig,out,name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--render-commit',required=True);a=p.parse_args()
    out=a.workspace/'reports/20260926_kl_training_progress/original_saved';out.mkdir(parents=True,exist_ok=True)
    steps=[0,1000,10000,25000,50000,75000,100000]
    style();data,inputs,metrics,parent_sha=load(a.workspace,steps)
    draw(data,steps[2:],out,'training_progress_saved')
    draw(data,steps[:2],out,'initial_0_1k')
    with (out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    (out/'PROVENANCE.json').write_text(json.dumps(dict(render_commit=a.render_commit,
        parent_provenance_sha256=parent_sha,reverse_L=1024,seeds=[0,1,2,3],bins=512,
        available_steps=steps,primary_steps=steps[2:],inputs=inputs,
        note='Original saved trajectories; missing multiples of 10K are not interpolated.'),indent=2)+'\n')
    (out/'caption.md').write_text('''# Original saved training progression

Columns match the supplied final figure: three Gaussian modes (Forward, Reverse),
then spike + ramp (Forward, Reverse). Rows are the actual saved 10K, 25K, 50K,
75K and 100K optimizer updates, not equally spaced time intervals. Initialization
and 1K are exported separately. No intermediate distribution is interpolated.

Reverse uses L=1024; N=M=128, batch 32 and four seeds match the original figure.
Curves average actual 512-bin histograms without smoothing; thin curves show
individual seeds. The original intermediate evaluations use 32,768 samples per
seed, and 100K uses 262,144 samples per seed. All original final input hashes
match the supplied final figure. Black dashed curves are the exact target.
Each column uses a fixed y-axis range across its displayed training times.
''')
    (out/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name!='SHA256.json'},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
