"""Analyze unchanged upstream runs; mixture density, no KDE fitting."""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.signal import find_peaks
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
GRID = np.linspace(-0.999, 0.999, 2001)
CENTERS = np.array([-0.6, 0., 0.6])
ZTARGET = np.mean(ndtr((1-CENTERS)/.1)-ndtr((-1-CENTERS)/.1))
TARGET = np.exp(-.5*((GRID[:, None]-CENTERS)/.1)**2).mean(1)/(.1*np.sqrt(2*np.pi)*ZTARGET)

def density(mu, ls):
    u = np.arctanh(GRID)
    total = np.zeros_like(GRID)
    for start in range(0, len(mu), 128):
        mean = mu[start:start+128].reshape(-1, 1)
        logstd = ls[start:start+128].reshape(-1, 1)
        total += np.exp(-.5*((u[None]-mean)*np.exp(-logstd))**2-logstd-.5*np.log(2*np.pi)).sum(0)
    return total / len(mu) / (1-GRID**2)

def inspect(path):
    d = np.load(path)
    den = density(d['mu'], d['log_sigma'])
    peaks, properties = find_peaks(den, prominence=.05, distance=100)
    loc = GRID[peaks]
    mass = np.histogram(d['samples'], [-1, -.3, .3, 1])[0]/len(d['samples'])
    matches = [bool(np.any(np.abs(loc-c) <= .15)) for c in CENTERS]
    return d, den, dict(step=int(path.stem.split('_')[1]),
        histogram_tv=float(.5*np.abs(d['histogram']-d['target_bin_mass']).sum()),
        basin_mass=mass.tolist(), peak_locations=loc.tolist(),
        peak_prominences=properties['prominences'].tolist(),
        target_peaks_present=matches, matched_target_peaks=sum(matches),
        density_latents=int(len(d['mu'])), sample_count=int(len(d['samples'])))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--step',type=int,default=20000)
    args=parser.parse_args()
    records=[]
    fig, axs=plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    plt.rcParams.update({'font.size': 10})
    for index, ax in enumerate(axs.flat):
        seed=index%4
        variant='baseline' if index<4 else 'init1'
        folder=ROOT/'runs'/f'{variant}_N64_M64_s{seed}'
        if not (folder/f'eval_{args.step:06d}.npz').exists():
            ax.set_title(f'Seed {seed}: pending')
            continue
        d, den, result=inspect(folder/f'eval_{args.step:06d}.npz')
        history=[inspect(p)[2] for p in sorted(folder.glob('eval_*.npz'))]
        config=json.loads((folder/'config.json').read_text())
        records.append(dict(seed=seed, variant=variant, source_commit=config['source_commit'], audit_commit=config['audit_commit'], final=result, history=history))
        ax.hist(d['samples'], bins=np.linspace(-1,1,129), density=True, color='#61a5c2', alpha=.4, label='32,768 policy samples')
        ax.plot(GRID, TARGET, color='#202a44', lw=2, ls='--', label='Target')
        ax.plot(GRID, den, color='#cf4b36', lw=2, label='Policy density (2,048 latent MC)')
        for x in result['peak_locations']:
            ax.plot(x, den[np.argmin(abs(GRID-x))], 'o', color='#cf4b36', ms=4)
        ax.set_title(f"{variant} / seed {seed}\nTarget peaks {result['matched_target_peaks']}/3 | TV {result['histogram_tv']:.3f}")
        ax.text(.02,.95,'Basin mass L/C/R: '+ '/'.join(f'{v:.1%}' for v in result['basin_mass']), transform=ax.transAxes,va='top',fontsize=9)
        ax.set_xlim(-1,1);ax.set_ylim(0,2.5);ax.grid(alpha=.15)
        ax.set_xlabel('Action')
        if seed==0:ax.set_ylabel(('Default mean init 1e-4' if variant=='baseline' else 'Larger mean init 1.0')+'\nDensity')
    handles, labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,.955), ncol=3, fontsize=10, frameon=False)
    fig.suptitle(f'Latest heejoon baseline | fresh latent | 64 x 64 | batch 32 | {args.step:,} updates',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.90])
    fig.savefig(ROOT/f'comparison_{args.step//1000}k.png', dpi=180)
    output=dict(criterion='Local maxima of 2,048-latent MC mixture density; prominence >= 0.05, separation >= 0.10, within 0.15 of each target center. Not KDE; finite latent MC approximation.', runs=records)
    (ROOT/f'summary_{args.step//1000}k.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps([dict(seed=r['seed'], variant=r['variant'], **r['final']) for r in records],indent=2))

if __name__=='__main__':main()
