"""Compare actual and identical-state NovelD bonuses without updating RND."""
import os
os.environ.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_PY_FORCE_CPU='1',
                  LD_LIBRARY_PATH='/home/heechan/.mujoco/mujoco210/bin',D4RL_SUPPRESS_IMPORT_ERROR='1')
from pathlib import Path
import sys,json,types
import numpy as np
import torch
torch.set_num_threads(1)
source=Path(sys.argv[1]);root=Path(sys.argv[2])
sys.path.insert(0,str(source/'antmaze'))
# The root package registers MuJoCo environments; this neural-network-only audit
# does not need its simulator imports. Load unchanged submodules by their paths.
package=types.ModuleType('ddiffpg')
package.__path__=[str(source/'antmaze/ddiffpg')]
package.LIB_PATH=source/'antmaze/ddiffpg'
sys.modules['ddiffpg']=package
from ddiffpg.utils.intrinsic import IntrinsicM
out={}
for run in sorted(root.glob('*')):
    if run.name != 'v1-optiq-s0' or not (run/'checkpoint-final.pt').exists():continue
    cp=torch.load(run/'checkpoint-final.pt',map_location='cpu',weights_only=False)
    i=np.random.default_rng(924).choice(len(cp['replay']['buf_obs']),8192,replace=False)
    obs=cp['replay']['buf_obs'][i];nxt=cp['replay']['buf_next_obs'][i]
    model=IntrinsicM((29,),env_name='antmaze-'+cp['config']['task'],normalize=False,device='cpu')
    model.rnd_model.load_state_dict(cp['intrinsic']['model'])
    with torch.no_grad():
        bonus=model.compute_reward(obs,nxt).numpy().ravel()
        same=model.compute_reward(obs,obs).numpy().ravel()
    all_xy=cp['replay']['buf_next_obs'][:,:2].numpy()
    bins=np.floor(all_xy/.5).astype(np.int32)
    unique,inverse,counts=np.unique(bins,axis=0,return_inverse=True,return_counts=True)
    occupancy=counts[inverse[i]]
    cut=np.quantile(occupancy,[.25,.75])
    rare=occupancy<=cut[0]; common=occupancy>=cut[1]
    spatial=dict(grid_m=.5,retained_replay_transitions=len(all_xy),visited_bins=len(unique),
        sampled_bin_count_quartiles=cut.tolist(),
        less_visited=dict(n=int(rare.sum()),noveld_mean=float(bonus[rare].mean()),noveld_std=float(bonus[rare].std())),
        more_visited=dict(n=int(common.sum()),noveld_mean=float(bonus[common].mean()),noveld_std=float(bonus[common].std())),
        corr_bonus_log_occupancy=float(np.corrcoef(bonus,np.log1p(occupancy))[0,1]),
        note='Visit frequency in retained last1M replay, not first-ever novelty; final frozen RND recomputation.')
    step_xy=torch.linalg.vector_norm(nxt[:,:2]-obs[:,:2],dim=-1).numpy()
    quartiles=np.quantile(step_xy,[0,.25,.5,.75,1])
    groups=[]
    for j in range(4):
        mask=(step_xy>=quartiles[j]) & (step_xy<=quartiles[j+1] if j==3 else step_xy<quartiles[j+1])
        groups.append(dict(xy_step_min=float(quartiles[j]),xy_step_max=float(quartiles[j+1]),
            count=int(mask.sum()),noveld_mean=float(bonus[mask].mean())))
    out[run.name]=dict(spatial_occupancy=spatial,checkpoint_step=cp['step'],checkpoint_source=cp['config']['source_commit'],sample_count=len(i),actual_mean=float(bonus.mean()),actual_std=float(bonus.std()),
        identical_state_bonus_mean=float(same.mean()),
        identical_state_to_actual_mean_ratio=float(same.mean()/bonus.mean()),
        corr_bonus_xy_step=float(np.corrcoef(bonus,step_xy)[0,1]),xy_step_quartiles=groups,
        note='Diagnostic on saved replay and final RND; no learning. Correlation is not a causal test.')
print('AUDIT_JSON='+json.dumps(out,allow_nan=False))
