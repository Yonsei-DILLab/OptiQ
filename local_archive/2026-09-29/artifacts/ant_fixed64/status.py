import csv,json,re,subprocess,sys
from pathlib import Path
p=Path(sys.argv[1]);m=json.loads((p/'manifest.json').read_text());out=[]
for job in m['jobs']:
 log=Path(job['log']);s=log.read_text(errors='replace') if log.exists() else '';urls=re.findall(r'https://wandb\.ai/[^\s\x1b]+/runs/[a-zA-Z0-9]+',s)
 configs=list((p/'runs').glob(job['name']+'*/config.json'))
 data={k:job[k] for k in ['name','gpu','seed']};data['wandb_url']=urls[-1] if urls else None
 data['supervisor']=subprocess.check_output(['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf','status',job['name']],text=True).strip()
 data['last_log_lines']=s.splitlines()[-5:]
 if configs:
  c=json.loads(configs[-1].read_text());alg=c['alg'];a=alg['actor'];data['config_path']=str(configs[-1]);data['verified_config']=dict(env=c['env_name'],seed=c['seed'],batch=alg['batch_size'],utd=alg['utd'],actor_lr=alg['optimizer']['lr_actor'],critic_lr=alg['optimizer']['lr_critic'],actor_width=a['hidden_dims'],critic_width=alg['critic']['hs'],temperature=a['temperature'],latent_prior=a['latent_prior'],latent_components=a['latent_components'],codebook_seed=a['latent_codebook_seed'],n=a['num_policy_samples'],m=a['num_policy_samples']*a['proposals_per_policy_sample'],loss=a['distillation_loss'],mean_init=a['mean_output_init_scale'],total_steps=c['total_steps'],runtime_commit=c['runtime']['git_commit'],runtime_dirty=c['runtime']['git_dirty'])
  progress=configs[-1].parent/'logs/progress.csv'
  if progress.exists():
   with progress.open() as f:rows=list(csv.DictReader(f))
   data['latest_metrics']={k:next((r[k] for r in reversed(rows) if r.get(k)),None) for k in ['time/total_timesteps','train/n_updates','train/actor_loss','train/critic_loss','rollout/ep_rew_mean','eval/mean_reward']}
 out.append(data)
print(json.dumps(dict(campaign=str(p),jobs=out),indent=2))
