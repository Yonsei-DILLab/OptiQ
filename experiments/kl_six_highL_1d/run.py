import argparse,hashlib,json,os,platform,shutil,time
from pathlib import Path
import flax.serialization
import jax
from ..kl_diverse_targets_1d.run import train,write
from .core import implementation
from .evaluate import evaluate

EXT=Path('/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions')

def reuse_forward(root,cfg,case,seed):
    out=root/'runtime/confirm'/case['id']/f'forward_s{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    stage='screen' if seed<2 else 'validate_seeds'
    src=EXT/case['forward_campaign']/'runtime'/stage/case['id']/f'forward_s{seed}'
    old=json.loads((src/'RUN.json').read_text());done=json.loads((src/'COMPLETE.json').read_text())
    assert done['step']==100000 and old['method']=='forward' and old['L']==0
    numerical=['n','m','batch','temperature','learning_rate','hidden_dims','log_std_min','log_std_max','initial_log_std','mean_output_init_scale','teacher_std_floor','action_bound','target_centers']
    numerical+=['shapes','shape_masses','core_intervals'] if case['target_kind']=='nongmm' else ['target_widths','target_masses']
    combined=dict(cfg,**case,density_chunk=256,train_block=100,checkpoint_interval=1000)
    assert all(combined[k]==old['config'][k] for k in numerical)
    exp=implementation(combined)(combined,'forward',0,seed)
    initial=hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest();assert initial==old['initial_parameter_sha256']
    exp.restore(src/'checkpoint.msgpack');assert int(exp.state.step)==100000
    shutil.copy2(src/'checkpoint.msgpack',out/'checkpoint.msgpack')
    write(out/'RUN.json',dict(config=combined,method='forward',L=0,seed=seed,stage='confirm',
        source_commit=old['source_commit'],evaluation_commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit'],
        initial_parameter_sha256=initial,reused=True,parent_run=str(src),
        parent_checkpoint_sha256=hashlib.sha256((src/'checkpoint.msgpack').read_bytes()).hexdigest(),
        job=os.environ.get('SLURM_JOB_ID'),hostname=platform.node(),starting_step=100000,time=time.time()))
    # Preserve learning curves as provenance; new 1M final metrics are distinct.
    shutil.copy2(src/'training.jsonl',out/'training.jsonl')
    for metric in src.glob('metrics_*.json'):
        if metric.name!='metrics_100000.json':shutil.copy2(metric,out/metric.name)
    write(out/'metrics_100000.json',evaluate(exp,out,100000))
    write(out/'COMPLETE.json',dict(step=100000,reused=True,source_commit=old['source_commit'],time=time.time()))
    write(out/'STATUS.json',dict(step=100000,phase='complete_reused',time=time.time()))
    del exp;jax.clear_caches()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('config.json').read_text());assert cfg['allow_large_L'] and cfg['reverse_L']==1048576 and jax.default_backend()=='gpu'
    case=cfg['cases'][a.index//4];seed=cfg['seeds'][a.index%4]
    reuse_forward(a.root,cfg,case,seed)
    train(a.root,cfg,case,seed,'reverse',100000,1048576,'confirm',experiment_cls=implementation(case),evaluate_fn=evaluate)
if __name__=='__main__':main()
