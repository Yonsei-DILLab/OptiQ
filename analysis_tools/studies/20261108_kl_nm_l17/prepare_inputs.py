"""Prepare a separate L17 runtime and audit completed FKL reuse, CPU only."""
import argparse
import json
import shutil
from pathlib import Path
from common import tasks, write, file_sha

NUMERICAL = ['n','m','batch','temperature','learning_rate','hidden_dims','log_std_min',
             'log_std_max','initial_log_std','mean_output_init_scale','teacher_std_floor',
             'action_bound','target_width','target_centers','target_kind','steps',
             'final_eval_samples','final_sample_chunk','histogram_bins','train_block','density_chunk']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();base=a.base.resolve()
    initial=json.loads((base/'REUSE_AUDIT.json').read_text())
    # RKL records here are initialization references only, never reusable results.
    write(root/'REUSE_AUDIT.json',dict(entries=initial['entries'],
          purpose='Historical paired initialization hashes; only FKL inputs below are reused.',
          reuse_forward_count=len(tasks('reuse')),fresh_reverse_count=54))
    records={}
    for task in tasks('reuse'):
        assert task['method']=='forward' and task['L']==0
        if task['n']==task['m']==128:
            src=base/'reused'/task['target']/f'forward_s{task["seed"]}'
        else:
            src=base/'revisions/v3/runtime/train'/task['id']
        run=json.loads((src/'RUN.json').read_text());done=json.loads((src/'COMPLETE.json').read_text())
        assert run['method']=='forward' and run['L']==0 and run['seed']==task['seed']
        assert done['step']==100000
        keys=NUMERICAL+(['target_widths','target_masses'] if task['config']['target_kind']=='gmm'
                        else ['shapes','shape_masses','core_intervals','expected_modes'])
        mismatch={k:[run['config'].get(k),task['config'][k]] for k in keys
                  if run['config'].get(k)!=task['config'][k]}
        assert not mismatch,(src,mismatch)
        sha=file_sha(src/'checkpoint.msgpack')
        assert run.get('parent_checkpoint_sha256',sha)==sha
        dst=root/'reused'/task['id'];dst.mkdir(parents=True,exist_ok=True)
        for name in ['checkpoint.msgpack','RUN.json','COMPLETE.json']:
            shutil.copy2(src/name,dst/name)
        for old in src.glob('metrics_*.json'):
            shutil.copy2(old,dst/old.name)
        rec=dict(id=task['id'],method='forward',target=task['target'],seed=task['seed'],
                 n=task['n'],m=task['m'],L=0,origin=str(src),checkpoint_sha256=sha,
                 source_commit=run['source_commit'],initial_parameter_sha256=run['initial_parameter_sha256'],
                 config_numerically_matched=True,matched_keys=keys)
        write(dst/'ORIGIN.json',rec);records[task['id']]=rec
    assert len(records)==10
    write(root/'REUSE_INPUTS.json',records)
    write(root/'HISTORY_ROOTS.json',['../..','../v2','../v3'])
    shutil.copy2(base/'ENVIRONMENT_LOCK.txt',root/'ENVIRONMENT_LOCK.txt')
    print(json.dumps(dict(reused_fkl=len(records),fresh_rkl=54,fresh_fkl=44,new_training_runs=98)))


if __name__=='__main__':main()
