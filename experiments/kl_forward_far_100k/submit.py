import argparse,json,os,subprocess,time,hashlib
from pathlib import Path
from .continuation import same_settings,ORIGINAL_COMMIT

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();parent=a.parent.resolve();rt=root/'runtime';(rt/'slurm').mkdir(parents=True,exist_ok=True)
    assert not (rt/'SUBMISSION.json').exists(),'Already submitted'
    manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text());records={}
    parent_manifest=json.loads((parent/'SOURCE_MANIFEST.json').read_text())
    for f in ('actor.py','core.py','box_gaussian.py','distillation.py','evaluate.py'):
        file='experiments/kl_forward_far_1d/'+f
        assert manifest['files'][file]==parent_manifest['files'][file],file
    for n in (128,256):
        cfg=json.loads((root/f'source/experiments/kl_forward_far_100k/config_{n}.json').read_text())
        d=root/f'N{n}M{n}';d.mkdir(exist_ok=True);(d/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for s in range(4):
            r=parent/f'N{n}M{n}/runtime/runs/forward_L0_s{s}'
            run=json.loads((r/'RUN.json').read_text());done=json.loads((r/'COMPLETE.json').read_text())
            assert done['step']==20000 and run['seed']==s and run['source_commit']==ORIGINAL_COMMIT
            same_settings(run['config'],cfg)
            records[f'N{n}M{n}/s{s}']=dict(parent=str(r),sha256=hashlib.sha256((r/'checkpoint.msgpack').read_bytes()).hexdigest(),step=20000,source_commit=ORIGINAL_COMMIT)
    (root/'PARENT_CHECKPOINTS.json').write_text(json.dumps(records,indent=2)+'\n')
    cmd=['sbatch','--parsable','--partition=big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000','--qos=big_qos','--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node35,node40','--array=0-7%8','--output='+str(rt/'slurm/%A_%a.out'),'--error='+str(rt/'slurm/%A_%a.err'),str(root/'source/experiments/kl_forward_far_100k/job.sbatch')]
    job=subprocess.check_output(cmd,env=dict(os.environ,STUDY_ROOT=str(root),PARENT_ROOT=str(parent)),text=True).strip().split(';')[0]
    record=dict(time=time.time(),job=job,source_commit=manifest['commit'],numerical_source_commit=ORIGINAL_COMMIT,parent_root=str(parent),command=cmd)
    (rt/'SUBMISSION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':main()
