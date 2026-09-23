import argparse,json,os,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['validate','screen','replicate','confirm'],required=True);p.add_argument('--selected-case',type=int)
    a=p.parse_args();r=a.root.resolve();rt=r/'runtime';(rt/'slurm').mkdir(parents=True,exist_ok=True)
    record=rt/(a.stage+'_SUBMISSION.json');assert not record.exists(),'Already submitted'
    src=json.loads((r/'SOURCE_MANIFEST.json').read_text());env=dict(os.environ,STUDY_ROOT=str(r),STAGE=a.stage,VALIDATION_ONLY='0')
    cmd=['sbatch','--parsable','--partition=suma_rtx4090,asus_6000ada,big_suma_rtx3090,base_suma_rtx3090,dell_rtx3090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_a5000,tyan_a6000','--qos=big_qos','--exclude=cs-gpu-01,node05,node14,node23,node24,node31,node33,node35,node40','--output='+str(rt/'slurm'/f'{a.stage}_%A_%a.out'),'--error='+str(rt/'slurm'/f'{a.stage}_%A_%a.err')]
    if a.stage=='validate':env['VALIDATION_ONLY']='1';cmd+=['--time=00:30:00']
    else:
        assert json.loads((rt/'GPU_VALIDATION.json').read_text())['passed']
        if a.stage=='screen':cmd+=['--array=0-23%8']
        elif a.stage=='replicate':
            assert a.selected_case is not None;cmd+=['--array=0-1%2']
        else:
            assert a.selected_case is not None;cmd+=['--array=0-7%8','--time=1-12:00:00','--job-name=KL-mode-confirm']
            assert (r/'SELECTION.json').exists()
    if a.selected_case is not None:env['SELECTED_CASE']=str(a.selected_case)
    cmd+=[str(r/'source/experiments/kl_mode_missing_search_1d/job.sbatch')]
    job=subprocess.check_output(cmd,env=env,text=True).strip().split(';')[0]
    info=dict(time=time.time(),stage=a.stage,selected_case=a.selected_case,job=job,source_commit=src['commit'],command=cmd)
    record.write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
if __name__=='__main__':main()
