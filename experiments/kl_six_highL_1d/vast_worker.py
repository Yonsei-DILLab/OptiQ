"""One locked GPU worker for explicitly migrated, independent case/seed jobs."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();root=a.root.resolve()
    cfg=json.loads((root/'VAST_ALLOCATION.json').read_text());indices=cfg['workers'][str(a.gpu)]
    assert json.loads((root/'runtime/GPU_VALIDATION_5090.json').read_text())['passed']
    assert cfg['login4_migration_confirmed'] and len(indices)==len(set(indices))
    source=json.loads((root/'SOURCE_MANIFEST.json').read_text())
    for name,h in source['files'].items():assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==h,name
    lock=(root/f'runtime/gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=sorted(os.sched_getaffinity(0));chosen=cpus[a.gpu*4:a.gpu*4+4];os.sched_setaffinity(0,chosen or cpus[:4])
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),JAX_PLATFORMS='cuda',JAX_THREEFRY_PARTITIONABLE='false',
      XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
      PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',OPTIQ_FORWARD_EXT=str(root/'forward_parents'),XLA_FLAGS=cfg.get('xla_flags',''),
      JAX_COMPILATION_CACHE_DIR=str(root/'runtime/jax_cache'),JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='2')
    for name in ['LD_LIBRARY_PATH','PYTHONPATH','JAX_DEFAULT_MATMUL_PRECISION']:env.pop(name,None)
    status=root/f'runtime/worker{a.gpu}.json'
    for index in indices:
        event=dict(gpu=a.gpu,index=index,source_commit=source['commit'],time=time.time(),state='running',worker_pid=os.getpid())
        write(status,event)
        with (root/f'runtime/vast_index{index:02d}.log').open('a') as log:
            rc=subprocess.call([sys.executable,'-u','-m','experiments.kl_six_highL_1d.run','--root',str(root),'--index',str(index)],cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        case=cfg['cases'][index//4];done=root/'runtime/confirm'/case/f'reverse_s{index%4}'/'COMPLETE.json'
        event.update(time=time.time(),returncode=rc,state='complete' if rc==0 and done.exists() else 'failed')
        with (root/'runtime/VAST_EVENTS.jsonl').open('a') as log:log.write(json.dumps(event)+'\n')
        write(status,event)
        if event['state']=='failed':raise SystemExit(f'Index {index} failed; later jobs on this GPU are preserved for inspection')
    write(status,dict(gpu=a.gpu,indices=indices,state='all_complete',time=time.time()))
if __name__=='__main__':main()
