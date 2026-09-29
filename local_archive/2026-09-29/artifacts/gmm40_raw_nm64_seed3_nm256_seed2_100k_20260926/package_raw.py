from pathlib import Path
import hashlib, json, shutil, zipfile
import numpy as np

root=Path('/Users/yunheechan/Documents/ChatGPT/OptiQ')
out=Path(__file__).resolve().parent
inputs=root/'artifacts/gmm40_dacer_off_nm_4090/inputs'
selected=[(64,3,'gmm40-ibolt-dacer-off-nm64-100k-4seed-4090-20260925'),(256,2,'gmm40-ibolt-nm128-256-512-100k-4seed-4090-20260925')]
manifest={'dataset':'Original GMM40 iBOLT evaluation samples, 100000 actor updates','branch':'direct-gmm-trg','transformations':'None: all original data and metadata files are byte-for-byte copies. No resampling, filtering or coordinate rescaling.','runs':[],'files':{}}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def copy_verified(source,dest,campaign,receipt):
    key=source.relative_to(campaign).as_posix()
    actual=sha(source)
    assert receipt['files'][key]==actual, ('original archive checksum mismatch',key)
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)
    assert sha(dest)==actual
    manifest['files'][dest.relative_to(out).as_posix()]={'sha256':actual,'source_path':str(source),'archive_sha256':receipt['archive_sha256']}

common_target=None
for nm,seed,name in selected:
    camp=inputs/name
    receipt=json.loads((camp/'LOCAL_COPY_VERIFIED.json').read_text())
    run=camp/'results'/f'ibolt_nm{nm}_s{seed}_100k'
    dst=out/f'nm{nm}_seed{seed}_100k'
    cfg=json.loads((run/'config.json').read_text())
    audit=json.loads((run/'update_count_audit.json').read_text())
    assert cfg['method']=='optiq_trg' and cfg['n']==cfg['m']==nm and cfg['seed']==seed
    assert audit['status']=='passed' and audit['actor_updates']==100000 and audit['full_budget_completed']
    for filename in ('config.json','update_count_audit.json','wandb_status.json','metrics.jsonl'):
        copy_verified(run/filename,dst/filename,camp,receipt)
    tpath=camp/'results/target/definition.json'
    target=json.loads(tpath.read_text())
    copy_verified(tpath,dst/'target_definition.json',camp,receipt)
    if common_target is None: common_target=target
    else:
        for key in ('means','std','weights','scale','bounded_mass'):
            assert target[key]==common_target[key]
    means=np.asarray(target['means']); std=np.asarray(target['std'])
    entry={'name':run.name,'N':nm,'M':nm,'seed':seed,'actor_updates':100000,'source_commit':cfg['source_git_commit'],'evaluation':{}}
    for mode,suffix in [('full_policy',''),('mu_only','_mu_only')]:
        ev=run/'evaluations/step_0100000'
        sample=ev/f'samples{suffix}.npy'; metric=ev/f'metrics{suffix}.json'
        a=np.load(sample,allow_pickle=False); m=json.loads(metric.read_text())
        assert a.shape==(10000,2) and np.isfinite(a).all() and (np.abs(a)<=40.0001).all()
        d=((a[:,None,:]-means[None,:,:])**2).sum(axis=2)/std[None,:]**2
        labels=d.argmin(axis=1); near=d[np.arange(len(a)),labels]<=9.0
        counts=np.bincount(labels[near],minlength=40)
        np.testing.assert_array_equal(counts,m['mode_counts_3sigma'])
        assert abs(near.mean()-m['high_density_fraction'])<1e-12
        coverage=int((counts>=np.asarray(m['coverage_threshold'])).sum())
        assert coverage==m['mode_coverage']==40
        copy_verified(sample,dst/sample.name,camp,receipt)
        copy_verified(metric,dst/metric.name,camp,receipt)
        entry['evaluation'][mode]={'samples':str((dst/sample.name).relative_to(out)),'shape':list(a.shape),'dtype':str(a.dtype),'coordinates':'physical x1,x2 in [-40,40]; already multiplied by 40','mode_coverage':coverage,'high_density_fraction':m['high_density_fraction'],'min_mode_count_3sigma':int(counts.min()),'counts_and_near_recomputed':True}
    manifest['runs'].append(entry)

readme='''# OptiQ / iBOLT GMM40 원시 샘플

100,000 actor optimizer updates 시점에 실제 저장된 원본 평가 데이터입니다.
재학습·재샘플링·필터링·정규화·순서 변경 없이 원본 파일을 그대로 복사했습니다.

- `nm64_seed3_100k/`: N=M=64, seed 3. 저장된 두 평가 모드 모두 coverage 40/40.
- `nm256_seed2_100k/`: N=M=256, seed 2. 저장된 두 평가 모드 모두 coverage 40/40.
- N=M=64는 요청에 따라 40/40인 seed 3을 선택했습니다. 전체 seed 평균을 나타내는 자료가 아닙니다.

각 폴더:
- `samples.npy`: fresh random latent + conditional sigma를 포함한 full-policy 원본 샘플.
- `samples_mu_only.npy`: fresh random latent, conditional sigma noise를 제거한 mu-only 원본 샘플.
- 두 배열은 각 (10000, 2), float32입니다. 각 행은 시각화 전 물리 좌표 (x1,x2)이며 이미 x=40*a 변환이 적용되어 있습니다. 다시 40을 곱하지 마세요.
- `metrics.json` / `metrics_mu_only.json`: 각각 해당 샘플의 원래 평가 지표.
- `target_definition.json`: 40개 GT 중심, 표준편차, 혼합 가중치, 범위 정의. GT 표준편차는 actor sigma와 다릅니다.
- `config.json`: 원래 실행 설정과 학습 소스 SHA.
- `update_count_audit.json`: 100k optimizer update 완료 증명.
- `metrics.jsonl`: 원래 저장된 학습/평가 지표 이력.
- `wandb_status.json`: 원래 W&B run 식별 정보.

공통 설정: 256x3 GELU, batch256, Adam3e-4, T1, beta1, random Gaussian latent,
mean-head variance scale16, actor log sigma[-5,-3.5], 초기-4, teacher-only sigma floor .05.
고정 Q=log p_GMM40를 사용하는 분포 모사 실험이며 critic/replay 학습은 없습니다.

Coverage는 기존 기준(가장 가까운 표준화 GT 중심의 3-sigma 안 샘플 수가
metrics의 coverage_threshold 이상인 성분 수)입니다. 원시 좌표로 counts/coverage를 재검증했습니다.
SHA256SUMS.txt와 MANIFEST.json에 원본 보관본과의 바이트 일치 검증을 기록했습니다.

읽기 예시 (이 폴더에서 실행):
```python
import numpy as np
x64 = np.load('nm64_seed3_100k/samples.npy', allow_pickle=False)
x256 = np.load('nm256_seed2_100k/samples.npy', allow_pickle=False)
mu64 = np.load('nm64_seed3_100k/samples_mu_only.npy', allow_pickle=False)
mu256 = np.load('nm256_seed2_100k/samples_mu_only.npy', allow_pickle=False)
```
'''
(out/'README_KO.md').write_text(readme)
(out/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
files=sorted(p for p in out.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS.txt','package_raw.py') and p.suffix!='.zip')
(out/'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(out).as_posix()}\n' for p in files))
files.append(out/'SHA256SUMS.txt')
zpath=out/'gmm40_optiq_nm64_seed3_nm256_seed2_100k_raw.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(out))
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None
    for p in files:assert hashlib.sha256(z.read(p.relative_to(out).as_posix())).hexdigest()==sha(p)
print(json.dumps({'zip':str(zpath),'bytes':zpath.stat().st_size,'zip_sha256':sha(zpath),'runs':manifest['runs']},ensure_ascii=False,indent=2))
