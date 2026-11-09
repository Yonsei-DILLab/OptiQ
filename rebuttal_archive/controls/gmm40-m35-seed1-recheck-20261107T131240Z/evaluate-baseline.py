"""Coverage-only diagnostic of the frozen original seed-1 checkpoint."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/"manifest.json").read_text())
sys.path.insert(0,manifest["worktree"])
from gmm40.learner import GMM40Learner
from gmm40.target import Target
from gmm40.metrics import assignments

baseline=Path(manifest["baseline_output"])
target=Target()
agent=GMM40Learner(target,seed=1,n=256,m=256,batch=256,teacher_std_floor=None)
checkpoint=baseline/"checkpoints/step_0100000.bin"
agent.restore(checkpoint)
assert agent.updates==int(agent.state.step)==100000
training_key=np.asarray(agent.key).copy()
reference=np.load(baseline/"reference.npy",allow_pickle=False)
_,_,reference_counts=assignments(reference,target)
threshold=np.maximum(10,.1*reference_counts)
original=json.loads((baseline/"evaluations/step_0100000/metrics_full_policy.json").read_text())
missing=sorted(set(range(40))-set(original["covered_modes"]))

def evaluate(seed):
    full,_,extra=agent.evaluate_samples(10000,seed)
    result={"evaluation_seed":seed}
    arrays={"full_policy":full,"mu_only":extra["mu_only"]}
    for mode,points in arrays.items():
        _,_,counts=assignments(points,target)
        result[mode]={"coverage":int((counts>=threshold).sum()),
                      "counts_3sigma":counts.tolist(),
                      "original_missing_component_counts":{str(i):int(counts[i]) for i in missing}}
    return result,arrays

replay,replayed=evaluate(900001)
for mode in ("full_policy","mu_only"):
    old=np.load(baseline/f"evaluations/step_0100000/samples_{mode}.npy",allow_pickle=False)
    replay[mode]["samples_exactly_match_original"]=bool(np.array_equal(replayed[mode],old))
    replay[mode]["maximum_absolute_difference"]=float(np.max(np.abs(replayed[mode]-old)))
records=[]
for seed in range(900101,900111):
    result,_=evaluate(seed);records.append(result)
    print(json.dumps({"eval_seed":seed,**{mode:result[mode]["coverage"] for mode in ("full_policy","mu_only")}}),flush=True)
pooled={}
for mode in ("full_policy","mu_only"):
    counts=np.sum([r[mode]["counts_3sigma"] for r in records],axis=0)
    total=10000*len(records)
    limits=np.maximum(10,.1*reference_counts*total/len(reference))
    pooled[mode]={"samples":total,"coverage":int((counts>=limits).sum()),
                  "counts_3sigma":counts.tolist(),"threshold":limits.tolist(),
                  "original_missing_component_counts":{str(i):int(counts[i]) for i in missing}}
assert np.array_equal(training_key,agent.key)
report={"code_revision":manifest["code_revision"],"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_seed":1,"training_updates":100000,"log_std_bounds":[-5,-3.5],"proposal_floor":None,
        "original_missing_components":missing,
        "missing_component_centers":{str(i):target.means[i].tolist() for i in missing},
        "original_evaluation_replay":replay,"independent_10k_evaluations":records,
        "pooled_100k_diagnostic":pooled,
        "note":"Diagnostic resampling only; original benchmark evaluation and training are unchanged."}
(ROOT/"evaluation-diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"pooled":{mode:{"samples":v["samples"],"coverage":v["coverage"],"missed_counts":v["original_missing_component_counts"]} for mode,v in pooled.items()}}),flush=True)

