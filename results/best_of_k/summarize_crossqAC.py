import json, glob, numpy as np
for task in ["humanoid", "walker2d"]:
    R = [json.load(open(f)) for f in sorted(glob.glob(f"results/best_of_k/crossqAC/ibolt_{task}-*.json"))]
    print(f"== {task}  n={len(R)} seeds, {R[0]['episodes']} episodes each")
    for m in ["z0", "k1", "k4", "k8", "k16", "k32"]:
        per = [np.mean(r["modes"][m]) for r in R]
        print(f"  {m:4s} {np.mean(per):7.0f} ± {np.std(per):4.0f}   seeds: {' '.join(f'{x:.0f}' for x in per)}")
