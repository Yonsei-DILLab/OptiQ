# Existing fixed64 latent prior: bounded v3/v4 screens

The active goal permits reward/hyperparameter experiments with the algorithm
structure unchanged. Continuous random-latent controls keep losing successful
route diversity; broader teacher candidates improve v3 majority acquisition but
do not establish a minority solution. v4 contact instrumentation showed movement
with little net progress, not sustained wall contact. These observations do not
identify the cause. Test whether the existing finite64 configuration reduces
changing-latent marginal approximation variability enough to help specialization.

Register two fresh seed0 runs on independent free199 slots. Compare v3 against
the completed d259301 teacherfloor1/T3/normalized-geodesic/gamma.999 control;
compare v4 against555bb7e teacherfloor.5/T1/original-geodesic/gamma.999. Change only
the existing latent_prior=finite, latent_components=64, codebook_seed=20260911
configuration. The codebook is the existing centered/scaled Gaussian draw.
Every training/evaluation action samples a uniform index independently; no
episode-persistent latent, route labels, goal conditioning or extra evaluation
noise. Training distillation uses all64 fixed components through existing code.
N=M64, actor sigma[-5,-1]/initial-1,256x3GELU, meaninit1, Adam3e-4/5e-4,tau.005,
replay1M,batch4096,256env,8updates/256,warmup8192,H/d+.7/500,NovelD OFF and every
other corresponding control parameter stay unchanged. No reward/physics edit.

Pin the seven existing computational files plus latent.py and box_gaussian.py
byte-for-byte to2564b59. Adapter forwarding, configuration verification and
accurate evaluation labels may change; loss, TD, optimizer, network and sampling
implementations may not. Omitted profile keeps the previous normal-prior path.
The finite-prior deterministic control is component0_mu, not z=0: label its
folder, NPZ, summary and W&B metrics accordingly. Direct-policy evaluation keeps
conditional sigma; mu-only native evaluation uses the same fixed codebook.
The unused standalone MuJoCo dual_mu_eval flag must be false for finite-prior
validation. AntMaze uses its own evaluator and continues both native and direct
modes. Record this evaluation compatibility flag separately from learning changes.

Commit/push/share/freeze before real8448transition/8update batch4096 preflights.
Check initial actor/critic parameters against the corresponding old controls,
full replay reward readback, fixed-codebook/stratified component identity,
behavior/direct/native sampler identity, serialization roundtrip and RNG/model
preservation. Final codebook must equal initial. Record actual source/config/PID,
not just registration. Preserve current Q on180 and all old sources/results;
no4090, cancelled resumes, extra seeds or automatic restarts/extensions.

Both budgets258304total/7816updates (250112 post-warmup),40direct+40mu-only
each50k,100final per mode and full checkpoint. Native700/original identical full
starts and goal radius/termination unchanged. Report all failures. Both routes
must reach goals before this is an acquisition candidate; entries or action
diversity alone do not meet the goal. Retention requires a separately justified
longer run and cannot be inferred from250k or one training seed.
