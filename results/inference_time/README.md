# Inference latency (Table 2, rebuttal)

Final numbers: `bench_interleaved.py` → `bench_interleaved_5090.txt`.
vast.ai RTX 5090 (GPU otherwise idle), AMD EPYC 9654, jax 0.4.33 for every method. Humanoid-v4, batch 1.

Protocol
- Same call convention for every method, as in SBX: the jitted action function takes the actor
  TrainState and the observation; stochastic modes split the PRNG key inside the same jitted call
  (one device dispatch per action); the action returns as numpy (this synchronizes the device), then
  clip + unscale. iBOLT uses the repo's `IBOLTPolicy._predict`, i.e. exactly the evaluation call.
- 5 seeds (network init + observation stream from a seeded random rollout; latency does not depend on
  trained weights). Every method/mode is compiled and warmed up (300 calls) before timing; compile
  time never enters the latency.
- Per seed 30 blocks; each block visits all methods/modes in a fresh random order and times 100 calls,
  so drift and order effects cancel. ± = std across the 5 seed means.

| method | eval action | latency (ms) | Δ vs iBOLT z=0 (ms) | GPU compute (µs) |
|---|---|---|---|---|
| iBOLT | z = 0 (paper eval) | 0.208 ± 0.012 | 0 | 3.83 |
| iBOLT | z ~ N(0, I), center | 0.228 ± 0.012 | +0.019 | 7.55 |
| iBOLT | z ~ N(0, I) + conditional σ (full policy) | 0.230 ± 0.013 | +0.021 | 8.77 |
| SAC | tanh(mean) | 0.211 ± 0.012 | +0.003 | 3.81 |
| SAC | sampled | 0.217 ± 0.013 | +0.009 | 6.31 |
| TD3 | deterministic | 0.210 ± 0.012 | +0.002 | 3.83 |

GPU compute = 2,000 sequential actions inside one jitted `lax.scan`, divided by 2,000.

Reading
- At batch 1 the latency (~0.21 ms) is host-side dispatch and transfer; the network itself costs
  ~4 µs. iBOLT z=0, SAC and TD3 have the same GPU cost: the extra latent input (17 dims) is negligible.
- Sampling adds the PRNG work on the GPU (+3.7 µs for z, +1.2 µs more for σ) and a key in/out of the
  jitted call; the full stochastic iBOLT policy is 0.02 ms (≈10%) slower than z=0.

What was wrong before (do not use the older numbers)
1. The repo's `IBOLTPolicy._predict` split the PRNG key in a separate dispatch before the action,
   even for z=0: +0.15 ms (0.36 ms). Fixed: z=0 uses no key; stochastic modes split inside one jitted
   call (`_act_and_advance`), giving the same actions and key stream as before (tested).
2. A first comparison timed the modes one after another per seed and gave SAC/TD3 parameters as
   jit constants while iBOLT passed its TrainState (26 leaves incl. optimizer state, ~0.015 ms of
   argument handling). That made sampled iBOLT look faster than z=0. The interleaved, same-convention
   benchmark above replaces it.
3. SBX itself (jax 0.10.2) is slower for every method from the newer JAX dispatch (SAC det 0.295,
   TD3 0.282, SAC sampled 0.524 ms; `latency_sbx_*.json`). Compare methods within one JAX version.
