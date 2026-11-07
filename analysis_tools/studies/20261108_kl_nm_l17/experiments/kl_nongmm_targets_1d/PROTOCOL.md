# Non-GMM target density search (2026-09-25)

The user's correction is authoritative: the target itself must be non-Gaussian,
not a differently weighted Gaussian mixture. The earlier image-inspired GMM round
was cancelled and its results are archived separately; it is not a substitute for
this round. Only the target changes. The semi-implicit Gaussian-mixture actor and
validated forward/reverse updates remain unchanged.

## Explicit non-GMM formulas
Let S(x)=sigmoid(x). Use these positive unnormalized shapes on a∈[-10,10]:

1. Logistic spike: f(a)=S((a-c)/h)S(-(a-c)/h)/h.
2. Tilted shelf: f(a)=S((a-l)/e_L)S((r-a)/e_R) exp(beta*(a-(l+r)/2)).
   beta=0 gives a nearly flat plateau with independently chosen edge sharpness.
   beta>0 gives a rising, asymmetric ramp ending at a sharp right edge.
3. Rippled shelf: the shelf above times [1+u*cos(omega*(a-c))],0<u<1.

Normalize each explicit shape by Z_k=∫f_k(a)da, then define
p*(a)=sum_k eta_k f_k(a)/Z_k, Q(a)=0.25 log p*(a).
These are NOT Gaussian components and no GMM is fitted to the target. The target
score is obtained by analytic automatic differentiation of this explicit log
formula, not from target samples, density estimation, a neural critic, or a GMM
approximation. Smooth sigmoid edges preserve the sharply cut visual appearance
while retaining a well-defined, finite reverse-KL action gradient. No zero-density
barrier or added probability floor is used. The only hard boundary is the same
[-10,10] action box used in the previous experiment.

## Eight candidate targets
config.json is authoritative. n00 matches the requested narrow left spike plus
broad rising right lobe. n01 changes their mass ratio. n02 has3flat plateaus;
n03 has3unequally sloped shelves; n04 has a narrow spike between2broad shelves;
n05 has4plateaus; n06 is ONE corrugated shelf with5peaks (not a mixture of5bumps);
n07 combines a spike, flat plateau and asymmetric ramp. Show the actual target
curves before interpreting the outcomes. Do not call the unidentified source
figure an exact replication or assume that its green curve necessarily denotes
a density rather than reward.

## Fixed learning settings
N=M128, batch32,100K fresh updates,Adam3e-4,temperature.25,mean-head initializer
scale3,256×256GELU,z~Normal(0,1),actorlogsigma[-5,-1],initial-1,TRG action[-10,10].
Forward uses the original importance-weighted marginalNLL. Reverse uses the same
pathwise score update with independent resampled density bankL1024. Neither
receives target component labels or direct target samples. Unused inherited
fields target_centers/target_width are constructor compatibility metadata ONLY;
log_f and target_score are completely overridden by the explicit non-GMM oracle.
Paired methods use the same initialization and GPU. No per-target actor tuning.

## Evaluation and selection
100K:262144 sampled actions/seed,512bins,noKDE. Intermediate evaluation32768.
Target normalizers:float64 adaptive quadrature. CDF:262145-point cumulative Simpson,
independently checked against adaptive quadrature on ALL512histogram bins with
max absolute mass tolerance2e-7. Geometry: actual density peaks and intervening
minima, verified against expected_modes before launch. Plateaus are rounded smooth
lobes, so they have well-defined local peaks. Core intervals are declared in config
before outcomes and cover a peak/plateau's interior (not an imaginary Gaussianstd).
Basin and core masses use the targetCDF.

Missing:both core and basin mass<25%of target. Allpeak recovery:each core and basin
>=50%of target, plus valley/core<=max(.5,2*target valley/core). Forward pass also
requires512-binTV<=.15. Firstscreen seeds0,1. Only both-seed pass pairs proceed to
fresh held-outseeds2,3. Final shortlist requires all4seeds and includes at most6
cases combined with the original GMM round. Preserve all failures,no-gap,reversed
outcomes. This is selected illustrative evidence; it is not a general benchmark or
a proof that reverseKL must miss modes. Report actual errors, never perfectfit.

## Approval boundary and provenance
NoL1048576 training before the user reviews and approves candidates. Code has no
high-L launch path. Commit all source/config/launch/protocol toheejoon before
execution; record fullSHA+sourcehashes+jobIDs. Separate immutable nonGMM campaign,
Slurm1GPU2CPU perpairedseed, up to8concurrent per accountallocation. Backup via
approved login4-to-dildata read-only route. Do not resume cancelled mistaken GMM jobs.
