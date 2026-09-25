# 3-GMM Forward-KL loss landscapes and initialization basins

All targets and fitted models have equal weights 1/3, common fixed sigma
0.5, and three trainable means. The loss is the same float64 population
NLL as gmm3_mean_local_minima, displayed after subtracting the target's
own NLL, so the plotted quantity is Forward KL. No variance/weight
training, truncation, SNIS, neural network, or SGD noise is introduced.

## Targets

Reuse all nine targets from the previous study: (-R,R,D), R in {1,1.5},
D in {4,5,6,8}, plus (-4.25,0,4.25). Saved stationary references in
config.json come from the previous independently checked analysis, not
new training in this study. Where no bad minimum was found, use the old
bad-structure initialization as a plotting reference and label it as such.

## Views, and what they do and do not establish

1. Conditional sections: x=mu1, y=mu2, color=KL, holding mu3 fixed.
   An 81x81 grid is evaluated at 13 regular mu3 values plus the exact
   rightmost true mean and the saved reference mean. An offline HTML
   slider will expose these sections. Duplicate wells due to label
   permutations are equivalent densities. A minimum in a section alone
   is not evidence of a full-3D local minimum.
2. A plane containing both reference b and the global solution c:
   mu=b+u(c-b)+v*d, where d is the normalized projection of (0,-1,1)
   perpendicular to c-b. Thus (u,v)=(0,0) is the reference and (1,0)
   is the true mean vector. Both a broad view and a local zoom are saved.
   Three-dimensional surface plots use (u,v,KL), not (mu1,mu2,mu3).
3. Straight path b+t(c-b), t in [0,1]. Its maximum above KL(b) is the
   barrier ALONG THIS LINE, not the minimum barrier among all paths.
4. Initialization-basin maps for the symmetric target, (-1.5,1.5,4),
   (-1,1,5), and (-1.5,1.5,6). Start mu3 at the rightmost target mean;
   mu1<=mu2 are the 153 ordered combinations from a 17-point grid.
   During training ALL THREE means are free. GD lr .01 runs up to 100K,
   stopping each trajectory when gradient norm <=1e-7. Four classes:
   global solution; stationary strict bad minimum; stationary saddle;
   unresolved within budget. Hessian and gradient are checked in full 3D,
   with doubled-grid verification at the endpoints. Exact equality of
   components may preserve symmetry and end at a saddle.
5. A thin-basin zoom for (-1.5,1.5,6): initial (a,a+delta,6), with
   a on 11 grid points in [5,5.75] and log10(delta) on 17 points from
   -12 to log10(.24). This examines successful near-coincident starts
   that an ordinary coarse grid can miss. No claim about a continuous
   basin boundary or probability-one failure follows from a finite grid.

## Numerical validation and execution

Use the previous 4097-point Simpson integral over target extremes plus
10 sigma, and check against 8193 points. Validation includes 24 random
parameter vectors plus reference/true vectors for three representative
targets, including the most separated one. Every endpoint gradient and
Hessian is rechecked on the finer grid.

There are 14 independent CPU jobs: nine surfaces, four basin maps,
one near-tie basin zoom. Two CPU cores per job; existing GPU jobs stay
untouched. Commit source, config, launch scripts and this protocol to
heejoon before execution. Record commit SHA and hashes in the manifest,
and job IDs with each task. Preserve source snapshots and store outputs
on dildata. Standard Matplotlib PNG/PDF figures and a self-contained
Markdown report will accompany an offline interactive HTML slice viewer.
