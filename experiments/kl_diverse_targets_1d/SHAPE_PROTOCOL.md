# Image-inspired narrow peak + broad shoulder candidates

User steering (2026-09-25): add a target resembling the supplied figure: a narrow,
high left peak plus a broad, lower right shoulder ending at an edge. This is a
qualitative target-density construction, not a reproduction of the unidentified
paper or an assumption about whether its green curve denotes density or reward.

## Exact target
Use the existing mixture oracle and the SAME actor/optimizer/proposal/settings.
On [-10,10], let q_k(a)=Normal(a;c_k,h_k^2)/Z_k, where
Z_k=Phi((10-c_k)/h_k)-Phi((-10-c_k)/h_k). Define
p*(a)=eta_1 q_1(a)+eta_2 q_2(a), with eta_1+eta_2=1.
The configuration stores box_component_masses=eta, and whole-Gaussian coefficients
rho_k=(eta_k/Z_k)/sum_l(eta_l/Z_l) in target_masses so that the existing oracle
Q(a)=.25 log(sum rho_k Normal(a;c_k,h_k^2)) has exactly this conditioned target.
The density is supported only on the existing action box. The broad center 9.5 is
near its right edge 10; conditioning creates the sharply cut edge without inventing
an interior discontinuity, zero-density gap or modifying action gradients.
There is a small decrease from9.5 to10: this is an approximation of the pictured
rising shoulder, not an exact trace. All target densities and numerical settings
are shown in the candidate report.

shape_config.json specifies eight candidates: narrow std .15/.2/.25, broad std
1.5/2.5/3.5/4.5, narrow mass .15/.2/.5, narrow center -4.25/-2.5, broad center
3/7.5/9.5. The interior broad Gaussian is a control for the right-edge construction.
No per-candidate actor tuning: mean-head initializer scale3, logsigma[-5,-1],
N=M128,batch32,temperature.25,Adam3e-4. Every method starts from its paired seed's
same random initial actor. Initial broad coverage is not supervised.

## Execution and selection
Follow PROTOCOL.md: screen seeds0,1 for100K updates, Reverse L1024. Held-outseeds2,3
only for candidates meeting BOTH initial-seed gates. Four-seed pass for shortlist;
retain failures and no-gap/reversed cases. At most6 environments are proposed
across both the original and image-inspired families. L1048576 is NOT authorized
and must not be submitted until the user reviews actual candidate figures.

At 100K use2^18 actual actor samples/seed,512bins,noKDE. Report target analytic
curve, forward/reverse histograms, TV, core/basin mass and per-seed missing modes.
Core/basin gates are unchanged. Use exact target-CDF masses, including boundary
normalization. Mean plots do not replace per-seed panels (different missed modes
can cancel in the average). Report broad-shape errors even if both modes survive.

## Provenance
Separate immutable campaign kl_sharp_broad_targets_20260925 with --config
shape_config.json. Existing original-family source/campaign is not overwritten.
Commit source/config/protocol/launch before execution onheejoon. CPU target math,
old f05 gradient parity, resume; then GPU preflight; thenSlurm jobs. Source and
checkpoints backed up todildata via already authorized read-only route.
