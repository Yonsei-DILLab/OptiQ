# Forward128x128: target width0.5 versus previous width1

User requested lowering the target Gaussian width to0.5. Width is target standard deviation,not actor conditional sigma. New independent runs,fresh seeds0–3,100K updates,N=M128,centers(-5,0,5),equal target weights,action and mean box[-10,10],mu10*tanh(head),mean initializer variance scale1,batch32,temperature0.25,Adam3e-4. Preserve the completed width1 study and all earlier studies.

Compared to kl_forward_wide_1d@5b23b3a30361082936073f50804488401deaf1a4,the only numerical training change is target width1->0.5 in f(a),with Q(a)=0.25 log f(a). Config differences are study label and target_width only. Reuse the exact actor,box Gaussian sampler/density,and forward marginal NLL modules. Keep log sigma[-5,-1],initial-1,proposal floor exp(-5),256x256 GELU,1D normal latent and zero state. Validate identical initial actor/Adam/RNG state and sampled actions for matched seeds across the two target widths. Training trajectories then diverge because Q/importance weights differ.

Forward teacher:draw N latent conditionals and M IID samples from their uniform mixture;weight by softmax(Q/tau-log proposal);stop teacher candidates/weights;optimize weighted student marginal NLL. No new OT,mode masks,teacher target samples,exploration injection or critic.

Keep the same evaluation schedule0,100,500,1K,2K,5K,7.5K,10K,then every5K through100K. Full checkpoint every500 updates. Evaluate32768 actual action samples and512-bin unsmoothed histograms over[-10,10];bin width0.0390625 resolves width0.5 peaks. Same basin boundaries(-10,-2.5,2.5,10). Per-mode zooms use center+-3target standard deviations. Exact target is normalized over[-10,10].

The previous width-relative three-peak rule now uses center+-0.5 core windows and midpoint+-0.5 valley windows:all cores hold>=half the analytic target core mass,and both valley/core ratios<=0.5. Verify the target itself passes. A passing diagnostic does not prove an exact fit;report histograms,TV,W1,core mass and basin mass. Evaluation has independent RNG and never changes training RNG.

Validate config-diff scope,target normalization,three maxima,energy formula,finite gradient,unchanged training implementation after resolving imports and WIDTH,identical initial state for all4seeds. Commit exact source/config/job/protocol to heejoon before launch. Four independent Slurm jobs,1GPU2CPUs16GB each,max4concurrent,one-hour walltime. Record source hashes and job IDs. Preserve full checkpoint and source/results on dildata.

Remote:login4:/scratch2/hobbit9882/OptiQ-SingleQ-N64-M256-K64-T025-20260921/extensions/forward_width05_20260923.
Central:dildata:/data1/heejoonorm/OptiQ/studies/20260923_forward_width05/campaign.
Report destination:reports/20260923_forward_width05.
