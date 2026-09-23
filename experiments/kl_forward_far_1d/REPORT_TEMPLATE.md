# Forward KL: distant three-mode target

## Completion and main finding
Report all16 run statuses, final three-peak pass counts and first observed passing saved updates. Distinguish basin visitation, core mass, and genuine local fitting.

## Settings
Explicitly give centers(-10,0,10),width0.1,box[-20,20],mu20*tanh(head),mean initializer1,sigma bounds unchanged,batch32,N=M64/128/256/512,4seeds,20K,tau0.25. State that this changes geometry, not just initialization.

## Figures
Actual32768-sample4096-bin full histograms, plus each mode zoom; allseeds and training curves. No KDE or policy integration.

## Metrics
HistogramTV,basinTV,W1,per-mode core mass,three-peak separation criterion and timing. Explain criterion before tables; include independent proposal/weighted-teacher diagnostics when needed to interpret failures.

## Provenance
Full training commit,run IDs,source manifest,validation,storage paths and reproducible report code. Results reflect this fixed-Q setting only.
