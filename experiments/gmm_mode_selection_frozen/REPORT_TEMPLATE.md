# Frozen Q: Direct GMM vs mode selection, batch128

Status: pending. Report actual completed/168 runs, timestamp and commit. No expected result is a measured result.

## 0. Setting and how to read figures

Include PROTOCOL hyperparameters, exact Q functions and mode boundaries. Explain histogram TV and basin TV BEFORE results. Describe actual32768action histograms,512bins,no smoothing,seed/step labels; known-basin oracle routing; both methods receive same training budget.

## 1. Density overview

Q/exact-target overview; seven N×M rows by three Q columns. Compare both final histograms to exact target. Show seed0 for overview with explicit selection rule; all four seeds separately. Add early/late tracking snapshots.

## 2. Speed and final accuracy

Histogram and basin TV vs updates and measured train wall time, allseeds; backup error; completion counts. Include diagnostic overhead separately. Report uncertainty and disagreements amongseeds.

## 3. Teacher and assignment

Proposal → weightedteacher → actor at selected measuredsteps. Original-order and sorted effective GMM assignment J, responsibility H. Explain normalization axes; no OT matrix exists here. Representativegroup0 must not be labelled fullbatch128. Show batchmean teacher basinmass too.

## 4. Gradient and latent movement

Baseline mode-gradient cosines; retained gradientmass; fixed-latent mean/sigma/specialistfractions. Same checkpoint/Adam/RNG paired counterfactual update: which othermode latents move? Distinguish output routing from interference through shared weights.

## 5. Conclusions and limits

Draw conclusions from complete pairedseeds. Mode selection also changes gradient magnitude and uses knownbasin labels; cannot claim automatically generalizes to RL. Rawrun/manifest/checkpoint locations.
