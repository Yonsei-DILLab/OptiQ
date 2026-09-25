# Six-target appendix: final high-L densities and score convergence

User request2026-09-26: main figure keeps two cases; appendix includes all six in the current style. This extension performs frozen-checkpoint evaluation only. No training is launched or resumed.

## Inventory and missing data
All six approved targets: t00_reference,n00_spike_ramp,n07_spike_flat_ramp,t01_two_offset,t05_unequal_mass,t06_minor_mode. All final100K runs complete except t06_minor_mode/reverse_s3; its backup STATUS is82935 and the originalVast SSH endpoint refuses connection. Do not pass it off as a100K checkpoint. The t06 comparison uses matched seeds0–2 in both methods; retain forward_s3 in raw data and explicitly label this exception. Other five use seeds0–3. Display47 available final evaluations in raw tables,46 in paired primary comparisons.

## Score measurements
Reuse the existing8 score evaluations for the main two targets from20260926_kl_paper_highL. Evaluate only the15 newly available final Reverse actors from the Vast snapshot5c918efad2a78f797da6eb928fadc697a691de8c (numerical base41a020577dbf8a87d8f96d714ab3888a4bb82260). Restore from the backed-up original actor source. No parent mutation. Independent MC sample counts2^7...2^24;16 repeats;four additional independent2^24 references;float32 actor and float64 stable ratio accumulator, exactly the previous measurement. Predefined main actions are seed0policy10/50/90% quantiles. All available completedseeds enter raw data and error summaries.128 additionalpolicyprobes and9targetprobes are inconfig. Commit source/config/launch before submission;15oneGPU/twoCPU jobs,maximum8concurrent.

## Figures
N=M128,batch32,100K updates,ReverseL2^20. Actual1M samples perseed,512bins,noKDE. BlueForward/orangeReverse fill, blackdashedtarget,sans-serif,method-only density titles,noTV labels,nooverlapping legend. Density grid3rows4columns; map pairs tocases incaption. Score panels allblue withoriginal10–90% MC shading, dotted independentreference and dashed trainingL2^20. Labelx Number of MC samples L. Zoomed and commony variants, per-casefigures and appendix pages. No grayreferenceband. Include combined density-plus-score overview. Keep mainfigure files unchanged.

## Interpretation and provenance
Report all cases, including weakerforward fitting, remainingMC scoreerror, and any behavior that does not fit the intendedexample. BigLreferenceis empiricalMC,notexact. Preserve checkpoints and samples on dildata,outsideGit. Record parent SHA,source manifest,job IDs,newmeasurement SHA,andseparaterendercommit.
