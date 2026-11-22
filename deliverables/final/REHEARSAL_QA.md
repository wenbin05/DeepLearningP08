# Rehearsal questions

**What is the contribution if you did not invent a new method?**
A controlled architectural ablation. The three main placements have exactly the same trainable count, data and optimizer settings, which makes their observed differences easier to interpret. We do not claim a new LoRA algorithm.

**Did cross-attention beat BART?**
It had the highest mean scores among the three isolated placements on CNN/DailyMail. Its ROUGE-1 was 38.56 versus 38.61 for unadapted BART, so it did not improve that baseline on this metric. Cross-attention's ROUGE-L was slightly higher. We do not claim statistical significance.

**Why is the unadapted model competitive?**
It already has denoising pretraining and can produce source-overlapping text. ROUGE rewards overlap. Short adaptation on 1,024 examples need not improve that baseline. These are plausible interpretations; the study does not isolate their causal contributions.

**Were training conditions identical?**
For the three main placements, yes in the declared hyperparameters, subsets and update budget. Full FT uses a lower learning rate and smaller microbatch after a memory failure. All adapted methods have the same nominal effective batch and 96 updates, but FLOPs and wall time are not equalized.

**Why not claim that a larger rank is worse?**
Only three ranks were tested under one short schedule. Alpha is fixed, so alpha/r also changes. The observed means are not monotonically increasing, but this does not establish a universal rule about rank.

**Does a 0.106% trainable fraction mean 0.106% of the computation or memory?**
No. It counts optimized parameters relative to total parameters including adapters. Backbone weights, activations, generation and gradient propagation still incur costs.

**Why did Prefix-Tuning perform badly?**
The saved outputs establish poor performance under this configuration. They do not determine the cause. The 20-token prefix baseline was not given a separate hyperparameter or convergence study, so it would be premature to reject the method in general.

**Why are XSum scores lower?**
The model was trained only on CNN/DailyMail, and XSum has a different summary style. The same 56-token minimum and other decoding settings were retained. This makes transfer and decoding mismatch plausible factors; no experiment separates their effects here.

**Why do all human packets contain the same articles?**
A shared set of five articles and four configurations lets us compare model outputs and measure agreement on the same items. Model identity is hidden and item order varies. Randomizing order does not create independent examples.

**Does 58% versus 54% prove better factuality?**
No. Those proportions are 29 versus 27 consistent ratings out of 50 per configuration, across just five outputs. Fleiss' kappa is 0.044. The ratings are dependent and agreement is low, so the comparison is descriptive.

**How were the metrics verified?**
The final checks validated manifests, required artifacts, seed coverage and protocol compatibility, and recomputed official stemmed macro-F1 ROUGE from saved predictions against original references. ROUGE-L was used, not ROUGE-Lsum.

**What did AI do?**
OpenAI Codex assisted with corrections, local CPU experiment orchestration, verification, analysis drafts, figures and writing. Academic-humanizer guided writing clarity. Results came from executed jobs; human judgments were team-supplied. The team must review and own the final interpretation as the guide requires.

**What would you do next?**
Repeat the exact-budget placement study with a larger fixed training subset and paired article-level uncertainty. Separately validate XSum length settings. Improve the human rubric through calibration and recruit ratings on more articles before estimating factuality differences.
