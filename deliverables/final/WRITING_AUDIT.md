# Writing audit before final drafting

Sources: uploaded proposal, midterm report, presentation handout, capstone guide, instructor feedback, verified final JSON summaries, raw prediction records, confirmed survey analysis.

This is a new final report, not a cosmetic rewrite of the midterm. The main structure follows the guide's final-report requirements. The midterm's pilot values remain historical; they are not substituted for held-out final results.

| Location / issue | Evidence status | Planned treatment |
|---|---|---|
| Proposal literature review, one paragraph per paper | Instructor requested synthesis | Organize around generation, adaptation-budget allocation, and factuality; distinguish this study from prior work. |
| Proposal hypothesis about placement and transfer | Hypothesis, not established outcome | Compare exact 147,456-parameter placements; qualify findings by dataset and metric. |
| Midterm remaining-work section | Superseded by 39 completed runs and 78 test evaluations | Replace prospective claims with final results, keeping subset limits explicit. |
| “Strictly matched training” across all methods | Supported only for main placements | Separate full-FT learning rate and microbatch exception; distinguish equal updates from equal FLOPs or wall time. |
| ROUGE as evidence of factuality | Unsupported | Explain lexical overlap; use human judgments only descriptively. |
| Human ranking | 5 articles, 20 summaries, 10 raters; kappa 0.044 | Report counts and dependence; do not claim a reliable factuality winner. |
| Best rank / target projection | Fixed alpha=16 means alpha/r changes with rank; parameter counts differ | Report confounds; do not imply a pure rank or placement effect. |
| Prefix weakness | Low observed scores in one short-budget setup | Report failure honestly; no general claim that Prefix Tuning is ineffective. |
| Prior contribution percentages | Earlier team declaration is 50/50 | Retain as declared, request confirmation of actual responsibilities. |
| AI usage | Assistant implemented fixes, launched local jobs, analyzed outputs, drafted artifacts | Disclose the actual assistance. The guide requires students' own analysis and interpretation, so author review and ownership remain necessary. |

Voice: follow the midterm's precise, qualified academic tone; remove promotional language, em dashes and clause-heavy sentences. Keep meaningful uncertainty. Every final numeric table will be generated from saved measurements.

Citation access: primary publisher/arXiv pages were blocked by the environment's network policy. Bibliographic entries and method descriptions use the supplied reports and established method documentation; independent primary-source rechecking is not claimed. See evidence/citation_check.json. Do not import unverified published benchmark scores into the final results.
