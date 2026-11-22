# Finalization status

All 39 configured runs and 78 held-out test evaluations completed on November 15, 2026 at 19:58 Toronto time. Final prediction-based ROUGE rescoring and matrix validation passed. Three of the 39 runs are deterministic unadapted controls; 36 involve training.

The completed protocol uses 1,024 CNN/DailyMail training examples, 64 validation examples, 128 test examples each for CNN/DailyMail and XSum, three epochs and seeds 42/123/999. It is a CPU subset study, not full-benchmark reproduction. Full FT uses batch 1 and accumulation 32 following the archived memory failure; the other runs use batch 2 and accumulation 16.

The confirmed human study contains 200 ratings from 10 participants over 20 summaries from five shared articles. Fleiss' kappa is 0.044; model proportions are descriptive.

The [final report, slides and script](../deliverables/final/README.md) are prepared. Remaining author tasks: review and own the interpretation, confirm individual contributions and citations, rehearse and record the 18–22 minute video, and submit the files. The course AI-use requirements are stated in the deliverable checklist. Local artifacts have not been represented as uploaded to GitHub or onQ.

The [November 11 status](FINALIZATION_STATUS_2026-11-11.md) is preserved as history. The final training source snapshot is unchanged by the writing pass.
