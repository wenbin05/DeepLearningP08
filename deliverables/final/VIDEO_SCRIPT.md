# Group 04 video presentation script

Target: 20:00. Slides 1–8: Choo Wen Bin (10:00). Slides 9–16: Nai Junn Rong (10:00). Both speakers introduce themselves and appear on camera. Timing includes pauses to point at figures. Rehearse; do not pad or rush to match a timestamp. The reference slides are included in the PDF as a closing resource.

This is an AI-assisted rehearsal draft. Revise the analysis into your own understood wording before recording; the course requires author-owned interpretation. Both members contributed equally.

## 01. Where to adapt BART? | 00:00–01:15 | Choo Wen Bin

Hello, we are Group 04. I am Choo Wen Bin, and my teammate is Nai Junn Rong. Our project studies where to place low-rank adapters in BART for news summarization.

The motivation is straightforward. A summary should preserve important information, but adapting a language model can be expensive. LoRA reduces the number of parameters we train. Our question is whether the location of those parameters matters when the budget stays the same.

We compare encoder self-attention, decoder self-attention and cross-attention. We also evaluate broader baselines and ablations. The completed experiment contains thirty-nine runs and seventy-eight test evaluations, using small, fixed data subsets on a CPU.

The main outcome is qualified: cross-attention leads the other isolated placements on our in-domain scores, but it does not beat unadapted BART on ROUGE-1. I will explain the design first. Junn Rong will then present the evidence and its limitations.

**Visual cue:** Pause on the main comparison before moving on.

## 02. Three ideas motivate the comparison | 01:15–02:30 | Choo Wen Bin

The literature connects three ideas. First, BART provides a pretrained encoder and decoder that can be adapted for text generation. We use the base checkpoint rather than claim to reproduce the large-model scores in the original paper.

Second, LoRA learns low-rank weight updates, while Prefix-Tuning learns continuous conditioning vectors. They reduce trainable capacity in different ways. Work on adaptive allocation, dynamic ranks and adapter selection suggests that the distribution of adaptation capacity deserves attention.

Third, factuality work such as FactCC explains why reference overlap is insufficient. A summary can retain most of an article’s words and still reverse a crucial statement.

Our experiment brings these ideas together through an interpretable comparison. We fix three architectural locations and match their trainable counts exactly. We are not proposing a new adapter algorithm, and we do not run the adaptive-allocation methods themselves. The contribution is the comparison and the evidence it produces.

**Visual cue:** Pause on the main comparison before moving on.

## 03. Three locations, three computational roles | 02:30–03:45 | Choo Wen Bin

To understand the comparison, consider the path from an article to a summary. The encoder reads the source using bidirectional self-attention. This builds representations of the article’s words in context.

The decoder uses causal self-attention over the output prefix. It conditions each next-token prediction on what has already been generated. Cross-attention connects those decoder states to the encoder’s representations of the article.

BART-base has six encoder layers and six decoder layers, with hidden dimension seven hundred and sixty-eight. Its matching attention dimensions make it possible to allocate exactly the same LoRA budget to each of our three regions.

We might expect source-conditioning changes to help summarization, which motivates studying cross-attention. But that is a hypothesis. Architectural intuition cannot establish better factuality or better scores. Those claims need the measured comparisons, and our study does not include a causal analysis of attention maps.

**Visual cue:** Pause on the main comparison before moving on.

## 04. Match capacity before comparing placement | 03:45–05:00 | Choo Wen Bin

LoRA represents a weight update as the product of two smaller matrices. The original projection stays frozen, and the added update is scaled by alpha divided by the rank.

For the main comparison, rank is eight, alpha is sixteen and adapter dropout is zero point zero five. We adapt the query and value projections in each of six attention blocks.

A single seven-hundred-and-sixty-eight by seven-hundred-and-sixty-eight projection needs rank times the sum of its input and output dimensions. Multiplying by two projections and six blocks gives exactly one hundred and forty-seven thousand, four hundred and fifty-six trainable parameters.

That is about zero point one zero six percent of the total model including adapters. The implementation checks actual target-module names and trainable counts. This directly addresses the proposal feedback: a score difference should not be explained simply by giving one of these three placements more capacity.

**Visual cue:** Pause on the main comparison before moving on.

## 05. Fixed subsets, separate held-out tests | 05:00–06:15 | Choo Wen Bin

The training data come from the official CNN and DailyMail training split. We use one thousand and twenty-four examples for training, sixty-four for validation and one hundred and twenty-eight for the held-out test evaluation.

We also evaluate on one hundred and twenty-eight examples from the XSum test split, without XSum fine-tuning. This is a transfer evaluation under a change in news-summary style.

Subset selection uses a fixed data seed, so every configuration sees the same selected examples. Training seeds forty-two, one hundred and twenty-three, and nine hundred and ninety-nine vary training randomness. Saved example IDs and hashes support membership checks.

The tokenizer limits sources to one thousand and twenty-four tokens and training targets to one hundred and twenty-eight. For ROUGE, however, we retain the original reference text. These are subset results. They should not be described as performance on the complete benchmarks, and source truncation remains a limitation.

**Visual cue:** Pause on the main comparison before moving on.

## 06. Training effort is controlled, with one exception | 06:15–07:30 | Choo Wen Bin

All adapted configurations train for three epochs, giving ninety-six optimizer updates. The effective batch size is thirty-two. We use AdamW, ten warmup steps, a linear learning-rate schedule and gradient clipping.

The main LoRA placements share the same learning rate and microbatch settings. Prefix-Tuning uses the same PEFT learning rate. Full fine-tuning uses a lower learning rate, as declared in the protocol.

There is also a practical exception. The original full fine-tuning attempt exceeded the eight-gibibyte memory limit. We archived that failed attempt and restarted from pretrained weights with microbatch one and thirty-two accumulation steps. The successful runs are stored separately.

The effective batch size remains thirty-two, but that does not guarantee identical token weighting or dropout behavior. We therefore treat full fine-tuning as a memory-adjusted reference. We match nominal data exposure and updates, not computation time or floating-point operations. The main placement comparison retains its shared settings.

**Visual cue:** Pause on the main comparison before moving on.

## 07. Evaluate generated summaries, not loss alone | 07:30–08:45 | Choo Wen Bin

Training minimizes teacher-forced cross-entropy: the negative log probability assigned to the reference target tokens. But a low training or validation loss does not guarantee that beam search produces the best summary.

We therefore evaluate the generated text using ROUGE-one, ROUGE-two and ROUGE-L. These measure unigram, bigram and longest-common-subsequence overlap. We use stemmed per-example F-one scores, average across examples and multiply by one hundred. Our ROUGE-L is not ROUGE-Lsum.

Decoding is deterministic, with four beams, a minimum length of fifty-six, a maximum of one hundred and twenty-eight and no repeated trigrams. We evaluate the final training checkpoint rather than choosing a checkpoint using the test set.

The tables report means and sample standard deviations over three training seeds. These deviations describe training variability on the same examples. They are not article-level confidence intervals, and we do not claim statistical significance from them.

**Visual cue:** Pause on the main comparison before moving on.

## 08. The completed experiment matrix | 08:45–10:00 | Choo Wen Bin

The full matrix contains thirteen configurations. These include the three isolated placements and unadapted BART. For broader LoRA comparisons, we vary query-and-value rank across four, eight and sixteen. We also test query-only, key-only, value-only and all four attention projections at rank eight.

Prefix-Tuning and full fine-tuning complete the baseline set. Multiplying thirteen configurations by three seed labels gives thirty-nine runs, of which thirty-six involve training. Every run has two held-out test evaluations.

The unadapted model deserves a distinction: it does not train, and its deterministic outputs are identical under the three seed labels. Its zero standard deviation should not be interpreted as evidence from three independently trained models.

Final checks verified the required artifacts and recomputed ROUGE from saved predictions. Our success criterion is an interpretable comparison, including negative results, rather than finding a winning method at any cost. Junn Rong will now explain what the results show.

**Visual cue:** Pause on the main comparison before moving on.

## 09. Cross-attention leads the matched placements | 10:00–11:15 | Nai Junn Rong

Thank you, Wen Bin. I am Nai Junn Rong, and I will cover the results and their interpretation.

This figure compares unadapted BART with the three equal-budget placements. On CNN and DailyMail, cross-attention reaches ROUGE-one of thirty-eight point five six. Encoder reaches thirty-seven point seven five, and decoder reaches thirty-seven point five zero. Cross-attention also has the highest mean ROUGE-two and ROUGE-L among these three placements.

However, the unadapted model reaches thirty-eight point six one on ROUGE-one. So the main conclusion is about the relative placement ranking, not a clear gain over unadapted BART.

On XSum, the three placement means are much closer. Cross-attention exceeds decoder by only about zero point zero two ROUGE-one points. Encoder has the highest ROUGE-two of the three. The transfer evidence therefore does not support a uniform winner across metrics. Error bars show seed standard deviations, not statistical significance.

**Visual cue:** Point to the plotted values and explain what the error bars represent.

## 10. Full FT is strongest; Prefix struggles here | 11:15–12:30 | Nai Junn Rong

The broader baseline comparison changes the story. Full fine-tuning reaches ROUGE-one of forty point zero six on CNN and DailyMail and nineteen point one seven on XSum. These are the highest baseline means, although full fine-tuning has the declared learning-rate and microbatch differences.

All-attention query-and-value LoRA reaches thirty-eight point nine two and seventeen point one six. It improves the mean over the isolated placements, but uses three times their trainable capacity. It cannot serve as evidence that placement alone explains the difference.

Prefix-Tuning performs poorly: its ROUGE-one means are fifteen point zero five and nine point seven five, with more seed variability. That is a real negative result of our configuration. We did not run a method-specific tuning study, so we cannot say whether more training, initialization or another optimization setup would resolve it.

Overall, the results do not support a claim that PEFT necessarily transfers better than full fine-tuning.

**Visual cue:** Pause on the main comparison before moving on.

## 11. More parameters do not guarantee higher ROUGE | 12:30–13:45 | Nai Junn Rong

The ablations show why trainable parameter count alone is not enough. For query-and-value adapters, increasing rank from four to eight to sixteen does not improve ROUGE-one monotonically on either test subset.

There is a caveat: alpha stays fixed at sixteen, so alpha divided by rank changes across this sweep. The experiment compares complete rank configurations; it does not isolate rank from update scaling.

Among the equal-count single-projection variants, value-only has the highest mean ROUGE-one on both datasets. Key-only has the highest CNN and DailyMail ROUGE-two, so even here the ranking depends on the metric.

Adapting query, key, value and output projections together gives the strongest LoRA ROUGE-one means, but uses eight hundred and eighty-four thousand, seven hundred and thirty-six trainable parameters. At the same count it exceeds rank-sixteen query-and-value LoRA. That comparison changes coverage, rank and scaling together, so it does not identify a single causal explanation.

**Visual cue:** Point to the plotted values and explain what the error bars represent.

## 12. Parameter efficiency and runtime are different | 13:45–15:00 | Nai Junn Rong

The three isolated placements optimize exactly the same number of parameters, yet their recorded training-loop times differ. Encoder-only adaptation averages about two point three one hours. Decoder-only averages one point four two, and cross-attention averages one point three nine.

These times include epoch validation and checkpoint saving. They exclude initial model and data preparation, and they exclude the separate final validation and test prediction passes. We should therefore avoid calling them total experiment runtimes.

Different backpropagation paths and sequence lengths may contribute to the gap, but we did not measure those factors separately. The timings are observations from this CPU environment, not a portable speed guarantee.

A small trainable fraction also does not mean that total memory falls by the same fraction. Frozen backbone weights and intermediate activations remain necessary. The full fine-tuning memory failure illustrates why memory constraints matter, but we did not collect a consistent peak-memory benchmark for every configuration.

**Visual cue:** Pause on the main comparison before moving on.

## 13. Human ratings: descriptive, with low agreement | 15:00–16:15 | Nai Junn Rong

For factual consistency, ten people each rated twenty outputs. These came from five shared articles and four configurations, using training seed forty-two. Model identities were hidden, and packet order was randomized. The team reports that people completed the ratings independently.

Cross-attention received fifty-eight percent consistent ratings, compared with fifty-four percent for unadapted BART, forty-six for decoder and thirty-eight for encoder. Each percentage comes from fifty judgments over only five summaries per configuration.

Agreement is the more important qualification. Pairwise agreement is fifty-two point two two percent, and Fleiss’ kappa is zero point zero four four. Some explanations also mix omissions or stylistic concerns with factual contradictions.

We retained the responses unchanged. These are descriptive ratings, not a reliable model ranking. The two extra consistent judgments for cross-attention over unadapted BART cannot establish a factuality advantage. Shared articles, outputs and participants also mean that the two hundred responses are not independent examples.

**Visual cue:** Point to the plotted values and explain what the error bars represent.

## 14. A concrete factual failure | 16:15–17:30 | Nai Junn Rong

This example makes the factuality issue concrete. The article concerns taekwondo athlete Aaron Cook. It states that he is twenty-four and has received a passport from Moldova.

The encoder-adapted output instead calls him thirty-one. It first says that he has been granted citizenship, then later says that he has been denied citizenship. It also changes the meaning of a quoted statement about which country he plans to represent.

All ten participants marked this particular output inconsistent. The saved source and prediction let us check the error directly rather than relying only on a participant’s explanation.

The summary still contains many correct names and topic words. That explains why lexical overlap cannot guarantee factual support: a small word change can reverse the meaning of a claim.

This is one selected case, not an estimate of the error rate. We also cannot infer the internal mechanism responsible for the mistake from the text alone.

**Visual cue:** Pause on the main comparison before moving on.

## 15. What we can and cannot conclude | 17:30–18:45 | Nai Junn Rong

We can now return to the research question. Within our shared, short-budget setup, cross-attention is the strongest isolated placement on the in-domain ROUGE means. Matching the trainable count and training settings makes this observation interpretable.

But the limits matter. The study uses a small training subset, small test subsets, three training seeds and one backbone. Seed variation does not account for uncertainty from selecting different articles. We did not conduct significance tests.

Full fine-tuning has an explicit microbatch exception, and computation time is not matched. The minimum generation length of fifty-six may also be unsuitable for some XSum references. Rank ablations change scaling, and Prefix-Tuning was not tuned separately. Human evidence is restricted to five articles and has low agreement.

These boundaries rule out broad claims of benchmark superiority or improved factual reliability. Negative and mixed results are still useful: they show exactly where our original expectation is supported and where stronger evidence is needed.

**Visual cue:** Pause on the main comparison before moving on.

## 16. Conclusion and next experiment | 18:45–20:00 | Nai Junn Rong

Our contribution is a controlled comparison of where a fixed LoRA budget enters BART, together with baseline, ablation and human-evaluation evidence. Cross-attention leads the isolated in-domain placements, but unadapted BART remains competitive. Wider projection coverage gives stronger LoRA ROUGE-one means at a larger budget. The human study does not establish a factuality winner.

The next experiment should repeat the matched placements on a larger fixed training subset and quantify article-level uncertainty. A separate validation study should examine generation-length settings before any new XSum test evaluation.

We contributed equally, fifty percent each.

We used OpenAI Codex for code implementation and the first report draft, with the academic-humanizer skill for writing feedback. Codex also helped run and check the experiments and prepare the analysis, figures and presentation. The measurements come from executed experiments, and the human ratings were supplied by the team. Thank you.

**Visual cue:** Pause on the main comparison before moving on.

## Recording and submission

The narration contains approximately 2,375 words. A 20-minute delivery averages 119 spoken words per minute including pauses. Rehearse once with a timer; 18–22 minutes is accepted and content after 22:00 is not assessed.

- Record slides at 720p or higher with clear audio and speaker camera visible. Both members must appear and speak; target ten minutes each.
- Introduce both full names. Keep a smooth handoff after slide 8.
- Use the figures and case excerpts as the main visuals. A demo is optional and not needed for this evidence-focused talk.
- Check every number and understand each comparison. Do not say “statistically significant,” “full benchmark,” or “proved factuality improvement.”
- Check the final MP4 plays with audio, ends before 22:00, and is named `Group04_Presentation.mp4`. Submit it with `Group04_Slides.pdf`.
- If using a OneDrive/Stream link, verify instructor/TA viewing access in a private browser window.
- Review AI disclosure and references before submission.