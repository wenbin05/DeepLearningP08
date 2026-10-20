| Configuration | Trainable parameters | ROUGE-1 | ROUGE-2 | ROUGE-L | Validation loss | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| cross_attn | 147,456 | 41.31 | 18.88 | 25.73 | 3.2401 | 103.7 |
| decoder_self_attn | 147,456 | 41.32 | 19.13 | 25.93 | 3.2647 | 105.5 |
| encoder_self_attn | 147,456 | 41.02 | 19.05 | 25.47 | 3.2659 | 135.4 |
| zero_shot | 0 | 41.04 | 18.87 | 25.84 | 3.6372 | 0.0 |

Single-seed pilot (seed 42): fixed 1,024-example CNN/DailyMail training subset; 32 optimizer updates for trained models; 64-example validation subset; official stemmed ROUGE F1 ×100 against full references. No standard deviations are available. Short training is not evidence of convergence or published-baseline reproduction.
