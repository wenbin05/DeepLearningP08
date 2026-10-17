# CISC 473: Deep Learning — Project P08
## Abstractive Summarization via Parameter-Efficient Fine-Tuning

**Group Members:**
- **Choo Wen Bin** (20630085) — Project Lead
- **Nai Junn Rong** (20630086) — Research Lead

**Repository:** `https://github.com/wenbin05/DeepLearningP08`

---

### Project Overview
This project investigates the impact of architectural placement on Parameter-Efficient Fine-Tuning (PEFT) for abstractive text summarization. Using **BART-base** (`facebook/bart-base`) evaluated on **CNN/DailyMail** and **XSum**, we perform a controlled, budget-matched ablation isolating:
1. **Encoder Self-Attention LoRA**
2. **Decoder Self-Attention LoRA**
3. **Encoder-Decoder Cross-Attention LoRA**

Because BART-base consists of 6 encoder and 6 decoder layers of identical hidden dimensionality ($d=768$), applying LoRA ($r=8$ on $W_q, W_v$) to each region yields an **identical budget of exactly 147,456 trainable parameters (0.11% of the base model)**. This isolates architectural placement as the independent variable without capacity confounding.

---

### Repository Structure
```
├── Proposal.pdf            # Project Proposal PDF (Submitted & Evaluated: 22/20)
├── Proposal_final.tex      # LaTeX source for Project Proposal
├── midterm-template.pdf    # Compiled Midterm Report PDF
├── midterm-template.tex    # LaTeX source for Midterm Report
├── README.md               # Project documentation and reproduction guide
├── requirements.txt        # Python package dependencies
├── train_colab.ipynb       # Colab/Kaggle notebook for GPU execution
├── figures/
│   └── pareto_tradeoff.png # ROUGE-1 vs. Trainable Parameters trade-off plot
└── src/
    ├── config.py           # Experiment hyperparameters and LoRA placement targets
    ├── count_params.py     # Verification script for exact parameter budgets
    ├── evaluate.py         # Standardized ROUGE evaluation metrics
    ├── model.py            # Model loader and targeted PEFT injection
    ├── plot_results.py     # Script to generate Pareto trade-off curve
    └── train.py            # Seq2Seq training script supporting multiple seeds
```

---

### Key Preliminary Findings (Midterm Milestone)

#### 1. Exact Parameter Invariant Verification
```
=====================================================================================
Configuration                    | Trainable Params   | Total Params     | Trainable %
-------------------------------------------------------------------------------------
Zero-Shot BART-base              |                  0 |      139,420,416 |    0.0000%
Full Fine-Tuning                 |        139,420,416 |      139,420,416 |  100.0000%
Standard LoRA (All Attention)    |            442,368 |      139,862,784 |    0.3173%
Encoder Self-Attention LoRA      |            147,456 |      139,567,872 |    0.1058%
Decoder Self-Attention LoRA      |            147,456 |      139,567,872 |    0.1058%
Cross-Attention LoRA             |            147,456 |      139,567,872 |    0.1058%
=====================================================================================
```

#### 2. Quantitative Results on CNN/DailyMail (Mean $\pm$ Std over 3 Seeds)
- **Published Baseline (Lewis et al., 2020):** ROUGE-1: $42.30$ | ROUGE-2: $19.50$ | ROUGE-L: $39.40$
- **Our Reproduction (Full Fine-Tuning):** ROUGE-1: **$42.54 \pm 0.18$** | ROUGE-2: **$19.68 \pm 0.14$** | ROUGE-L: **$39.52 \pm 0.16$** (Matches published baseline within $0.5\%$).
- **Standard LoRA (All Attention):** ROUGE-1: $41.88 \pm 0.15$ | ROUGE-2: $19.12 \pm 0.12$ | ROUGE-L: $38.94 \pm 0.14$
- **Cross-Attention LoRA (147k Params, 0.11%):** ROUGE-1: **$41.42 \pm 0.16$** | ROUGE-2: **$18.74 \pm 0.14$** | ROUGE-L: **$38.56 \pm 0.15$**
  - *Captures 98.9% of standard LoRA performance using 66.7% fewer adapter parameters.*
  - *Outperforms Encoder Self-Attention LoRA ($39.82$) by $+1.60$ ROUGE-1 under the exact same parameter capacity.*

---

### Quickstart & Reproduction

#### 1. Parameter Budget Verification
Run the standalone parameter verification script:
```bash
python src/count_params.py
```

#### 2. Generate Pareto Trade-Off Plot
```bash
python src/plot_results.py
```

#### 3. Training with Targeted LoRA Placement
To run training across experimental seeds (`42`, `123`, `999`):
```bash
# Cross-Attention LoRA (Primary Proposed Configuration)
python src/train.py --placement cross_attn --seed 42

# Encoder Self-Attention LoRA
python src/train.py --placement encoder_self_attn --seed 42

# Decoder Self-Attention LoRA
python src/train.py --placement decoder_self_attn --seed 42
```
