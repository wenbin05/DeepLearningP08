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

Because BART-base consists of 6 encoder and 6 decoder layers of identical hidden dimensionality ($d=768$), applying LoRA ($r=8$ on $W_q, W_v$) to each region yields an **identical budget of exactly 147,456 trainable parameters (0.11% of the model)**. This isolates architectural placement as the independent variable without capacity confounding.

---

### Repository Structure
```
├── midterm-template.tex    # LaTeX source for Midterm Report
├── README.md               # Project documentation and reproduction guide
├── requirements.txt        # Python package dependencies
├── src/
│   ├── config.py           # Experiment hyperparameters and LoRA placement targets
│   ├── count_params.py     # Verification script for exact parameter budgets
│   ├── evaluate.py         # Standardized ROUGE evaluation metrics
│   ├── model.py            # Model loader and targeted PEFT injection
│   └── train.py            # Seq2Seq training script supporting multiple seeds
└── train_colab.ipynb       # Colab/Kaggle notebook for GPU execution
```

---

### Quickstart & Reproduction

#### 1. Parameter Budget Verification
Run the standalone parameter verification script to confirm exact parameter matching across all configurations:
```bash
python src/count_params.py
```
Output:
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

#### 2. Training with Targeted LoRA Placement
To run training across the 3 experimental seeds (42, 123, 999):
```bash
# Cross-Attention LoRA (Primary Proposed Configuration)
python src/train.py --placement cross_attn --seed 42

# Encoder Self-Attention LoRA
python src/train.py --placement encoder_self_attn --seed 42

# Decoder Self-Attention LoRA
python src/train.py --placement decoder_self_attn --seed 42
```
