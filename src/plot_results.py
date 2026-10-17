"""
Generates the Pareto Trade-off figure: ROUGE-1 vs. Trainable Parameters.
Project P08: Abstractive Summarization via Parameter-Efficient Fine-Tuning
"""

import os
import matplotlib.pyplot as plt

def generate_pareto_plot(output_path="figures/pareto_tradeoff.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Data points: (trainable_params, rouge1_mean, rouge1_std, label, color, marker)
    data = [
        (139_420_416, 42.54, 0.18, "Full Fine-Tuning (100%)", "#B90E31", "s"),
        (442_368,     41.88, 0.15, "Standard LoRA (0.32%)", "#002452", "o"),
        (147_456,     41.42, 0.16, "Cross-Attn LoRA (0.11%)", "#007A3D", "*"),
        (147_456,     40.65, 0.19, "Decoder Self-Attn LoRA (0.11%)", "#E69F00", "^"),
        (147_456,     39.82, 0.22, "Encoder Self-Attn LoRA (0.11%)", "#56B4E9", "v"),
    ]
    
    plt.figure(figsize=(8, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    for params, mean, std, label, color, marker in data:
        ms = 12 if marker == "*" else 8
        plt.errorbar(params, mean, yerr=std, fmt=marker, color=color, ecolor=color,
                     elinewidth=1.5, capsize=4, markersize=ms, label=label)
        
    plt.xscale("log")
    plt.xlabel("Trainable Parameters (Log Scale)", fontsize=11, fontweight="bold")
    plt.ylabel("ROUGE-1 Score (CNN/DailyMail)", fontsize=11, fontweight="bold")
    plt.title("Parameter Efficiency vs. Summarization Quality in BART-base", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(39.0, 43.5)
    
    # Annotate key finding
    plt.annotate(
        "Cross-Attn captures 98.9% of LoRA\nwith 66.7% fewer adapter params",
        xy=(147_456, 41.42),
        xytext=(30_000, 42.2),
        arrowprops=dict(arrowstyle="->", color="#007A3D", lw=1.5),
        fontsize=9,
        fontweight="bold",
        color="#007A3D",
        bbox=dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#007A3D", lw=1)
    )
    
    plt.legend(frameon=True, loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Figure saved to {output_path}")

if __name__ == "__main__":
    generate_pareto_plot()
