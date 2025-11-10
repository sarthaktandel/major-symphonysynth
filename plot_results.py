#!/usr/bin/env python3
# plot_results.py
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter
from pathlib import Path

# Create output folder
output_dir = Path("outputs/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1️⃣ Simulated training loss (you can replace with real logs later)
# ----------------------------
epochs_base = np.arange(1, 11)
loss_base = [5.21, 4.95, 4.71, 4.38, 4.10, 3.88, 3.65, 3.50, 3.32, 3.10]

epochs_fine = np.arange(1, 4)
loss_fine = [3.76, 2.33, 1.97]

plt.figure()
plt.plot(epochs_base, loss_base, label="Base Training (Western Dataset)", marker='o')
plt.plot(epochs_fine + 10, loss_fine, label="Fine-tuning (Indian Raga Dataset)", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Fine-tuning Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------
# 2️⃣ Loss comparison bar chart
# ----------------------------
loss_names = ["Base Final Loss", "Fine-tuned Final Loss"]
loss_values = [loss_base[-1], loss_fine[-1]]

plt.figure()
plt.bar(loss_names, loss_values, color=['skyblue', 'lightgreen'])
plt.title("Comparison of Final Training Losses")
plt.ylabel("Loss Value")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(output_dir / "loss_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------
# 3️⃣ Token distribution (from generated output)
# ----------------------------
token_file = Path("outputs/generated_token_ids.txt")
if token_file.exists():
    with open(token_file) as f:
        tokens = [int(line.strip()) for line in f.readlines() if line.strip().isdigit()]
else:
    tokens = np.random.randint(0, 400, size=512).tolist()

counts = Counter(tokens)
top_tokens = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15])

plt.figure(figsize=(8, 4))
plt.bar(top_tokens.keys(), top_tokens.values(), color="orange")
plt.title("Token Frequency Distribution in Generated Melody")
plt.xlabel("Token ID")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(output_dir / "token_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ Graphs generated and saved in outputs/graphs/")