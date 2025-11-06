# results_plots.py
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json

# ============ CONFIG ============
loss_log = {
    "train": [3.76, 2.33, 1.97],  # <-- replace with actual printed losses if available
    "epochs": [1, 2, 3]
}

token_counts_file = "outputs/generated_tokens.txt"  # or path to your token text
midi_token_lengths = {
    "Peelu.mid": 4544, "Miyan_ki_Malhar.mid": 4554, "Bhairav.mid": 1949,
    "Gaud.mid": 3329, "Behag.mid": 2504, "Pahadi.mid": 4124, "Des.mid": 3599,
    "Chandra_Kauns.mid": 3279, "Bahar.mid": 2674, "JaunaPuri.mid": 4909,
    "Tilang.mid": 3644, "Bageshri.mid": 7364, "Purvi.mid": 3354, "Adana.mid": 4154,
    "Alhiya_Bilawal.mid": 3274, "Durga.mid": 4219, "bazigar.mid": 23884,
    "Tilak_Kamode.mid": 5644
}
# =================================

# ---------- Plot 1: Training Loss ----------
plt.figure(figsize=(6,4))
plt.plot(loss_log["epochs"], loss_log["train"], marker='o', color='b')
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results_loss_curve.png", dpi=300)
plt.close()

# ---------- Plot 2: Token Distribution ----------
if os.path.exists(token_counts_file):
    with open(token_counts_file, "r") as f:
        tokens = f.read().split()
    token_freq = Counter(tokens)
    top_tokens = dict(token_freq.most_common(20))
    plt.figure(figsize=(8,4))
    plt.bar(list(top_tokens.keys()), list(top_tokens.values()), color='skyblue')
    plt.xticks(rotation=90)
    plt.title("Top 20 Tokens in Generated Sequence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("results_token_distribution.png", dpi=300)
    plt.close()
else:
    print("⚠️ Token file not found, skipping token distribution plot.")

# ---------- Plot 3: Sequence Lengths ----------
plt.figure(figsize=(6,4))
plt.barh(list(midi_token_lengths.keys()), list(midi_token_lengths.values()), color='lightgreen')
plt.title("Token Sequence Length per MIDI File")
plt.xlabel("Token Count")
plt.tight_layout()
plt.savefig("results_sequence_lengths.png", dpi=300)
plt.close()

print("✅ All result graphs saved as:")
print("   results_loss_curve.png")
print("   results_token_distribution.png (if tokens available)")
print("   results_sequence_lengths.png")