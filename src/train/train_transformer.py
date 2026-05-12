#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline Transformer Training Script
- Loads tokenized REMI-M data (.npy files)
- Trains a Transformer-based next-token predictor
- Saves checkpoints under experiments/
"""
import sys
from pathlib import Path
import os
import json
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.melody_model_mamba import MelodyTransformerMamba

sys.path.append(str(Path(__file__).resolve().parents[1]))

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("experiments")
BATCH_SIZE = 8
SEQ_LEN = 512
EPOCHS = 2
LR = 2e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "mamba"   # "transformer" or "mamba"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------------
# Dataset
# -------------------------------

class MelodyDataset(Dataset):
    def __init__(self, data_dir: Path, seq_len=512):
        self.files = list(data_dir.glob("*_tokens.npy"))
        self.seq_len = seq_len
        self.samples = []

        for file in self.files:
            arr = np.load(file)
            if len(arr) > 10:
                self.samples.append(torch.tensor(arr, dtype=torch.long))

        print(f"📦 Loaded {len(self.samples)} sequences from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        if len(seq) <= self.seq_len:
            pad_len = self.seq_len - len(seq)
            seq = torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        else:
            start = random.randint(0, len(seq) - self.seq_len - 1)
            seq = seq[start:start + self.seq_len]
        x = seq[:-1]
        y = seq[1:]
        return x, y

# -------------------------------
# Transformer Model
# -------------------------------

class MelodyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch, seq]
        x = self.embedding(x) * math.sqrt(x.size(1))
        x = x.permute(1, 0, 2)  # seq, batch, dim
        out = self.transformer(x)
        out = out.permute(1, 0, 2)  # back to batch, seq, dim
        logits = self.output(out)
        return logits

# -------------------------------
# Training
# -------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Load vocab size
    from tokenizer.remi_m_tokenizer import vocab
    vocab_size = len(vocab)
    print(f"🧠 Vocabulary size: {vocab_size}")

    dataset = MelodyDataset(DATA_DIR, seq_len=SEQ_LEN)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    if MODEL_TYPE == "transformer":
        model = MelodyTransformer(vocab_size).to(DEVICE)
        # 🔥 LOAD pretrained transformer weights
        pretrained_path = "experiments/fine_tuned/fine_tuned_epoch3.pt"
        ckpt = torch.load(pretrained_path, map_location=DEVICE)

        model.load_state_dict(ckpt, strict=False)
        print("✅ Loaded pretrained Transformer weights into hybrid model")
        # 🔒 Freeze transformer
        for param in model.transformer.parameters():
            param.requires_grad = False

        print("🔒 Transformer frozen, training Mamba only")
    elif MODEL_TYPE == "mamba":
        model = MelodyTransformerMamba(vocab_size).to(DEVICE)
    else:
        raise ValueError("Invalid MODEL_TYPE")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Training on {DEVICE} for {EPOCHS} epochs")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        ckpt_path = CHECKPOINT_DIR / f"{MODEL_TYPE}_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

    print("✅ Training complete.")

if __name__ == "__main__":
    main()