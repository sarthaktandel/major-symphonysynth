#!/usr/bin/env python3
# robust generator that supports both int and string tokenizers

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import random
import json
from pathlib import Path
import pretty_midi
from tokenizer.remi_m_tokenizer import REMIMTokenizer
import argparse
from src.models.melody_model_mamba import MelodyTransformerMamba

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=str, default=None)
args = parser.parse_args()

vocab_path = Path("experiments/fine_tuned/extended_vocab.json")
if vocab_path.exists():
    with open(vocab_path) as f:
        vocab = json.load(f)
    print(f"✅ Loaded extended vocab ({len(vocab)} tokens)")
else:
    from tokenizer.remi_m_tokenizer import vocab
    print(f"⚠️ Using default tokenizer vocab ({len(vocab)} tokens)")

# -----------------------------
# Model Definition (same as training)
# -----------------------------
class MelodyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        emb = self.embedding(src)
        out = self.transformer(emb)
        return self.output(out)

# -----------------------------
# Config
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
checkpoint_path = "experiments/mamba_epoch2.pt"
seed_midi_path = args.seed if args.seed else "data/fine_tune_hinduraga/Bageshri.mid"
save_dir = Path("outputs")
save_dir.mkdir(exist_ok=True)
MODEL_TYPE = "mamba"   # or "transformer"
print("\n🎵 Melody Generator (int-compatible)")

# -----------------------------
# Load tokenizer and model
# -----------------------------
tokenizer = REMIMTokenizer()

# ✅ force tokenizer to use extended vocab
if vocab_path.exists():
    tokenizer.vocab = vocab
    tokenizer.token2id = {t: i for i, t in enumerate(vocab)}
    tokenizer.id2token = {i: t for i, t in enumerate(vocab)}

ckpt = torch.load(checkpoint_path, map_location=device)
ckpt = torch.load(checkpoint_path, map_location=device)

if "embedding.weight" in ckpt:
    ckpt_vocab_size = ckpt["embedding.weight"].shape[0]
else:
    ckpt_vocab_size = len(tokenizer.vocab)

vocab_size = ckpt_vocab_size
print(f"✅ Detected vocab size from checkpoint (or tokenizer): {vocab_size}")

if MODEL_TYPE == "transformer":
    model = MelodyTransformer(vocab_size).to(device)
elif MODEL_TYPE == "mamba":
    model = MelodyTransformerMamba(vocab_size).to(device)
else:
    raise ValueError("Invalid MODEL_TYPE")
model.load_state_dict(ckpt, strict=False)
model.eval()
print(f"✅ Loaded model from {checkpoint_path}")

# -----------------------------
# Tokenize seed
# -----------------------------
try:
    seed_tokens = tokenizer.midi_to_tokens(seed_midi_path)
    print(f"🎼 Tokenized seed MIDI: {len(seed_tokens)} tokens from {seed_midi_path}")
except Exception as e:
    print(f"⚠️ Failed to tokenize seed MIDI: {e}")
    seed_tokens = []

# Detect token type (int or str)
if len(seed_tokens) > 0 and isinstance(seed_tokens[0], int):
    token_type = "int"
else:
    token_type = "str"
print(f"🧠 Token type detected: {token_type}")

# Map seed to tensor
if token_type == "int":
    seed_ids = seed_tokens[:128]  # just limit for context
else:
    token2id = tokenizer.token2id if hasattr(tokenizer, "token2id") else {t: i for i, t in enumerate(tokenizer.vocab)}
    seed_ids = [token2id[t] for t in seed_tokens if t in token2id][:128]

if len(seed_ids) == 0:
    print("⚠️ Empty seed, defaulting to [0]")
    seed_ids = [0]

print(f"🎶 Seed length: {len(seed_ids)}")

# -----------------------------
# Generation setup
# -----------------------------
input_ids = torch.tensor([seed_ids], dtype=torch.long, device=device)
generated = input_ids
max_new_tokens = 512
temperature = 0.9
top_k = 20

def sample_top_k_logits(logits, prev_token, tokenizer, time_shift_count, top_k=10, temperature=1.0):
    logits = logits / max(temperature, 1e-8)

    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, top_k)

    valid_indices = []

    for idx in indices:
        token = tokenizer.id2token.get(idx.item(), None)

        # 🔒 SAFETY CHECK
        if token is None or not isinstance(token, str):
            continue
        
        # -----------------------------
        # 🎼 DURATION CONSTRAINT
        # -----------------------------
        if prev_token is not None and isinstance(prev_token, str):
            if prev_token.startswith("Note_on") and not token.startswith("Duration"):
                continue

        # -----------------------------
        # BASIC GRAMMAR RULES
        # -----------------------------
        if prev_token is not None and isinstance(prev_token, str):
            if prev_token.startswith("Bar") and not token.startswith("Position"):
                continue
            if prev_token.startswith("Position") and not token.startswith("Note_on"):
                continue

        # -----------------------------
        # LIMIT TIME SHIFT
        # -----------------------------
        if token.startswith("Time_shift") and time_shift_count > 3:
            continue

        valid_indices.append(idx)

    if len(valid_indices) == 0:
        valid_indices = indices

    valid_indices = torch.tensor(valid_indices, device=logits.device)
    valid_probs = probs[valid_indices]
    valid_probs = valid_probs / valid_probs.sum()

    choice = torch.multinomial(valid_probs, 1)
    return valid_indices[choice].item()

# -----------------------------
# Generate
# -----------------------------
print("🎹 Generating melody...")
time_shift_count = 0
MAX_TIME_SHIFT = 4

with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_logits = logits[:, -1, :].squeeze(0)

        # 🚫 Block UNK
        if hasattr(tokenizer, "token2id") and "UNK" in tokenizer.token2id:
            unk_id = tokenizer.token2id["UNK"]
            next_logits[unk_id] = -1e9

        # get previous token safely
        prev_token = tokenizer.id2token.get(generated[0, -1].item(), None)
        if not isinstance(prev_token, str):
            prev_token = None

        # sample next token
        next_id = sample_top_k_logits(
            next_logits,
            prev_token,
            tokenizer,
            time_shift_count,
            top_k=top_k,
            temperature=temperature
        )

        # track time shift
        token = tokenizer.id2token.get(next_id, None)
        if isinstance(token, str):
            if token.startswith("Time_shift"):
                time_shift_count += 1
            else:
                time_shift_count = 0

        # append token
        next_id_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        generated = torch.cat([generated, next_id_tensor], dim=1)

# skip invalid tokens
        if token is None or not isinstance(token, str):
            continue

        # ⏱ Time shift control
        if token.startswith("Time_shift"):
            time_shift_count += 1
        else:
            time_shift_count = 0

        if time_shift_count > MAX_TIME_SHIFT:
            continue
        generated = torch.cat([generated, next_id_tensor], dim=1)

generated_ids = generated[0].tolist()
print(f"✅ Generated {len(generated_ids)} tokens (including seed)")

# Save token IDs for debugging
(save_dir / "generated_token_ids.txt").write_text("\n".join(map(str, generated_ids)))

# -----------------------------
# Convert back to MIDI
# -----------------------------
midi_path = save_dir / "generated.mid"
print("🎼 Converting tokens → MIDI...")

# Proper conversion from int → string before decoding
try:
    tokens = []

    for i in generated_ids:
        token = tokenizer.id2token.get(i, None)
        if token is None:
            continue
        tokens.append(token)
    
    print(f"🎯 Converted {len(tokens)} tokens to string format")

    tokenizer.tokens_to_midi(tokens, out_path=str(midi_path))
    print(f"✅ MIDI saved → {midi_path}")
except Exception as e:
    print(f"⚠️ tokenizer.tokens_to_midi failed: {e}")
    # fallback minimal audible
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    time = 0.0
    for i in generated_ids[:100]:
        pitch = 60 + (i % 24)
        note = pretty_midi.Note(velocity=90, pitch=pitch, start=time, end=time + 0.4)
        inst.notes.append(note)
        time += 0.4
    midi.instruments.append(inst)
    midi.write(str(midi_path))
    print(f"✅ Fallback MIDI created → {midi_path}")

print("🎉 Generation complete!")