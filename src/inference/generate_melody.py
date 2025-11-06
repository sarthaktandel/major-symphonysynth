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
checkpoint_path = "experiments/fine_tuned/fine_tuned_epoch3.pt"
seed_midi_path = "data/fine_tune_hinduraga/Bageshri.mid"
save_dir = Path("outputs")
save_dir.mkdir(exist_ok=True)

print("\n🎵 Melody Generator (int-compatible)")

# -----------------------------
# Load tokenizer and model
# -----------------------------
tokenizer = REMIMTokenizer()
ckpt = torch.load(checkpoint_path, map_location=device)

if "embedding.weight" in ckpt:
    ckpt_vocab_size = ckpt["embedding.weight"].shape[0]
else:
    ckpt_vocab_size = len(tokenizer.vocab)

vocab_size = ckpt_vocab_size
print(f"✅ Detected vocab size from checkpoint (or tokenizer): {vocab_size}")

model = MelodyTransformer(vocab_size).to(device)
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
temperature = 1.1
top_k = 10

def sample_top_k_logits(logits, top_k=10, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    if top_k <= 0:
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    values, indices = torch.topk(logits, top_k)
    probs = torch.softmax(values, dim=-1)
    idx = torch.multinomial(probs, 1)
    return indices[idx].item()

# -----------------------------
# Generate
# -----------------------------
print("🎹 Generating melody...")
with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_logits = logits[:, -1, :].squeeze(0)
        next_id = sample_top_k_logits(next_logits, top_k=top_k, temperature=temperature)
        next_id_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
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
    if token_type == "int":
        # ensure tokenizer has id2token mapping
        if hasattr(tokenizer, "id2token"):
            tokens = [tokenizer.id2token[i] for i in generated_ids if i < len(tokenizer.id2token)]
        else:
            print("⚠️ tokenizer.id2token not found — building temporary reverse map")
            id2token = {i: t for i, t in enumerate(tokenizer.vocab)}
            tokens = [id2token[i] for i in generated_ids if i in id2token]
    else:
        tokens = generated_ids  # already string tokens

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