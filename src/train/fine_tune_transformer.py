import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from tokenizer.remi_m_tokenizer import REMIMTokenizer, vocab

# -------------------------------------------------
# Step 1: Extend vocabulary (NEW)
# -------------------------------------------------
def extend_vocab_with_new_tokens(data_dir):
    print("🔍 Scanning raga files for unseen tokens...")
    tokenizer_tmp = REMIMTokenizer()
    new_tokens = set()

    for f in Path(data_dir).rglob("*.mid"):
        try:
            toks = tokenizer_tmp.midi_to_tokens(str(f))
            for t in toks:
                if t not in vocab:
                    new_tokens.add(t)
        except Exception as e:
            print(f"⚠️ Skipped {f.name}: {e}")

    if new_tokens:
        print(f"✨ Found {len(new_tokens)} new tokens. Extending vocabulary...")
        vocab.extend(sorted(new_tokens))
    else:
        print("✅ No new tokens found, vocabulary unchanged.")


# -------------------------------------------------
# Dataset Loader (Fixed)
# -------------------------------------------------
class MIDITokenDataset(Dataset):
    def __init__(self, data_dir):
        self.files = list(Path(data_dir).rglob("*.mid"))
        self.tokenizer = REMIMTokenizer()
        self.samples = []
        self.token2id = {tok: i for i, tok in enumerate(vocab)}
        print(f"🎼 Found {len(self.files)} MIDI files for fine-tuning")

        for f in self.files:
            try:
                tokens = self.tokenizer.midi_to_tokens(str(f))
                ids = [self.token2id[t] for t in tokens if t in self.token2id]

                if len(ids) > 10:
                    self.samples.append(torch.tensor(ids[:1024], dtype=torch.long))
                    print(f"✅ {f.name}: {len(tokens)} tokens")
                else:
                    print(f"⚠️ Skipped {f.name}: too few valid tokens ({len(ids)})")

            except Exception as e:
                print(f"⚠️ Skipped {f.name}: {e}")

        print(f"📊 Total usable sequences: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        inp = seq[:-1]
        tgt = seq[1:]
        return inp, tgt


# -------------------------------------------------
# Model (must match training model)
# -------------------------------------------------
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


# -------------------------------------------------
# Fine-tuning Config
# -------------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
base_checkpoint = Path("experiments/transformer_epoch10.pt")
fine_tune_dir = Path("experiments/fine_tuned")
fine_tune_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path("data/fine_tune_hinduraga")

# -------------------------------------------------
# Step 2: Extend vocab before dataset creation
# -------------------------------------------------
extend_vocab_with_new_tokens(data_dir)

# -------------------------------------------------
# Step 3: Load Dataset
# -------------------------------------------------
dataset = MIDITokenDataset(data_dir)
if len(dataset) == 0:
    raise RuntimeError("❌ No usable sequences found. Check tokenization or MIDI validity.")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# -------------------------------------------------
# Step 4: Load Base Model (safe partial loading)
# -------------------------------------------------
vocab_size = len(vocab)
model = MelodyTransformer(vocab_size).to(device)

checkpoint = torch.load(base_checkpoint, map_location=device)
model_dict = model.state_dict()

# Filter out mismatched shapes (embedding/output)
compatible_weights = {
    k: v for k, v in checkpoint.items()
    if k in model_dict and model_dict[k].shape == v.shape
}

print(f"⚙️ Loading {len(compatible_weights)}/{len(checkpoint)} layers from base checkpoint")
model_dict.update(compatible_weights)
model.load_state_dict(model_dict, strict=False)

print(f"✅ Loaded compatible base weights from {base_checkpoint}")

# -------------------------------------------------
# Step 5: Fine-tuning setup
# -------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
epochs = 3

print("🚀 Starting fine-tuning...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(inp)
        loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    ckpt_path = fine_tune_dir / f"fine_tuned_epoch{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Epoch {epoch+1}/{epochs} | Loss={avg_loss:.4f} | Saved {ckpt_path}")

print("🎉 Fine-tuning complete!")

import json
vocab_path = fine_tune_dir / "extended_vocab.json"
with open(vocab_path, "w") as f:
    json.dump(vocab, f, indent=2)
print(f"🧩 Saved extended vocab → {vocab_path}")