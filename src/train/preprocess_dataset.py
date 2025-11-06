#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Preprocessing Script
- Reads all MIDI files from data/raw/ (or given path)
- Converts to REMI-M tokens
- Saves each sequence as .npy in data/processed/
- Creates dataset_index.json for tracking
- Logs skipped files into skipped_files.txt
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Import tokenizer
sys.path.append(str(Path(__file__).resolve().parents[1]))
from tokenizer.remi_m_tokenizer import REMIMTokenizer

# Default paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INDEX_FILE = PROCESSED_DIR / "dataset_index.json"
SKIPPED_FILE = PROCESSED_DIR / "skipped_files.txt"

def preprocess_all(raw_dir: Path):
    if not raw_dir.exists():
        print(f"❌ Raw data folder not found: {raw_dir}")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = REMIMTokenizer()
    dataset_index = {}
    skipped = []

    midi_files = list(raw_dir.glob("**/*.mid")) + list(raw_dir.glob("**/*.midi"))
    print(f"🎵 Found {len(midi_files)} MIDI files in {raw_dir}")

    for i, midi_file in enumerate(midi_files, 1):
        try:
            tokens = tokenizer.midi_to_tokens(str(midi_file), mode="major", instrument=0)
            out_path = PROCESSED_DIR / (midi_file.stem + "_tokens.npy")
            np.save(out_path, np.array(tokens, dtype=np.int32))

            dataset_index[str(midi_file)] = str(out_path)
            print(f"[{i}/{len(midi_files)}] ✅ Processed {midi_file.name} → {out_path.name}")

        except Exception as e:
            skipped.append((str(midi_file), str(e)))
            print(f"[{i}/{len(midi_files)}] ⚠️ Skipped {midi_file.name}: {e}")

    # Save index
    with open(INDEX_FILE, "w") as f:
        json.dump(dataset_index, f, indent=2)

    # Save skipped log
    if skipped:
        with open(SKIPPED_FILE, "w") as f:
            for midi_path, err in skipped:
                f.write(f"{midi_path} ::: {err}\n")
        print(f"\n⚠️ Logged {len(skipped)} skipped files to {SKIPPED_FILE}")

    print(f"📂 Saved dataset index to {INDEX_FILE}")
    print("✅ Preprocessing complete.")

if __name__ == "__main__":
    # Allow optional custom path (e.g. python preprocess_dataset.py data/raw/lmd_matched/A/)
    if len(sys.argv) > 1:
        raw_dir = Path(sys.argv[1])
    else:
        raw_dir = RAW_DIR

    preprocess_all(raw_dir)