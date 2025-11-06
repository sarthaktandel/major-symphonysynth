#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REMI-M Tokenizer (with Instrument + Mode support)
MacOS Compatible Version
"""

import os
import sys
import pretty_midi
from typing import List

# -------------------------------
# Event vocabulary setup
# -------------------------------

BAR = "Bar"
POS = "Pos"
NOTE_ON = "Note_on"
DURATION = "Duration"
MODE_START = "Mode_start"
MODE_END = "Mode_end"
MODE_TYPE = "Mode_type"
NOTE_TYPE = "Note_type"
INSTRUMENT = "Instrument"

MODE_TYPES = ["major", "minor", "pentatonic", "dorian", "phrygian"]
NOTE_TYPES = ["mode_note", "transition_note"]
INSTRUMENTS = list(range(0, 128))   # General MIDI program numbers
DURATIONS = list(range(1, 17))      # 1..16 steps (e.g. 16th–whole note)

vocab = []
id2token = {}
token2id = {}

def build_vocab():
    global vocab, id2token, token2id
    vocab = []

    # Bars & positions
    vocab += [f"{BAR}"]
    for pos in range(0, 16):
        vocab.append(f"{POS}_{pos}")

    # Durations
    for dur in DURATIONS:
        vocab.append(f"{DURATION}_{dur}")

    # Pitches
    for pitch in range(128):
        vocab.append(f"{NOTE_ON}_{pitch}")

    # Modes
    for m in MODE_TYPES:
        vocab.append(f"{MODE_TYPE}_{m}")
    vocab.append(MODE_START)
    vocab.append(MODE_END)

    # Note types
    for n in NOTE_TYPES:
        vocab.append(f"{NOTE_TYPE}_{n}")

    # Instruments
    for inst in INSTRUMENTS:
        vocab.append(f"{INSTRUMENT}_{inst}")

    id2token = {i: tok for i, tok in enumerate(vocab)}
    token2id = {tok: i for i, tok in enumerate(vocab)}

build_vocab()

# -------------------------------
# Tokenizer
# -------------------------------

class REMIMTokenizer:
    def __init__(self):
        self.vocab = vocab
        self.token2id = token2id
        self.id2token = id2token

    def midi_to_tokens(self, midi_path: str, mode: str = "major", instrument: int = 0) -> List[int]:
        pm = pretty_midi.PrettyMIDI(midi_path)
        tokens = []

        # Add mode + instrument
        tokens.append(self.token2id[MODE_START])
        tokens.append(self.token2id[f"{MODE_TYPE}_{mode}"])
        tokens.append(self.token2id[f"{INSTRUMENT}_{instrument}"])

        for inst in pm.instruments:
            for note in inst.notes:
                bar = int(note.start // 4)
                pos = int((note.start % 4) * 4)

                tokens.append(self.token2id[BAR])
                tokens.append(self.token2id[f"{POS}_{pos}"])
                tokens.append(self.token2id[f"{NOTE_ON}_{note.pitch}"])

                dur = int(round(note.end - note.start))
                dur = max(1, min(dur, max(DURATIONS)))
                tokens.append(self.token2id[f"{DURATION}_{dur}"])

                tokens.append(self.token2id[f"{NOTE_TYPE}_mode_note"])

        tokens.append(self.token2id[MODE_END])
        return tokens

    def tokens_to_midi(self, tokens, out_path="reconstructed.mid", tempo=120):
        """
        Convert a token sequence back to a MIDI file.
        This version is robust to incomplete or malformed token groups.
        """
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)

        current_bar = 0
        current_pos = 0.0
        time_per_bar = 4.0 * 60.0 / tempo  # 4 beats per bar (common time)
        time_per_pos = time_per_bar / 16.0  # assume 16 positions per bar

        active_notes = []  # [(pitch, start_time)]

        for i, tok in enumerate(tokens):
            # Bar: reset position
            if tok.startswith("Bar"):
                current_bar += 1
                current_pos = 0.0
                continue

            # Position token (Pos_X)
            if tok.startswith("Pos_"):
                try:
                    current_pos = int(tok.split("_")[1]) * time_per_pos
                except:
                    continue
                continue

            # Note_on_X
            if tok.startswith("Note_on_"):
                try:
                    pitch = int(tok.split("_")[2])
                except:
                    continue

                # Find duration within next few tokens
                duration = 0.25  # default 1/4 beat
                for look_ahead in tokens[i+1:i+6]:
                    if look_ahead.startswith("Duration_"):
                        try:
                            duration_val = int(look_ahead.split("_")[1])
                            duration = duration_val * time_per_pos
                        except:
                            pass
                        break

                start_time = current_bar * time_per_bar + current_pos
                end_time = start_time + duration

                # Clamp duration to avoid 0-length notes
                if end_time <= start_time:
                    end_time = start_time + 0.1

                # Add note safely
                try:
                    note = pretty_midi.Note(
                        velocity=90, pitch=pitch, start=start_time, end=end_time
                    )
                    instrument.notes.append(note)
                except Exception as e:
                    print(f"⚠️ Skipped invalid note: {tok} ({e})")

                continue

            # Ignore structural or irrelevant tokens
            if tok.startswith("Note_type") or tok.startswith("Mode_") or tok.startswith("Key_"):
                continue

            # Duration tokens on their own are ignored — already handled
            if tok.startswith("Duration_"):
                continue

        # Finalize
        pm.instruments.append(instrument)
        pm.write(out_path)
        print(f"✅ Reconstructed audible MIDI saved → {out_path}")


# -------------------------------
# CLI
# -------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 remi_m_tokenizer.py <midi_file>")
        sys.exit(1)

    midi_file = sys.argv[1]
    tokenizer = REMIMTokenizer()

    print("Encoding MIDI...")
    tokens = tokenizer.midi_to_tokens(midi_file, mode="major", instrument=0)
    print("Token sequence length:", len(tokens))

    print("Decoding back to MIDI...")
    tokenizer.tokens_to_midi(tokens, out_path="reconstructed.mid")