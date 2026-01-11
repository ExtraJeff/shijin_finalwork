# -*- coding: utf-8 -*-
"""
Stage 2: Data Preprocessing
---------------------------
1. Load raw Wordle data
2. Clean required fields
3. Encode feedback sequences (🟩🟨⬜ → 2/1/0)
4. Pad sequences to fixed length (7)
"""

import os
import ast
import pandas as pd
import numpy as np

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "01_raw_data", "wordle_games.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "02_data_preprocessing", "output", "wordle_preprocessed.csv")

MAX_TRIALS = 7  # unified sequence length

# =====================
# Feedback encoding
# =====================
FEEDBACK_MAP = {
    "🟩": 2,
    "🟨": 1,
    "⬜": 0
}

def encode_single_guess(guess_str):
    """Encode a single guess string into numeric list"""
    return [FEEDBACK_MAP[ch] for ch in guess_str]

def encode_feedback_sequence(feedback_list):
    """
    Convert a list of guess strings into a T x 5 matrix
    """
    encoded = [encode_single_guess(g) for g in feedback_list]
    return encoded

def pad_sequence(seq, max_len=7):
    """
    Pad or truncate feedback sequence to fixed length
    Shape: (max_len, 5)
    """
    seq = seq[:max_len]
    while len(seq) < max_len:
        seq.append([0, 0, 0, 0, 0])
    return seq

# =====================
# Main preprocessing
# =====================
def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Keep only necessary columns
    required_cols = [
        "Username",
        "Game",
        "Trial",
        "processed_text",
        "target"
    ]
    df = df[required_cols].dropna()

    print(f"✅ Raw records: {len(df)}")

    # Parse processed_text safely
    print("Parsing feedback sequences...")
    
    def parse_processed_text(text_str):
        """Parse processed_text string with space-separated guesses"""
        try:
            # Replace spaces with commas to make it a valid list of strings
            text_str = text_str.replace(' ', ',')
            return ast.literal_eval(text_str)
        except Exception as e:
            # Fallback to empty list if parsing fails
            return []
    
    df["processed_text"] = df["processed_text"].apply(parse_processed_text)

    # Encode and pad feedback sequences
    encoded_sequences = []
    for feedback in df["processed_text"]:
        # Skip empty sequences
        if not feedback:
            # Use all zeros for invalid sequences
            padded = [[0, 0, 0, 0, 0] for _ in range(MAX_TRIALS)]
        else:
            encoded = encode_feedback_sequence(feedback)
            padded = pad_sequence(encoded, MAX_TRIALS)
        encoded_sequences.append(padded)

    df["feedback_sequence"] = encoded_sequences

    # Convert to numpy-friendly format
    df["feedback_sequence"] = df["feedback_sequence"].apply(lambda x: np.array(x).flatten().tolist())

    # Sort by player and time
    df = df.sort_values(by=["Username", "Game"]).reset_index(drop=True)

    # Save result
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Preprocessed data saved to: {OUTPUT_PATH}")
    print("Stage 2 completed successfully!")

if __name__ == "__main__":
    main()