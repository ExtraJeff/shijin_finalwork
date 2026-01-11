# -*- coding: utf-8 -*-
"""
Stage 4: Build LSTM Supervised Dataset
-------------------------------------
Input  : wordle_with_player_features.csv
Output : numpy arrays for LSTM training
"""

import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "03_feature_engineering",
    "output",
    "wordle_with_player_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "04_dataset_construction",
    "output"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Feature configuration
# =====================
SEQ_COL = "feedback_sequence"   # Wordle feedback sequence
TARGET_COL = "Trial"

PLAYER_FEATURES = [
    "hist_game_count",
    "hist_avg_trial",
    "hist_success_rate",
    "recent_avg_trial",
    "recent_success_rate",
    "recent_stability",  # Added stability feature
    "feedback_entropy"    # Added feedback entropy feature
]

# Word difficulty features
WORD_DIFFICULTY_FEATURES = [
    "num_vowels",
    "num_consonants",
    "avg_letter_frequency",
    "num_unique_letters",
    "has_repeated_letters",
    "total_letter_frequency"
]

# Combine all features
ALL_FEATURES = PLAYER_FEATURES + WORD_DIFFICULTY_FEATURES

SEQ_LEN = 7
WORD_LEN = 5

# =====================
# Main dataset builder
# =====================
def main():
    print("📥 Loading feature data...")
    df = pd.read_csv(INPUT_PATH)

    # Convert string to Python list
    df[SEQ_COL] = df[SEQ_COL].apply(ast.literal_eval)

    # Ensure temporal order per player
    df = df.sort_values(by=["Username", "Game"]).reset_index(drop=True)

    X_seq_list = []
    X_feat_list = []
    y_list = []

    print("🔄 Building supervised samples (t → t+1)...")

    for username, group in tqdm(df.groupby("Username")):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        for t in range(len(group) - 1):
            current_row = group.loc[t]
            next_row = group.loc[t + 1]

            # =====================
            # Sequence input
            # =====================
            seq_raw = current_row[SEQ_COL]
            seq_array = np.array(seq_raw, dtype=np.int32)

            # 🔴 核心修复点：强制 reshape 成 (7,5)
            if seq_array.shape == (SEQ_LEN * WORD_LEN,):
                seq_array = seq_array.reshape(SEQ_LEN, WORD_LEN)

            if seq_array.shape != (SEQ_LEN, WORD_LEN):
                raise ValueError(
                    f"Invalid feedback_sequence shape: {seq_array.shape} "
                    f"for user={username}, game_index={t}"
                )

            # =====================
            # Player features + Word difficulty features
            # =====================
            X_feat = current_row[ALL_FEATURES].values.astype(np.float32)

            # =====================
            # Target (next game)
            # =====================
            y = int(next_row[TARGET_COL])

            X_seq_list.append(seq_array)
            X_feat_list.append(X_feat)
            y_list.append(y)

    # =====================
    # Convert to numpy arrays
    # =====================
    X_seq = np.array(X_seq_list, dtype=np.int32)     # (N, 7, 5)
    X_feat = np.array(X_feat_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print("📊 Dataset summary:")
    print(f"  X_seq  shape: {X_seq.shape}")   # (N, 7, 5)
    print(f"  X_feat shape: {X_feat.shape}") # (N, F)
    print(f"  y      shape: {y.shape}")       # (N,)

    # =====================
    # Save datasets
    # =====================
    np.save(os.path.join(OUTPUT_DIR, "X_seq.npy"), X_seq)
    np.save(os.path.join(OUTPUT_DIR, "X_feat.npy"), X_feat)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("💾 LSTM dataset saved to 04_dataset_construction/output/")
    print("🎉 Stage 4 completed successfully!")

if __name__ == "__main__":
    main()
