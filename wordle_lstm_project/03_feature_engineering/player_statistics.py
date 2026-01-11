# -*- coding: utf-8 -*-
"""
Stage 3: Player Feature Engineering
-----------------------------------
1. Compute player historical statistics
2. Add rolling features (recent N games)
3. Assign player activity level
4. Add stability and entropy features
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import entropy

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(
    PROJECT_ROOT, "02_data_preprocessing", "output", "wordle_preprocessed.csv"
)
OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "03_feature_engineering", "output", "wordle_with_player_features.csv"
)

RECENT_N = 5  # rolling window size

# =====================
# Helper functions
# =====================
def assign_activity_level(total_games):
    """
    Assign player activity level based on total games played
    """
    if total_games <= 5:
        return "newbie"
    elif total_games <= 20:
        return "casual"
    elif total_games <= 50:
        return "active"
    elif total_games <= 100:
        return "veteran"
    else:
        return "master"

# English letter frequencies (approximate)
LETTER_FREQUENCY = {
    'A': 8.167, 'B': 1.492, 'C': 2.782, 'D': 4.253, 'E': 12.702,
    'F': 2.228, 'G': 2.015, 'H': 6.094, 'I': 6.966, 'J': 0.153,
    'K': 0.772, 'L': 4.025, 'M': 2.406, 'N': 6.749, 'O': 7.507,
    'P': 1.929, 'Q': 0.095, 'R': 5.987, 'S': 6.327, 'T': 9.056,
    'U': 2.758, 'V': 0.978, 'W': 2.360, 'X': 0.150, 'Y': 1.974, 'Z': 0.074
}

def calculate_word_difficulty(word):
    """
    Calculate word difficulty features based on various metrics
    """
    word = word.upper()
    
    # Number of vowels
    vowels = set('AEIOU')
    num_vowels = sum(1 for c in word if c in vowels)
    
    # Number of consonants
    num_consonants = 5 - num_vowels
    
    # Average letter frequency
    avg_frequency = sum(LETTER_FREQUENCY.get(c, 0) for c in word) / 5
    
    # Number of unique letters
    num_unique = len(set(word))
    
    # Has repeated letters
    has_repeats = 1 if len(word) != num_unique else 0
    
    # Sum of letter frequencies (higher sum = easier word)
    total_frequency = sum(LETTER_FREQUENCY.get(c, 0) for c in word)
    
    return {
        'word_length': 5,  # All words are 5 letters in Wordle
        'num_vowels': num_vowels,
        'num_consonants': num_consonants,
        'avg_letter_frequency': avg_frequency,
        'num_unique_letters': num_unique,
        'has_repeated_letters': has_repeats,
        'total_letter_frequency': total_frequency
    }

# =====================
# Main feature engineering
# =====================
def main():
    print("Loading preprocessed data...")
    df = pd.read_csv(INPUT_PATH)

    # Ensure correct order
    df = df.sort_values(by=["Username", "Game"]).reset_index(drop=True)

    # Success indicator
    df["is_success"] = (df["Trial"] <= 6).astype(int)

    # Containers for features
    feature_rows = []

    print("Computing player historical features...")

    # Precompute word difficulty features for all targets
    print("Precomputing word difficulty features...")
    word_difficulty_cache = {}
    for word in df["target"].unique():
        word_difficulty_cache[word] = calculate_word_difficulty(word)

    for username, group in df.groupby("Username"):
        group = group.reset_index(drop=True)

        total_games = len(group)
        activity_level = assign_activity_level(total_games)

        # Cumulative statistics (shifted to avoid leakage)
        cum_games = np.arange(total_games)
        cum_avg_trial = group["Trial"].expanding().mean().shift(1)
        cum_success_rate = group["is_success"].expanding().mean().shift(1)

        # Rolling statistics
        rolling_avg_trial = (
            group["Trial"]
            .rolling(window=RECENT_N, min_periods=1)
            .mean()
            .shift(1)
        )

        rolling_success_rate = (
            group["is_success"]
            .rolling(window=RECENT_N, min_periods=1)
            .mean()
            .shift(1)
        )
        
        # Rolling stability feature (standard deviation of recent trials)
        rolling_stability = (
            group["Trial"]
            .rolling(window=RECENT_N, min_periods=1)
            .std()
            .shift(1)
        )
        # Fill NaN with 0 for first game
        rolling_stability = rolling_stability.fillna(0)

        for i in range(total_games):
            row = group.loc[i].to_dict()
            target_word = row["target"]
            
            # Calculate feedback sequence entropy if processed_text is available
            feedback_entropy = 0.0
            if "processed_text" in row and pd.notna(row["processed_text"]):
                # Count the frequency of each feedback type (assuming processed_text has encoded values)
                feedback_seq = str(row["processed_text"])
                if len(feedback_seq) > 0:
                    # Count frequencies of each character
                    freq = {}
                    for c in feedback_seq:
                        freq[c] = freq.get(c, 0) + 1
                    # Calculate entropy
                    values = list(freq.values())
                    if len(values) > 1:  # Need at least two different values to calculate entropy
                        feedback_entropy = entropy(values, base=2)
            
            # Historical features (fill NaN for first game)
            row["hist_game_count"] = cum_games[i]
            row["hist_avg_trial"] = (
                cum_avg_trial[i] if not np.isnan(cum_avg_trial[i]) else 0.0
            )
            row["hist_success_rate"] = (
                cum_success_rate[i] if not np.isnan(cum_success_rate[i]) else 0.0
            )
            row["recent_avg_trial"] = (
                rolling_avg_trial[i] if not np.isnan(rolling_avg_trial[i]) else 0.0
            )
            row["recent_success_rate"] = (
                rolling_success_rate[i] if not np.isnan(rolling_success_rate[i]) else 0.0
            )
            row["recent_stability"] = rolling_stability[i]
            
            # Feedback sequence features
            row["feedback_entropy"] = feedback_entropy
            
            # Static player feature
            row["activity_level"] = activity_level

            # Add word difficulty features
            word_diff = word_difficulty_cache[target_word]
            row.update(word_diff)

            feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)

    # Save result
    feature_df.to_csv(OUTPUT_PATH, index=False)
    print(f"💾 Player feature data saved to: {OUTPUT_PATH}")
    print("Stage 3 completed successfully!")

if __name__ == "__main__":
    main()
