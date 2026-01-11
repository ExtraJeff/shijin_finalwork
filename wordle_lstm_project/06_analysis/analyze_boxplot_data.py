# -*- coding: utf-8 -*-
"""
Analyze the data used to generate the boxplot for LSTM error by activity level
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "04_dataset_construction", "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "05_models", "lstm")

# =====================
# Load test data
# =====================
def load_test_data():
    """
    Load and prepare test data
    """
    print("Loading test data...")
    
    # Load full dataset
    X_seq = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
    X_feat = np.load(os.path.join(DATA_DIR, "X_feat.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    y = y.astype("float32")
    
    # Reshape sequences to (seq_len, seq_dim)
    seq_len = 7
    seq_dim = 5
    X_seq = X_seq.reshape(-1, seq_len, seq_dim)
    
    # Split into train/val/test (same as training script)
    X_seq_train, X_seq_temp, X_feat_train, X_feat_temp, y_train, y_temp = train_test_split(
        X_seq, X_feat, y, test_size=0.3, random_state=42
    )
    
    X_seq_val, X_seq_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
        X_seq_temp, X_feat_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"📊 Dataset split:")
    print(f" Train: {X_seq_train.shape[0]}")
    print(f" Val  : {X_seq_val.shape[0]}")
    print(f" Test : {X_seq_test.shape[0]}")
    
    return X_seq_test, X_feat_test, y_test

# =====================
# Load model and generate predictions
# =====================
def load_model_and_predict(X_seq_test, X_feat_test):
    """
    Load LSTM model and generate predictions
    """
    print("\nLoading LSTM model...")
    
    model_path = os.path.join(MODEL_DIR, "output", "lstm_main_model.keras")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return None
    
    model = tf.keras.models.load_model(model_path)
    
    print("Generating predictions...")
    y_pred_reg, y_pred_class_prob = model.predict([X_seq_test, X_feat_test], batch_size=128, verbose=1)
    y_pred_reg = y_pred_reg.flatten()
    
    return y_pred_reg

# =====================
# Calculate error by activity level
# =====================
def calculate_error_by_activity(y_test, y_pred_reg, X_feat_test):
    """
    Calculate prediction errors grouped by player activity level
    """
    print("\nCalculating error by activity level...")
    
    # Create activity groups based on historical average attempts
    # Using the first player feature (historical average attempts) as proxy for activity
    avg_attempts = X_feat_test[:, 0]  # Historical average attempts
    
    # Create activity bins using percentiles
    bins = np.percentile(avg_attempts, [0, 25, 50, 75, 100])
    labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    
    print(f"\n📋 Activity level bins:")
    for i in range(len(bins)-1):
        print(f"   {labels[i]}: {bins[i]:.4f} - {bins[i+1]:.4f}")
    
    # Assign each sample to an activity group
    activity_groups = np.digitize(avg_attempts, bins, right=True) - 1
    
    # Calculate absolute errors
    errors = np.abs(y_pred_reg - y_test)
    
    # Group errors by activity level and calculate statistics
    activity_stats = []
    
    for i in range(4):
        group_mask = (activity_groups == i)
        group_errors = errors[group_mask]
        group_size = len(group_errors)
        
        if group_size > 0:
            # Calculate statistics
            mean_error = np.mean(group_errors)
            median_error = np.median(group_errors)
            q1 = np.percentile(group_errors, 25)
            q3 = np.percentile(group_errors, 75)
            iqr = q3 - q1
            min_error = np.min(group_errors)
            max_error = np.max(group_errors)
            std_error = np.std(group_errors)
            
            # Calculate outliers (beyond 1.5*IQR from Q1 and Q3)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = group_errors[(group_errors < lower_bound) | (group_errors > upper_bound)]
            num_outliers = len(outliers)
            
            stats = {
                'level': labels[i],
                'size': group_size,
                'mean': mean_error,
                'median': median_error,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'min': min_error,
                'max': max_error,
                'std': std_error,
                'outliers': num_outliers,
                'outlier_ratio': num_outliers / group_size if group_size > 0 else 0
            }
            
            activity_stats.append(stats)
            
            print(f"\n📊 {labels[i]} Activity Level:")
            print(f"   Samples: {group_size}")
            print(f"   Mean Error: {mean_error:.4f}")
            print(f"   Median Error: {median_error:.4f}")
            print(f"   Q1: {q1:.4f}, Q3: {q3:.4f}, IQR: {iqr:.4f}")
            print(f"   Min: {min_error:.4f}, Max: {max_error:.4f}")
            print(f"   Std Dev: {std_error:.4f}")
            print(f"   Outliers: {num_outliers} ({stats['outlier_ratio']:.2%})")
    
    return activity_stats

# =====================
# Main function
# =====================
def main():
    """
    Main function to analyze boxplot data
    """
    print("Starting boxplot data analysis...")
    
    # Load test data
    X_seq_test, X_feat_test, y_test = load_test_data()
    
    # Load model and generate predictions
    y_pred_reg = load_model_and_predict(X_seq_test, X_feat_test)
    if y_pred_reg is None:
        print("Failed to load model, exiting...")
        return
    
    # Calculate overall error metrics
    overall_mae = mean_absolute_error(y_test, y_pred_reg)
    print(f"\n📋 Overall Model Performance:")
    print(f"   MAE: {overall_mae:.4f}")
    
    # Calculate error by activity level
    activity_stats = calculate_error_by_activity(y_test, y_pred_reg, X_feat_test)
    
    # Print summary
    print(f"\n🎉 Analysis completed!")
    print(f"\n📋 Summary of Error by Activity Level:")
    print(f"{'Level':<15} {'Samples':<10} {'Mean Error':<12} {'Median Error':<15} {'Outliers':<10}")
    print("-" * 60)
    for stats in activity_stats:
        print(f"{stats['level']:<15} {stats['size']:<10} {stats['mean']:<12.4f} {stats['median']:<15.4f} {stats['outliers']:<10}")

if __name__ == "__main__":
    main()