# -*- coding: utf-8 -*-
"""
Train BiLSTM model with visualization
"""

import os
import numpy as np
import tensorflow as tf
import wandb

# Set CPU cores for training
# Use 16 cores for both intra and inter operation parallelism
tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

from sklearn.model_selection import train_test_split
from bilstm_model import build_bilstm_model

# Add utils to path for visualization functions
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
from vis_utils import (
    plot_training_history,
    plot_prediction_distribution,
    plot_metrics_summary,
    plot_error_by_player_activity
)

# =====================
# Path configuration
# =====================
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODEL_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get project root for data path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(
    PROJECT_ROOT, "04_dataset_construction", "output"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "07_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
# Initialize WandB
# =====================
print("Initializing WandB...")
wandb.init(
    # 使用正确的entity名称
    entity="legendjeff-nju",
    # 使用新的project名称
    project="shijin",
    config={
        "model_type": "bilstm",
        "seq_len": 7,
        "seq_dim": 5,
        "embed_dim": 8,
        "lstm_units": 128,
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 0.0003,
        "patience": 5
    },
    name="bilstm-train-" + os.path.basename(MODEL_DIR)
)

# =====================
# Load data
# =====================
print("Loading dataset...")

X_seq = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
X_feat = np.load(os.path.join(DATA_DIR, "X_feat.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

y = y.astype("float32")

# Generate classification labels: 1 if success (trial <= 6), 0 otherwise
y_class = (y <= 6).astype("float32")

# =====================
# Reshape sequences to (seq_len, seq_dim)
# =====================
seq_len = 7
seq_dim = 5
X_seq = X_seq.reshape(-1, seq_len, seq_dim)

# =====================
# Train / val / test split
# =====================
X_seq_train, X_seq_temp, X_feat_train, X_feat_temp, y_train, y_temp, y_class_train, y_class_temp = train_test_split(
    X_seq, X_feat, y, y_class, test_size=0.3, random_state=42
)

X_seq_val, X_seq_test, X_feat_val, X_feat_test, y_val, y_test, y_class_val, y_class_test = train_test_split(
    X_seq_temp, X_feat_temp, y_temp, y_class_temp, test_size=0.5, random_state=42
)

print("Dataset split:")
print(f" Train: {X_seq_train.shape[0]}")
print(f" Val  : {X_seq_val.shape[0]}")
print(f" Test : {X_seq_test.shape[0]}")

# =====================
# Build model
# =====================
model = build_bilstm_model(
    seq_len=seq_len,
    seq_dim=seq_dim,
    embed_dim=8,
    lstm_units=128,
    num_player_features=X_feat.shape[1]
)

model.summary()

# =====================
# Callbacks
# =====================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    # WandB callback for automatic logging
    wandb.keras.WandbCallback(save_model=False, save_graph=False)
]

# =====================
# Train
# =====================
print("Training BiLSTM model...")

history = model.fit(
    [X_seq_train, X_feat_train],
    {"reg_output": y_train, "class_output": y_class_train},
    validation_data=([X_seq_val, X_feat_val], {"reg_output": y_val, "class_output": y_class_val}),
    epochs=30,
    batch_size=128,
    callbacks=callbacks
)

# =====================
# Evaluate comprehensively
# =====================
print("Evaluating comprehensively on test set...")

# Make predictions
y_pred_reg, y_pred_class_prob = model.predict([X_seq_test, X_feat_test], batch_size=128, verbose=1)
y_pred_reg = y_pred_reg.flatten()
y_pred_class = (y_pred_class_prob > 0.5).astype(int).flatten()

# Calculate metrics from sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score

mae = mean_absolute_error(y_test, y_pred_reg)
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)

# Classification metrics (success/failure)
y_true_class = y_class_test.astype(int)
accuracy = accuracy_score(y_true_class, y_pred_class)
f1 = f1_score(y_true_class, y_pred_class)
auc = roc_auc_score(y_true_class, y_pred_class_prob.flatten())

# Print all metrics
print("Evaluation metrics:")
print(f"   mae: {mae:.4f}")
print(f"   mse: {mse:.4f}")
print(f"   rmse: {rmse:.4f}")
print(f"   accuracy: {accuracy:.4f}")
print(f"   f1_score: {f1:.4f}")
print(f"   auc: {auc:.4f}")

# Log final metrics to WandB
print("Logging final metrics to WandB...")
wandb.log({
    "final_mae": mae,
    "final_mse": mse,
    "final_rmse": rmse,
    "final_accuracy": accuracy,
    "final_f1_score": f1,
    "final_auc": auc
})

# Save evaluation results to file
metrics = {
    "model": "bilstm",
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "accuracy": accuracy,
    "f1_score": f1,
    "auc": auc
}

import pandas as pd
results_df = pd.DataFrame([metrics])
results_path = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")

# Append to existing file if it exists
if os.path.exists(results_path):
    existing_df = pd.read_csv(results_path)
    # Remove existing entry for this model if it exists
    existing_df = existing_df[existing_df["model"] != metrics["model"]]
    results_df = pd.concat([existing_df, results_df], ignore_index=True)

results_df.to_csv(results_path, index=False)
print(f"💾 Results saved to: {results_path}")

# =====================
# Save model
# =====================
model_path = os.path.join(
    OUTPUT_DIR,
    "bilstm_main_model.keras"
)

model.save(model_path)
print(f"💾 Model saved to: {model_path}")

# =====================
# Generate visualizations
# =====================
print("Generating visualizations...")

# Plot training history
plot_training_history(history, "bilstm", OUTPUT_DIR)

# Plot prediction distribution and errors
plot_prediction_distribution(y_test, y_pred_reg, "bilstm", OUTPUT_DIR)

# Plot metrics summary
plot_metrics_summary(metrics, "bilstm", OUTPUT_DIR)

# Plot error by player activity
plot_error_by_player_activity(y_test, y_pred_reg, X_feat_test, "bilstm", OUTPUT_DIR)

# Finish WandB run
print("Finishing WandB run...")
wandb.finish()

print("BiLSTM model training and evaluation completed!")
