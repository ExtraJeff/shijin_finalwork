# -*- coding: utf-8 -*-
"""
Generate confusion matrix and ROC curve for trained models
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score

# Add utils to path for visualization functions
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from vis_utils import plot_confusion_matrix

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "04_dataset_construction", "output")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "07_results")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Model configuration
# =====================
models_to_process = [
    "lstm",
    "bilstm",
    "lstm_attention",
    "transformer"
]

# Sequence parameters
seq_len = 7
seq_dim = 5

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
    
    # Generate classification labels: 1 if success (trial <= 6), 0 otherwise
    y_class = (y <= 6).astype("float32")
    
    # Reshape sequences to (seq_len, seq_dim)
    X_seq = X_seq.reshape(-1, seq_len, seq_dim)
    
    # Split into train/val/test (same as training script)
    from sklearn.model_selection import train_test_split
    
    X_seq_train, X_seq_temp, X_feat_train, X_feat_temp, y_train, y_temp, y_class_train, y_class_temp = train_test_split(
        X_seq, X_feat, y, y_class, test_size=0.3, random_state=42
    )
    
    X_seq_val, X_seq_test, X_feat_val, X_feat_test, y_val, y_test, y_class_val, y_class_test = train_test_split(
        X_seq_temp, X_feat_temp, y_temp, y_class_temp, test_size=0.5, random_state=42
    )
    
    print(f"📊 Test data loaded: {X_seq_test.shape[0]} samples")
    return X_seq_test, X_feat_test, y_test, y_class_test

# =====================
# Load model and generate predictions
# =====================
def process_model(model_name, X_seq_test, X_feat_test, y_test, y_class_test):
    """
    Load trained model, generate predictions, and create visualizations
    """
    try:
        print(f"\n🔄 Processing {model_name} model...")
        
        # Import custom layers if needed
        if model_name == "lstm_attention":
            # Add model directory to path
            model_dir = os.path.join(PROJECT_ROOT, "05_models", model_name)
            sys.path.append(model_dir)
            from lstm_attention_model import AttentionLayer
        elif model_name == "transformer":
            # Add model directory to path
            model_dir = os.path.join(PROJECT_ROOT, "05_models", model_name)
            sys.path.append(model_dir)
            from transformer_model import PositionalEncoding, MultiHeadAttentionLayer, TransformerEncoderLayer
        
        # Load the model
        model_dir = os.path.join(PROJECT_ROOT, "05_models", model_name)
        model_path = os.path.join(model_dir, "output", f"{model_name}_main_model.keras")
        
        # Adjust model path for specific models
        if model_name in ["lstm_attention", "transformer"]:
            model_path = os.path.join(model_dir, "output", f"{model_name}_model.keras")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found for {model_name} at {model_path}")
            return None
        
        print(f"📥 Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Make predictions
        print(f"🔮 Generating predictions for {model_name}...")
        y_pred_reg, y_pred_class_prob = model.predict([X_seq_test, X_feat_test], batch_size=128, verbose=1)
        y_pred_reg = y_pred_reg.flatten()
        y_pred_class = (y_pred_class_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        print(f"📊 Calculating metrics for {model_name}...")
        mae = mean_absolute_error(y_test, y_pred_reg)
        mse = mean_squared_error(y_test, y_pred_reg)
        rmse = np.sqrt(mse)
        
        y_true_class = y_class_test.astype(int)
        accuracy = accuracy_score(y_true_class, y_pred_class)
        f1 = f1_score(y_true_class, y_pred_class)
        auc = roc_auc_score(y_true_class, y_pred_class_prob.flatten())
        
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc
        }
        
        print(f"✅ {model_name} metrics:")
        print(f"   MAE: {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC: {auc:.4f}")
        
        # Generate confusion matrix and ROC curve
        print(f"📊 Generating confusion matrix and ROC curve for {model_name}...")
        plot_confusion_matrix(y_true_class, y_pred_class, y_pred_class_prob.flatten(), model_name, OUTPUT_DIR)
        
        # Clear memory
        del model
        
        return metrics
        
    except Exception as e:
        print(f"❌ Failed to process {model_name} model: {e}")
        return None

# =====================
# Main function
# =====================
def main():
    """
    Main function to generate confusion matrix and ROC curve for all models
    """
    print("Generating confusion matrix and ROC curve for trained models...")
    
    # Load test data
    X_seq_test, X_feat_test, y_test, y_class_test = load_test_data()
    
    # Process each model
    all_metrics = {}
    for model_name in models_to_process:
        metrics = process_model(model_name, X_seq_test, X_feat_test, y_test, y_class_test)
        if metrics:
            all_metrics[model_name] = metrics
    
    # Print summary
    print("\nSummary of generated visualizations:")
    for model_name in all_metrics:
        print(f"✅ Generated confusion matrix and ROC curve for {model_name}")
    
    print("\nAll models processed! Confusion matrices and ROC curves have been generated.")

if __name__ == "__main__":
    main()