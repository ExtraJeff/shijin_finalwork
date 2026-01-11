# -*- coding: utf-8 -*-
"""
Visualization utilities for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Use default font settings (English)

def save_fig(fig, filename, output_dir, dpi=300):
    """
    Save figure to file with consistent styling
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    plt.close(fig)

def plot_training_history(history, model_name, output_dir):
    """
    Plot training and validation loss/mae curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE curve - dynamically find the correct MAE keys
    mae_keys = [key for key in history.history.keys() if 'mae' in key and 'val_' not in key]
    val_mae_keys = [key for key in history.history.keys() if 'mae' in key and 'val_' in key]
    
    # Use the first found MAE keys
    train_mae_key = mae_keys[0] if mae_keys else 'mae'
    val_mae_key = val_mae_keys[0] if val_mae_keys else 'val_mae'
    
    ax2.plot(history.history[train_mae_key], label='Training MAE')
    ax2.plot(history.history[val_mae_key], label='Validation MAE')
    ax2.set_title(f'{model_name} - Training and Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, f'{model_name}_training_curves.png', output_dir)

def plot_prediction_distribution(y_true, y_pred, model_name, output_dir):
    """
    Plot distribution of predictions vs actual values
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction vs actual scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.1, s=1)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', alpha=0.8, label='Ideal Prediction (y=x)')
    
    # Add tolerance lines parallel to y=x
    tolerance = 0.5  # Adjust tolerance as needed
    ax1.plot([y_true.min(), y_true.max()], 
             [y_true.min() - tolerance, y_true.max() - tolerance], 
             'g--', alpha=0.6, label=f'Tolerance Band (y=x±{tolerance})')
    ax1.plot([y_true.min(), y_true.max()], 
             [y_true.min() + tolerance, y_true.max() + tolerance], 
             'g--', alpha=0.6)
    
    ax1.set_title(f'{model_name} - Predictions vs Actual Values')
    ax1.set_xlabel('Actual Attempts')
    ax1.set_ylabel('Predicted Attempts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error distribution
    errors = y_pred - y_true
    ax2.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title(f'{model_name} - Prediction Error Distribution')
    ax2.set_xlabel('Prediction Error (Predicted - Actual)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, f'{model_name}_prediction_analysis.png', output_dir)

def plot_metrics_summary(metrics, model_name, output_dir):
    """
    Plot a bar chart summary of metrics
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract metrics for plotting - removed R²
    metric_names = ['MAE', 'RMSE', 'Accuracy', 'F1-Score']
    metric_values = [metrics['mae'], metrics['rmse'], metrics['accuracy'], metrics['f1_score']]
    
    # Create bar chart
    bars = ax.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_title(f'{model_name} - Evaluation Metrics')
    ax.set_ylabel('Metric Value')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_fig(fig, f'{model_name}_metrics_summary.png', output_dir)

def plot_model_comparison(metrics_list, output_dir):
    """
    Plot comparison of MAE and RMSE across models
    """
    # Convert metrics_list to dictionaries
    models = [m['model'] for m in metrics_list]
    maes = [m['mae'] for m in metrics_list]
    rmses = [m['rmse'] for m in metrics_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MAE comparison
    bars1 = ax1.bar(models, maes, color='skyblue')
    ax1.set_title('Model Comparison - MAE')
    ax1.set_ylabel('MAE')
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # RMSE comparison  
    bars2 = ax2.bar(models, rmses, color='lightgreen')
    ax2.set_title('Model Comparison - RMSE')
    ax2.set_ylabel('RMSE')
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars2, rmses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_fig(fig, 'model_comparison_mae_rmse.png', output_dir)

def plot_error_by_player_activity(y_true, y_pred, player_features, model_name, output_dir):
    """
    Plot prediction errors by player activity level
    """
    # Create activity groups based on historical attempts
    # Using the first player feature (historical average attempts) as proxy for activity
    avg_attempts = player_features[:, 0]  # Assuming first feature is historical average attempts
    
    # Create activity bins
    bins = np.percentile(avg_attempts, [0, 25, 50, 75, 100])
    labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    activity_groups = np.digitize(avg_attempts, bins, right=True) - 1
    
    # Calculate errors for each group
    errors = []
    for i in range(4):
        group_mask = (activity_groups == i)
        if np.sum(group_mask) > 0:
            group_errors = np.abs(y_pred[group_mask] - y_true[group_mask])
            errors.append(group_errors)
    
    # Plot boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(errors, labels=labels)
    ax.set_title(f'{model_name} - Prediction Error by Player Activity Level')
    ax.set_xlabel('Player Activity Level')
    ax.set_ylabel('Absolute Prediction Error')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_fig(fig, f'{model_name}_error_by_activity.png', output_dir)

def plot_heatmap_correlation(y_true, y_pred, player_features, model_name, output_dir):
    """
    Plot heatmap of prediction errors vs player features
    """
    # Calculate errors
    errors = np.abs(y_pred - y_true)
    
    # Create correlation matrix
    feature_names = [f'Feature_{i+1}' for i in range(player_features.shape[1])] + ['Error']
    data = np.column_stack([player_features, errors])
    corr_matrix = np.corrcoef(data, rowvar=False)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=feature_names, yticklabels=feature_names, ax=ax)
    ax.set_title(f'{model_name} - Correlation Between Features and Prediction Error')
    
    plt.tight_layout()
    save_fig(fig, f'{model_name}_feature_correlation_heatmap.png', output_dir)


def plot_confusion_matrix(y_true, y_pred_class, y_pred_class_prob, model_name, output_dir):
    """
    Plot confusion matrix and ROC curve side by side
    """
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # =====================
    # Confusion Matrix
    # =====================
    cm = confusion_matrix(y_true, y_pred_class)
    
    # Plot confusion matrix with larger font size for annotations
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Failure', 'Success'], yticklabels=['Failure', 'Success'],
                annot_kws={'size': 20})  # Increase font size of numbers in confusion matrix
    ax1.set_title(f'{model_name} - Confusion Matrix', fontsize=16)
    ax1.set_xlabel('Predicted Label', fontsize=14)
    ax1.set_ylabel('True Label', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # =====================
    # ROC Curve
    # =====================
    fpr, tpr, _ = roc_curve(y_true, y_pred_class_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=14)
    ax2.set_ylabel('True Positive Rate', fontsize=14)
    ax2.set_title(f'{model_name} - ROC Curve', fontsize=16)
    ax2.legend(loc="lower right", fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    save_fig(fig, f'{model_name}_confusion_matrix_roc.png', output_dir)
