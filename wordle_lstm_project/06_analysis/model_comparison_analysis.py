# -*- coding: utf-8 -*-
"""
Model comparison and error analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path for visualization functions
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from vis_utils import save_fig, plot_model_comparison

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "07_results")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_DIR = os.path.join(PROJECT_ROOT, "04_dataset_construction", "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Load data
# =====================
def load_evaluation_results():
    """
    Load evaluation results from CSV file
    """
    results_path = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")
    if not os.path.exists(results_path):
        print(f"❌ Evaluation results not found at {results_path}")
        print("   Please run the model training scripts first.")
        return None
    
    results_df = pd.read_csv(results_path)
    print(f"📥 Loaded evaluation results for {len(results_df)} models")
    return results_df

def load_test_data():
    """
    Load test data for further analysis
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
    from sklearn.model_selection import train_test_split
    
    X_seq_train, X_seq_temp, X_feat_train, X_feat_temp, y_train, y_temp = train_test_split(
        X_seq, X_feat, y, test_size=0.3, random_state=42
    )
    
    X_seq_val, X_seq_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
        X_seq_temp, X_feat_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_seq_test, X_feat_test, y_test

# =====================
# Generate model comparison plots
# =====================
def generate_model_comparison_plots(results_df):
    """
    Generate comparison plots for different models
    """
    print("Generating model comparison plots...")
    
    # Convert dataframe to list of dictionaries for plotting
    metrics_list = results_df.to_dict('records')
    
    # Plot MAE and RMSE comparison
    plot_model_comparison(metrics_list, OUTPUT_DIR)
    
    # Generate additional comparison plots
    
    # 1. Bar plot comparing all metrics
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set width of bars
    bar_width = 0.15
    
    # Set positions of bars on x-axis
    r1 = np.arange(len(results_df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    
    # Make the plot
    bars1 = ax.bar(r1, results_df['mae'], width=bar_width, label='MAE')
    bars2 = ax.bar(r2, results_df['rmse'], width=bar_width, label='RMSE')
    
    # Initialize bars list with basic metrics
    all_bars = [bars1, bars2]
    next_pos = r2
    
    # Add Accuracy
    r3 = [x + bar_width for x in next_pos]
    bars3 = ax.bar(r3, results_df['accuracy'], width=bar_width, label='Accuracy')
    all_bars.append(bars3)
    next_pos = r3
    
    # Add F1-Score if available
    if 'f1_score' in results_df.columns:
        r4 = [x + bar_width for x in next_pos]
        bars4 = ax.bar(r4, results_df['f1_score'], width=bar_width, label='F1-Score')
        all_bars.append(bars4)
        next_pos = r4
    
    # Add AUC if available
    if 'auc' in results_df.columns:
        r5 = [x + bar_width for x in next_pos]
        bars5 = ax.bar(r5, results_df['auc'], width=bar_width, label='AUC')
        all_bars.append(bars5)
    
    # Add labels and title
    ax.set_title('Comprehensive Model Comparison', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    
    # Set x-ticks position based on the number of bars
    ax.set_xticks([r + (len(all_bars)-1)*bar_width/2 for r in range(len(results_df))])
    ax.set_xticklabels(results_df['model'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bars in all_bars:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    save_fig(fig, 'model_comparison_all_metrics.png', OUTPUT_DIR)
    
    # 2. Radar chart for model comparison
    from math import pi
    
    # Select relevant metrics (only include available columns)
    available_metrics = ['mae', 'rmse', 'accuracy']
    optional_metrics = ['f1_score', 'auc']
    
    # Add optional metrics if they exist and have valid data
    metrics = available_metrics.copy()
    for metric in optional_metrics:
        if metric in results_df.columns and not results_df[metric].isnull().all():
            metrics.append(metric)
    
    # Normalize metrics for radar chart (since some can be negative)
    results_norm = results_df.copy()
    
    # Normalize MAE and RMSE (invert since lower is better)
    results_norm['mae'] = (results_norm['mae'].max() - results_norm['mae']) / (results_norm['mae'].max() - results_norm['mae'].min())
    results_norm['rmse'] = (results_norm['rmse'].max() - results_norm['rmse']) / (results_norm['rmse'].max() - results_norm['rmse'].min())
    
    # Create radar chart
    categories = [metric.upper() for metric in metrics]
    N = len(categories)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Set angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the plot
    
    # Plot each model
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'purple', 'orange']
    for i, (idx, row) in enumerate(results_df.iterrows()):
        values = results_norm.loc[idx, metrics].tolist()
        values += values[:1]  # Close the plot
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=results_df.loc[idx, 'model'])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add y-axis labels
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    ax.set_title('Model Comparison Radar Chart', y=1.1, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    save_fig(fig, 'model_comparison_radar.png', OUTPUT_DIR)

# =====================
# Generate error analysis plots
# =====================
def generate_error_analysis_plots(X_seq_test, X_feat_test, y_test):
    """
    Generate error analysis plots by player activity level
    """
    print("Generating error analysis plots...")
    
    # Load predictions from all models
    model_predictions = {}
    
    # Load each model and get predictions
    models_dirs = [
        ("lstm", os.path.join(PROJECT_ROOT, "05_models", "lstm")),
        ("bilstm", os.path.join(PROJECT_ROOT, "05_models", "bilstm")),
        ("lstm_attention", os.path.join(PROJECT_ROOT, "05_models", "lstm_attention")),
        ("transformer", os.path.join(PROJECT_ROOT, "05_models", "transformer"))
    ]
    
    # Try to load all models, but continue if some fail
    for model_name, model_dir in models_dirs:
        try:
            # Skip complex models that are causing issues
            if model_name in ["lstm_attention", "transformer"]:
                print(f"⚠️  Skipping {model_name} model due to deserialization issues")
                continue
            
            import tensorflow as tf
            
            # Add model directory to path
            sys.path.append(model_dir)
            
            # Load the model
            model_path = os.path.join(model_dir, "output", f"{model_name}_main_model.keras")
            
            if not os.path.exists(model_path):
                print(f"⚠️  Model file not found for {model_name} at {model_path}")
                continue
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Get predictions (multi-task model returns [regression_output, classification_output])
            y_pred = model.predict([X_seq_test, X_feat_test], batch_size=128, verbose=0)
            y_pred_reg = y_pred[0]  # First output is regression (trial count)
            y_pred_reg = y_pred_reg.flatten()
            
            model_predictions[model_name] = y_pred_reg
            print(f"✅ Loaded predictions for {model_name}")
            
            # Clear memory
            del model
        except Exception as e:
            print(f"❌ Failed to load {model_name} model: {e}")
            continue
    
    # Add model evaluation results to the plot for all models
    # We'll use the evaluation metrics to enhance the visualization
    print("Loading model evaluation results...")
    results_df = pd.read_csv(os.path.join(PROJECT_ROOT, "07_results", "model_evaluation_results.csv"))
    print(f"✅ Loaded evaluation results for {len(results_df)} models")
    
    # Only generate plots if we have predictions
    if not model_predictions:
        print("No model predictions loaded, skipping error analysis plots")
        return
    
    # Create activity groups
    avg_attempts = X_feat_test[:, 0]  # Assuming first feature is historical average attempts
    bins = np.percentile(avg_attempts, [0, 25, 50, 75, 100])
    labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    activity_groups = np.digitize(avg_attempts, bins, right=True) - 1
    
    # Generate error by activity level for all models
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect errors for each model and activity group
    errors_by_model = {}
    for model_name, y_pred in model_predictions.items():
        errors = []
        for i in range(4):
            group_mask = (activity_groups == i)
            if np.sum(group_mask) > 0:
                group_errors = np.abs(y_pred[group_mask] - y_test[group_mask])
                errors.append(group_errors)
        errors_by_model[model_name] = errors
    
    # Create boxplot - ensure all boxes have the same number of groups
    model_names = list(errors_by_model.keys())
    n_groups = len(model_names)
    
    # Use a different approach for boxplot that handles varying group sizes
    # Create a box plot for each model's errors across activity levels
    positions = np.arange(1, n_groups + 1) * 2  # Space models apart
    
    for i, (model_name, errors) in enumerate(errors_by_model.items()):
        pos = positions[i] + np.arange(len(errors)) * 0.3  # Space activity levels within each model
        bp = ax.boxplot(errors, positions=pos, widths=0.25, patch_artist=True, 
                       boxprops=dict(facecolor=f'C{i}', alpha=0.7))
        
        # Add label for the model at the center position
        if i == 0:
            for j, label in enumerate(labels):
                ax.text(pos[j], ax.get_ylim()[1] * 1.02, label, ha='center', va='bottom', fontsize=9)
    
    # Set x-axis labels and ticks
    ax.set_xticks(positions + (len(labels) - 1) * 0.15)
    ax.set_xticklabels(model_names, rotation=45)
    
    # Add titles and labels
    ax.set_title('Prediction Error by Player Activity Level', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Absolute Prediction Error', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    save_fig(fig, 'error_by_activity_all_models.png', OUTPUT_DIR)
    
    # Generate heatmap of error correlation with features
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (model_name, y_pred) in enumerate(model_predictions.items()):
        # Calculate errors
        errors = np.abs(y_pred - y_test)
        
        # Create correlation matrix with actual feature names
        actual_feature_names = [
            'hist_game_count', 'hist_avg_trial', 'hist_success_rate',
            'recent_avg_trial', 'recent_success_rate', 'recent_stability',
            'feedback_entropy', 'num_vowels', 'num_consonants',
            'avg_letter_frequency', 'num_unique_letters', 'has_repeated_letters',
            'total_letter_frequency'
        ]
        feature_names = actual_feature_names + ['Error']
        data = np.column_stack([X_feat_test, errors])
        corr_matrix = np.corrcoef(data, rowvar=False)
        
        # Plot heatmap
        ax = axes[i]
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    xticklabels=feature_names, yticklabels=feature_names, ax=ax)
        ax.set_title(f'{model_name} - Feature vs Error Correlation', fontsize=12)
        
    # Remove extra subplots if there are fewer models
    for i in range(len(model_predictions), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, 'feature_error_correlation_heatmaps.png', OUTPUT_DIR)

# =====================
# Main function
# =====================
def main():
    """
    Main function to generate all analysis plots
    """
    print("Starting model comparison and error analysis...")
    
    # Load evaluation results
    results_df = load_evaluation_results()
    if results_df is None:
        return
    
    # Load test data
    X_seq_test, X_feat_test, y_test = load_test_data()
    
    # Generate model comparison plots
    generate_model_comparison_plots(results_df)
    
    # Generate error analysis plots
    generate_error_analysis_plots(X_seq_test, X_feat_test, y_test)
    
    print(f"🎉 All analysis plots generated and saved to {OUTPUT_DIR}")
    print("Generated plots:")
    print("   - model_comparison_mae_rmse.png")
    print("   - model_comparison_all_metrics.png")
    print("   - model_comparison_radar.png")
    print("   - error_by_activity_all_models.png")
    # print("   - feature_error_correlation_heatmaps.png")

if __name__ == "__main__":
    main()
