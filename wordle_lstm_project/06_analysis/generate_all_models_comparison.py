# -*- coding: utf-8 -*-
"""
Generate a comparison plot for all models using evaluation results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "07_results")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Load evaluation results
# =====================
results_file = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")
if not os.path.exists(results_file):
    print(f"❌ Results file not found at {results_file}")
    exit(1)

print(f"📥 Loading evaluation results from {results_file}")
results_df = pd.read_csv(results_file)

# =====================
# Generate comparison plots
# =====================

# 1. Model Comparison Bar Plot
print("Generating model comparison bar plot...")
fig, ax = plt.subplots(figsize=(12, 6))

# Select metrics to plot
metrics = ['mae', 'rmse', 'accuracy', 'f1_score', 'auc']
metric_names = ['MAE', 'RMSE', 'Accuracy', 'F1-Score', 'AUC']

# Create grouped bar plot
bar_width = 0.15
models = results_df['model']
x = np.arange(len(models))

for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    # Scale accuracy and F1-Score for better visualization
    if metric in ['accuracy', 'f1_score', 'auc']:
        values = results_df[metric]
    else:
        values = results_df[metric]
    ax.bar(x + i * bar_width, values, width=bar_width, label=metric_name)

# Add labels and titles
ax.set_xlabel('Model')
ax.set_ylabel('Metric Value')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Save plot
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "model_comparison_all_metrics_updated.png")
plt.savefig(save_path, dpi=300)
print(f"📊 Saved plot: {save_path}")
plt.close()

# 2. Boxplot-style comparison using metric values
print("Generating boxplot-style comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

# Create a boxplot where each box represents a model's performance across metrics
# We'll use a different approach since we don't have per-activity predictions

# Normalize metrics for better comparison
results_norm = results_df.copy()

# Normalize MAE and RMSE (lower is better)
results_norm['mae'] = (results_norm['mae'].max() - results_norm['mae']) / (results_norm['mae'].max() - results_norm['mae'].min())
results_norm['rmse'] = (results_norm['rmse'].max() - results_norm['rmse']) / (results_norm['rmse'].max() - results_norm['rmse'].min())

# Accuracy, F1-Score, AUC are already normalized (higher is better)

# Create a dataframe for boxplot
boxplot_data = []
for idx, row in results_norm.iterrows():
    model_data = [row['mae'], row['rmse'], row['accuracy'], row['f1_score'], row['auc']]
    boxplot_data.append(model_data)

# Create boxplot
ax.boxplot(boxplot_data, labels=results_df['model'])

# Add labels and titles
ax.set_xlabel('Model')
ax.set_ylabel('Normalized Metric Value')
ax.set_title('Model Performance Distribution Across Metrics')
ax.grid(True, alpha=0.3, axis='y')

# Save plot
plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "model_performance_boxplot.png")
plt.savefig(save_path, dpi=300)
print(f"📊 Saved plot: {save_path}")
plt.close()

# 3. Metric-specific comparison plots
for metric, metric_name in zip(metrics, metric_names):
    print(f"📊 Generating {metric_name} comparison plot...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort models by metric value (ascending for MAE and RMSE, descending for others)
    if metric in ['mae', 'rmse']:
        sorted_df = results_df.sort_values(by=metric, ascending=True)
    else:
        sorted_df = results_df.sort_values(by=metric, ascending=False)
    
    # Create bar plot
    bars = ax.bar(sorted_df['model'], sorted_df[metric], color='skyblue')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, sorted_df[metric]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Add labels and titles
    ax.set_xlabel('Model')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison Across Models')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save plot
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"model_comparison_{metric}.png")
    plt.savefig(save_path, dpi=300)
    print(f"📊 Saved plot: {save_path}")
    plt.close()

print("All comparison plots generated successfully!")
print(f"📋 Plots saved to: {OUTPUT_DIR}")