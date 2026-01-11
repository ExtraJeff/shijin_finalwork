# -*- coding: utf-8 -*-
"""
Simple Attention Weights Analysis for LSTM-Attention Model
This script creates synthetic attention weights and visualizes them to demonstrate the concept.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# Path configuration
# =====================
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Generate synthetic attention weights
# =====================
def generate_synthetic_attention_weights():
    """Generate synthetic attention weights based on typical patterns"""
    print("Generating synthetic attention weights...")
    
    # Number of samples and sequence length
    num_samples = 1000
    seq_len = 7
    
    # Generate attention weights with a peak at the 3rd step
    # This simulates the scenario where the 3rd attempt has the highest attention
    attention_weights = np.zeros((num_samples, seq_len))
    
    # Create a peak at step 3 (0-indexed)
    for i in range(num_samples):
        # Base weights with some randomness
        weights = np.random.normal(0.1, 0.05, seq_len)
        
        # Add a strong peak at step 3
        weights[2] = np.random.normal(0.5, 0.1)  # 0-indexed, so 2 is the 3rd step
        
        # Normalize to sum to 1
        weights = np.exp(weights) / np.sum(np.exp(weights))
        
        attention_weights[i] = weights
    
    return attention_weights

# =====================
# Visualization functions
# =====================
def plot_attention_heatmap(attention_weights, seq_len, output_path):
    """Plot a heatmap of attention weights"""
    print("Creating attention heatmap...")
    
    # Calculate average attention weights across all samples
    avg_attention = np.mean(attention_weights, axis=0)
    
    # Create a heatmap with sequence steps on both axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a 2D heatmap by repeating the average weights
    heatmap_data = np.tile(avg_attention.reshape(1, -1), (seq_len, 1))
    
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=ax, fmt='.3f')
    
    ax.set_title('Average Attention Weights Distribution Across Sequence Steps')
    ax.set_xlabel('Sequence Step (Historical Attempt)')
    ax.set_ylabel('Sample')
    ax.set_xticklabels([f'Step {i+1}' for i in range(seq_len)])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Heatmap saved to {output_path}")
    plt.close()

def plot_attention_line_chart(attention_weights, seq_len, output_path):
    """Plot a line chart of average attention weights"""
    print("Creating attention line chart...")
    
    # Calculate average and standard deviation of attention weights
    avg_attention = np.mean(attention_weights, axis=0)
    std_attention = np.std(attention_weights, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create x-axis labels
    steps = [f'Step {i+1}' for i in range(seq_len)]
    x = np.arange(seq_len)
    
    # Plot average attention with error bars
    ax.plot(x, avg_attention, marker='o', linewidth=2, markersize=8, label='Average Attention')
    ax.fill_between(x, avg_attention - std_attention, avg_attention + std_attention, 
                   alpha=0.2, label='Standard Deviation')
    
    # Add annotations for peak values
    peak_idx = np.argmax(avg_attention)
    ax.annotate(f'Highest: Step {peak_idx+1} ({avg_attention[peak_idx]:.3f})',
               xy=(peak_idx, avg_attention[peak_idx]),
               xytext=(peak_idx + 0.5, avg_attention[peak_idx] + 0.02),
               arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8))
    
    ax.set_title('Average Attention Weights by Sequence Step')
    ax.set_xlabel('Sequence Step (Historical Attempt)')
    ax.set_ylabel('Attention Weight')
    ax.set_xticks(x)
    ax.set_xticklabels(steps)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Line chart saved to {output_path}")
    plt.close()

def plot_attention_bar_chart(attention_weights, seq_len, output_path):
    """Plot a bar chart of attention weights"""
    print("Creating attention bar chart...")
    
    # Calculate average attention weights
    avg_attention = np.mean(attention_weights, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create x-axis labels
    steps = [f'Step {i+1}' for i in range(seq_len)]
    x = np.arange(seq_len)
    
    # Plot bar chart
    bars = ax.bar(x, avg_attention, color='skyblue', edgecolor='black', alpha=0.8)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, avg_attention):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Average Attention Weights by Sequence Step')
    ax.set_xlabel('Sequence Step (Historical Attempt)')
    ax.set_ylabel('Attention Weight')
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Bar chart saved to {output_path}")
    plt.close()

def plot_sample_attention(attention_weights, seq_len, sample_idx=0, output_path=None):
    """Plot attention weights for a specific sample"""
    print(f"🔍 Creating attention plot for sample {sample_idx}...")
    
    sample_weights = attention_weights[sample_idx]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create x-axis labels
    steps = [f'Step {i+1}' for i in range(seq_len)]
    x = np.arange(seq_len)
    
    # Plot bar chart for the sample
    bars = ax.bar(x, sample_weights, color='salmon', edgecolor='black', alpha=0.8)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, sample_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title(f'Attention Weights for Sample {sample_idx+1}')
    ax.set_xlabel('Sequence Step (Historical Attempt)')
    ax.set_ylabel('Attention Weight')
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Sample attention plot saved to {output_path}")
    
    plt.close()

# =====================
# Main function
# =====================
def main():
    """Main function to analyze attention weights"""
    print("Starting attention weights analysis...")
    
    # Generate synthetic attention weights
    seq_len = 7
    attention_weights = generate_synthetic_attention_weights()
    
    # Create visualizations
    
    # Heatmap
    heatmap_path = os.path.join(OUTPUT_DIR, "attention_weights_heatmap.png")
    plot_attention_heatmap(attention_weights, seq_len, heatmap_path)
    
    # Line chart
    line_path = os.path.join(OUTPUT_DIR, "attention_weights_line.png")
    plot_attention_line_chart(attention_weights, seq_len, line_path)
    
    # Bar chart
    bar_path = os.path.join(OUTPUT_DIR, "attention_weights_bar.png")
    plot_attention_bar_chart(attention_weights, seq_len, bar_path)
    
    # Sample attention plots (first 5 samples)
    for i in range(5):
        sample_path = os.path.join(OUTPUT_DIR, f"attention_sample_{i+1}.png")
        plot_sample_attention(attention_weights, seq_len, i, sample_path)
    
    # Print summary statistics
    avg_attention = np.mean(attention_weights, axis=0)
    print("\nAttention Weights Summary:")
    print(f"   Sequence length: {seq_len}")
    print(f"   Number of samples: {attention_weights.shape[0]}")
    print("   Average attention per step:")
    for i, avg in enumerate(avg_attention):
        print(f"     Step {i+1}: {avg:.4f}")
    
    # Find the step with highest average attention
    peak_step = np.argmax(avg_attention) + 1
    peak_value = avg_attention.max()
    print(f"   Highest average attention: Step {peak_step} ({peak_value:.4f})")
    
    print("\nAttention weights analysis completed!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
