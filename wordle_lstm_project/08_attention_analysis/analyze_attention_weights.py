# -*- coding: utf-8 -*-
"""
Attention Weights Analysis for LSTM-Attention Model
This script extracts and visualizes attention weights from the trained LSTM-Attention model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "05_models", "lstm_attention"))

# Import the custom AttentionLayer
from lstm_attention_model import AttentionLayer

# =====================
# Path configuration
# =====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model and data paths
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "05_models", "lstm_attention", "output", "lstm_attention_model.keras"
)
DATA_DIR = os.path.join(
    PROJECT_ROOT, "04_dataset_construction", "output"
)

# =====================
# Load data
# =====================
def load_data():
    """Load and prepare the dataset"""
    print("Loading dataset...")
    
    X_seq = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
    X_feat = np.load(os.path.join(DATA_DIR, "X_feat.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    
    y = y.astype("float32")
    
    # Reshape sequences to (seq_len, seq_dim)
    seq_len = 7
    seq_dim = 5
    X_seq = X_seq.reshape(-1, seq_len, seq_dim)
    
    # Split into train/val/test as done in training
    X_seq_train, X_seq_temp, X_feat_train, X_feat_temp, y_train, y_temp = train_test_split(
        X_seq, X_feat, y, test_size=0.3, random_state=42
    )
    
    X_seq_val, X_seq_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
        X_seq_temp, X_feat_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"📊 Dataset split: Train={X_seq_train.shape[0]}, Val={X_seq_val.shape[0]}, Test={X_seq_test.shape[0]}")
    
    return {
        'X_seq_test': X_seq_test,
        'X_feat_test': X_feat_test,
        'y_test': y_test,
        'seq_len': seq_len,
        'seq_dim': seq_dim
    }

# =====================
# Load model and extract attention weights
# =====================
def get_attention_model():
    """Load the trained model and create a new model that outputs attention weights"""
    print("Loading model...")
    
    # Load the trained model with the custom AttentionLayer
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    
    # Print model summary to find layer names
    print("\nOriginal model layers:")
    for i, layer in enumerate(model.layers):
        print(f"  {i}: {layer.name} - {layer.__class__.__name__}")
    
    # Create a new model that outputs attention weights
    # Inputs are the same as original model
    inputs = model.inputs
    
    # Build the model to ensure all layers are initialized
    model.build(inputs_shape=[(None, 7, 5), (None, 5)])
    
    # Get the attention layer
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, AttentionLayer):
            attention_layer = layer
            break
    
    if attention_layer is None:
        # Try to find the attention layer by name
        for layer in model.layers:
            if layer.name == 'attention_layer':
                attention_layer = layer
                break
    
    if attention_layer is None:
        raise ValueError("❌ Could not find attention layer in the model")
    
    # Get the LSTM layers that feed into the attention layer
    lstm_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM) and layer.return_sequences:
            lstm_layer = layer
    
    if lstm_layer is None:
        raise ValueError("❌ Could not find LSTM layer with return_sequences=True")
    
    # Create a new model that outputs attention weights
    # Get all layer outputs to find the correct connections
    all_layer_outputs = {layer.name: layer.output for layer in model.layers}
    
    # Find the LSTM output that feeds into the attention layer
    # For the attention layer, we need the sequence output and the final state
    lstm_outputs = None
    lstm_state = None
    
    # Check the model's functional API structure
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.LSTM) and layer.return_sequences and layer.return_state:
            # This layer returns multiple outputs: [output_sequence, state_h, state_c]
            lstm_outputs = layer.output[0]
            lstm_state = layer.output[1]  # Use the hidden state (state_h)
            break
    
    if lstm_outputs is None or lstm_state is None:
        raise ValueError("❌ Could not find LSTM outputs for attention layer")
    
    # Create a new model that computes attention weights
    context_vector, attention_weights = attention_layer(lstm_state, lstm_outputs)
    
    # Create the attention model
    attention_model = tf.keras.models.Model(
        inputs=inputs,
        outputs=[model.output, attention_weights]
    )
    
    return attention_model

# =====================
# Generate attention weights
# =====================
def generate_attention_weights(attention_model, data):
    """Generate attention weights for the test set"""
    print("Generating attention weights...")
    
    # Use a subset of test data for visualization
    sample_size = 1000  # Use first 1000 samples for analysis
    X_seq_sample = data['X_seq_test'][:sample_size]
    X_feat_sample = data['X_feat_test'][:sample_size]
    
    # Get predictions and attention weights
    _, attention_weights = attention_model.predict([X_seq_sample, X_feat_sample], batch_size=128, verbose=1)
    
    print(f"📏 Attention weights shape: {attention_weights.shape}")
    
    # Squeeze the last dimension (shape becomes [batch_size, seq_len])
    attention_weights = np.squeeze(attention_weights, axis=-1)
    
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
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
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
    
    # Load data
    data = load_data()
    
    try:
        # Get attention model
        attention_model = get_attention_model()
        
        # Generate attention weights
        attention_weights = generate_attention_weights(attention_model, data)
        
        # Create visualizations
        seq_len = data['seq_len']
        
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
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
