# -*- coding: utf-8 -*-
"""
Update prediction analysis plot with new tolerance band
"""

import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
from vis_utils import plot_prediction_distribution

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, '04_dataset_construction', 'output')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(MODEL_DIR, 'output')

# Load data
print('📥 Loading data...')
X_seq = np.load(os.path.join(DATA_DIR, 'X_seq.npy'))
X_feat = np.load(os.path.join(DATA_DIR, 'X_feat.npy'))
y = np.load(os.path.join(DATA_DIR, 'y.npy'))
y = y.astype('float32')

# Reshape sequences
seq_len = 7
seq_dim = 5
X_seq = X_seq.reshape(-1, seq_len, seq_dim)

# Use smaller subset for faster prediction (10,000 samples)
subset_size = 10000
if len(X_seq) > subset_size:
    print(f'📋 Using subset of {subset_size} samples for faster prediction...')
    X_seq = X_seq[:subset_size]
    X_feat = X_feat[:subset_size]
    y = y[:subset_size]

# Load model
print('🤖 Loading model...')
model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, 'lstm_main_model.keras'))

# Make predictions
print('🔮 Making predictions...')
y_pred_reg, _ = model.predict([X_seq, X_feat], batch_size=256, verbose=1)
y_pred_reg = y_pred_reg.flatten()

# Generate updated prediction analysis plot
print('📊 Generating updated prediction analysis plot...')
plot_prediction_distribution(y, y_pred_reg, 'lstm', OUTPUT_DIR)
print('🎉 Done!')