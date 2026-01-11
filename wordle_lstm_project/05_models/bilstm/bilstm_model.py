# -*- coding: utf-8 -*-
"""
BiLSTM model definition
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_bilstm_model(
    seq_len=7,
    seq_dim=5,
    embed_dim=8,
    lstm_units=256,
    num_player_features=5
):
    # =====================
    # Sequence input
    # =====================
    seq_input = layers.Input(shape=(seq_len, seq_dim), name="seq_input")

    # Embedding for feedback values (0,1,2)
    x = layers.Embedding(
        input_dim=3,
        output_dim=embed_dim
    )(seq_input)

    x = layers.Reshape((seq_len, seq_dim * embed_dim))(x)

    # =====================
    # BiLSTM layers with dropout and batch normalization
    # =====================
    # First Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True
        )
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Second Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True
        )
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Third Bidirectional LSTM layer
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_state=False
        )
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # =====================
    # Player feature input
    # =====================
    feat_input = layers.Input(
        shape=(num_player_features,),
        name="player_feat_input"
    )

    # =====================
    # Fusion
    # =====================
    combined = layers.Concatenate()([x, feat_input])

    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.2)(combined)
    
    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(0.2)(combined)

    # =====================
    # Multi-task outputs
    # =====================
    # Regression branch: predict trial count
    reg_output = layers.Dense(1, activation="linear", name="reg_output")(combined)
    
    # Classification branch: predict success (1 if trial <= 6, 0 otherwise)
    class_output = layers.Dense(1, activation="sigmoid", name="class_output")(combined)

    model = Model(
        inputs=[seq_input, feat_input],
        outputs=[reg_output, class_output]
    )

    # Compile with multi-task losses
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0003,
        weight_decay=0.0005
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            "reg_output": "mse",
            "class_output": "binary_crossentropy"
        },
        loss_weights={
            "reg_output": 0.7,
            "class_output": 0.3
        },
        metrics={
            "reg_output": ["mae"],
            "class_output": ["accuracy"]
        }
    )

    return model
