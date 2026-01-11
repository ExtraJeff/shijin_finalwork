# -*- coding: utf-8 -*-
"""
LSTM model definition
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def build_lstm_model(
    seq_len=7,
    seq_dim=5,
    embed_dim=8,
    lstm_units=128,
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
    # LSTM layers with dropout
    # =====================
    x = layers.LSTM(
        lstm_units,
        return_sequences=True
    )(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.LSTM(
        lstm_units,
        return_state=False
    )(x)
    x = layers.Dropout(0.3)(x)

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

    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)

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

    # Compile with AdamW optimizer and smaller initial learning rate
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.0005,
        weight_decay=0.001
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            "reg_output": "mse",
            "class_output": "binary_crossentropy"
        },
        loss_weights={
            "reg_output": 0.8,
            "class_output": 0.2
        },
        metrics={
            "reg_output": ["mae"],
            "class_output": ["accuracy"]
        }
    )

    return model
