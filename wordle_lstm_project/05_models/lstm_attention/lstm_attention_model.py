# -*- coding: utf-8 -*-
"""
LSTM + Attention model definition
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    """Custom attention layer for sequence modeling"""
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
        self.units = units
    
    def get_config(self):
        """Return layer configuration for serialization"""
        config = super(AttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config
    
    def call(self, query, values):
        # query: [batch_size, hidden_size] from LSTM output
        # values: [batch_size, seq_len, hidden_size] from LSTM return_sequences=True
        
        # Expand query dimensions for broadcasting
        # [batch_size, 1, hidden_size]
        query_expanded = tf.expand_dims(query, 1)
        
        # Calculate attention scores
        # [batch_size, seq_len, 1]
        score = self.V(tf.nn.tanh(
            self.W1(query_expanded) + self.W2(values)
        ))
        
        # Apply softmax to get attention weights
        # [batch_size, seq_len, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Multiply weights with values to get context vector
        # [batch_size, hidden_size]
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

def build_lstm_attention_model(
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
    # LSTM layers with dropout for attention
    # =====================
    # First LSTM layer
    lstm_layer1 = layers.LSTM(
        lstm_units,
        return_sequences=True,
        return_state=True,
        dropout=0.2,
        recurrent_dropout=0.1
    )
    
    lstm_outputs1, state_h1, state_c1 = lstm_layer1(x)
    
    # Second LSTM layer
    lstm_layer2 = layers.LSTM(
        lstm_units,
        return_sequences=True,
        return_state=True,
        dropout=0.2,
        recurrent_dropout=0.1
    )
    
    lstm_outputs2, state_h2, state_c2 = lstm_layer2(lstm_outputs1)

    # =====================
    # Attention layer
    # =====================
    attention_layer = AttentionLayer(units=lstm_units)
    context_vector, attention_weights = attention_layer(state_h2, lstm_outputs2)

    # =====================
    # Player feature input
    # =====================
    feat_input = layers.Input(
        shape=(num_player_features,),
        name="player_feat_input"
    )

    # =====================
    # Fusion with enhanced architecture
    # =====================
    combined = layers.Concatenate()([context_vector, feat_input])
    
    # First dense layer with layer normalization
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.LayerNormalization()(combined)
    combined = layers.Dropout(0.2)(combined)
    
    # Second dense layer
    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.LayerNormalization()(combined)
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
