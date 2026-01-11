# -*- coding: utf-8 -*-
"""
Transformer model definition
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    """Positional encoding layer for Transformer"""
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.positional_encoding = self.positional_encoding_matrix()
    
    def positional_encoding_matrix(self):
        pos = tf.range(self.sequence_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
        angle_rads = pos * angle_rates
        
        # Create a mask for even and odd indices
        indices = tf.range(self.d_model, dtype=tf.float32)
        even_mask = tf.equal(tf.math.mod(indices, 2), 0)
        odd_mask = tf.logical_not(even_mask)
        
        # Apply sin and cos using boolean masks
        sin_part = tf.where(even_mask, tf.sin(angle_rads), angle_rads)
        pos_encoding = tf.where(odd_mask, tf.cos(angle_rads), sin_part)
        
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        return inputs + self.positional_encoding
    
    def get_config(self):
        """Return layer configuration for serialization"""
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "d_model": self.d_model
        })
        return config

@tf.keras.utils.register_keras_serializable()
class MultiHeadAttentionLayer(layers.Layer):
    """Multi-head attention layer"""
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # Transpose to get (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # Calculate attention scores
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Multiply weights with values
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        
        return output, attention_weights
    
    def get_config(self):
        """Return layer configuration for serialization"""
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerEncoderLayer(layers.Layer):
    """Transformer encoder layer"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Store parameters as instance attributes for serialization
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.mha = MultiHeadAttentionLayer(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training, mask=None):
        attn_output, _ = self.mha(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        """Return layer configuration for serialization"""
        config = super(TransformerEncoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config

def build_transformer_model(
    seq_len=7,
    seq_dim=5,
    embed_dim=8,
    d_model=128,
    num_heads=8,
    dff=256,
    num_layers=3,
    num_player_features=5,
    dropout_rate=0.2
):
    # =====================
    # Sequence input
    # =====================
    seq_input = layers.Input(shape=(seq_len, seq_dim), name="seq_input")

    # =====================
    # Embedding layer
    # =====================
    # First, reshape the sequence to flatten the 5 letters per guess
    x = layers.Reshape((seq_len, seq_dim))(seq_input)
    
    # Linear projection to d_model dimensions
    x = layers.Dense(d_model)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # =====================
    # Positional encoding
    # =====================
    x = PositionalEncoding(seq_len, d_model)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # =====================
    # Transformer encoder stack
    # =====================
    for i in range(num_layers):
        x = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate
        )(x, training=True)
    
    # =====================
    # Global average pooling
    # =====================
    x = layers.GlobalAveragePooling1D()(x)
    
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
    
    # Enhanced dense layers with BatchNormalization
    combined = layers.Dense(128, activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    
    combined = layers.Dense(64, activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    
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
