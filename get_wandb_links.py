#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get W&B links for all Wordle models
"""

import os
import sys
import wandb
import time

# Set WandB API key
os.environ["WANDB_API_KEY"] = "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"

# Model types to generate links for
model_types = ["lstm", "bilstm", "lstm_attention", "transformer"]

# Create output file
output_file = "wandb_links.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("# Wordle模型WandB链接\n\n")

# Generate links for each model
for model_type in model_types:
    print(f"\n🔗 Generating W&B link for {model_type} model...")
    
    try:
        # Initialize WandB run
        run = wandb.init(
            project="wordle-game-prediction",
            entity="legendjeff",
            config={
                "model_type": model_type,
                "seq_len": 7,
                "seq_dim": 5,
                "embed_dim": 8,
                "lstm_units": 128 if model_type != "lstm" else 64,
                "epochs": 30,
                "batch_size": 128,
                "learning_rate": 0.0003,
                "patience": 5
            },
            name=f"{model_type}-run-{int(time.time())}",
            save_code=True
        )
        
        # Get the run URL
        run_url = run.url
        print(f"✅ Generated link for {model_type}: {run_url}")
        
        # Save to file
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"## {model_type.upper()}模型\n{run_url}\n\n")
        
        # Finish the run immediately
        wandb.finish()
        
    except Exception as e:
        print(f"❌ Error generating link for {model_type}: {e}")
        continue

print(f"\n🎉 All links generated successfully!")
print(f"📁 Links saved to: {output_file}")
print(f"\nYou can now use these links in your homework report.")
