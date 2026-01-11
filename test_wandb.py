#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test WandB connection with user's API key
"""

import wandb
import os
import time

# Set WandB API key from user input
os.environ["WANDB_API_KEY"] = "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"

# Test WandB initialization
try:
    print("🔍 Testing WandB connection...")
    
    # Initialize WandB project
    run = wandb.init(
        project="wordle-game-prediction",
        entity="legendjeff",
        config={
            "test_param": "test_value",
            "model_type": "test"
        },
        name="test-run-" + str(int(time.time()))
    )
    
    # Log some test metrics
    print("📊 Logging test metrics...")
    for i in range(5):
        wandb.log({
            "test_loss": 0.5 - i*0.1,
            "test_accuracy": 0.5 + i*0.1,
            "step": i
        })
        time.sleep(0.5)
    
    # Finish the run
    print("✅ Test completed successfully!")
    print(f"📋 WandB Run URL: {run.url}")
    
    wandb.finish()
    
    print("\n🎉 WandB connection test passed!")
    print("\nYou can now view the test run at:")
    print(run.url)
    
except Exception as e:
    print(f"❌ Error during WandB test: {e}")
    print("\nPlease check your API key and username, and ensure you have internet connection.")
