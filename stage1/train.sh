#!/bin/bash

# This script runs the training for the Stage 1 model.
# Ensure you have installed the required dependencies from requirements.txt

export HF_HOME="./cache"

export TORCH_HOME="./torch_cache"


echo "Starting Stage 1 training..."
python3 -m stage1.train "$@"
echo "Training finished."
