#!/bin/bash

# This script runs inference for the Stage 1 model.
# It uses a pre-trained model checkpoint specified in the config file.

echo "Starting Stage 1 inference..."
python3 -m stage1.inference "$@"
echo "Inference finished."
