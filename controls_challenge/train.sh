#!/bin/bash

# Default values
DEFAULT_LOGDIR_PATH="./logdir/comma-controls-1"
DEFAULT_CONFIG_FLAGS="comma"

# Assign arguments to variables with defaults
resume=${1:-false}
logdir_path=${2:-$DEFAULT_LOGDIR_PATH}
shift
config_flags=${@:-$DEFAULT_CONFIG_FLAGS}


if [ "$resume" = false ]; then
    # Delete "logdir" directory recursively
    echo "Deleting logdir directory..."
    rm -rf $logdir_path
    echo "logdir directory deleted."
fi

# Run training by calling dreamerv3-torch/dreamer.py with the provided configs and logdir
echo "Starting training with configs: $config_flags and logdir: $logdir_path"
python dreamerv3-torch/dreamer.py --configs $config_flags --logdir $logdir_path
