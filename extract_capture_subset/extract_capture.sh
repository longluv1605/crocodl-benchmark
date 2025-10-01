#!/usr/bin/env bash
set -e 

CONFIG_FILE="./extract_capture_subset/config.yaml"

CAPTURE_SUBSET_DIR=$(grep "capture_subset_dir:" "$CONFIG_FILE" | awk '{print $2}')

if [ -d "$CAPTURE_SUBSET_DIR" ]; then
    echo "Removing existing directory: $CAPTURE_SUBSET_DIR"
    rm -rf "$CAPTURE_SUBSET_DIR"
fi

echo "Running extract_capture..."
python3 -m extract_capture_subset.extract_capture --config "$CONFIG_FILE"
