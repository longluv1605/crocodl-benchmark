#!/bin/bash

# Simple evaluation script for cross-device pose estimation
# Computes success rate matrices for LOCATION and OVERALL performance
# Based on benchmark script structure for consistency

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -z "$CAPTURE_DIR" ]; then
  echo "[ERROR] CAPTURE_DIR env var not set. Make sure to export CAPTURE_DIR=/path/to/data/root."
  exit 1
fi

# Configuration matching benchmark script structure
BENCHMARKING_DIR="long/benchmarking_results"
OUTPUT_DIR="${CAPTURE_DIR}/evaluation_results"
LOCAL_FEATURE_METHOD="superpoint"
MATCHING_METHOD="lightglue"
GLOBAL_FEATURE_METHOD="megaloc"
SCENES=("hydro" "succu")
DEVICES_MAP=("ios" "hl" "spot")
DEVICES_QUERY=("ios" "hl" "spot")
POSITION_THRESHOLD=2
ROTATION_THRESHOLD=20
PYTHON_SCRIPT="${SCRIPT_DIR}/evaluate.py"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Cross-device pose estimation evaluation with success rate matrix analysis.

OPTIONS:
    -o, --output_dir DIR        Output directory for results (default: ${OUTPUT_DIR})
    -p, --position_threshold T  Position error threshold in meters (default: ${POSITION_THRESHOLD})
    -r, --rotation_threshold T  Rotation error threshold in degrees (default: ${ROTATION_THRESHOLD})
    -h, --help                  Show this help message

EXAMPLES:
    $0                                          # Use default settings
    $0 --output_dir ./results                   # Specify output directory
    $0 --position_threshold 2.0 --rotation_threshold 10.0  # Custom thresholds

This script evaluates cross-device pose estimation performance by computing success rate matrices.
The success rate is calculated as: (successful poses / total ground truth poses) * 100%

A pose is considered successful if both position error < position_threshold AND rotation error < rotation_threshold.

Output includes:
- Location-wise success rate matrices (HYDRO, SUCCULENT)
- Overall success rate matrix (aggregated across all locations)
- JSON files with detailed results
- CSV files with formatted matrices

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--position_threshold)
            POSITION_THRESHOLD="$2"
            shift 2
            ;;
        -r|--rotation_threshold)
            ROTATION_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -d "$CAPTURE_DIR" ]]; then
    echo "Error: capture directory not found: $CAPTURE_DIR"
    echo "Please set CAPTURE_DIR environment variable or check the path"
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python evaluation script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check for required Python packages
echo "Checking Python dependencies..."
python3 -c "import numpy, pandas; print('Dependencies OK')" 2>/dev/null || {
    echo "Error: Required Python packages not found."
    echo "Please install: pip install numpy pandas"
    exit 1
}

echo "You are running with parameters: "
echo "  Capture: ${CAPTURE_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Benchmarking dir: ${BENCHMARKING_DIR}"
echo "  Local feature method: ${LOCAL_FEATURE_METHOD}"
echo "  Matching method: ${MATCHING_METHOD}"
echo "  Global feature method: ${GLOBAL_FEATURE_METHOD}"
echo "  Scenes: ${SCENES[@]}"
echo "  Devices map: ${DEVICES_MAP[@]}"
echo "  Devices query: ${DEVICES_QUERY[@]}"
echo "  Position threshold: ${POSITION_THRESHOLD} meters"
echo "  Rotation threshold: ${ROTATION_THRESHOLD} degrees"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Running evaluation..."
python3 -m evaluate.evaluate \
    --capture_dir "$CAPTURE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --benchmarking_dir "$BENCHMARKING_DIR" \
    --local_feature_method "$LOCAL_FEATURE_METHOD" \
    --matching_method "$MATCHING_METHOD" \
    --global_feature_method "$GLOBAL_FEATURE_METHOD" \
    --scenes "${SCENES[@]}" \
    --devices_map "${DEVICES_MAP[@]}" \
    --devices_query "${DEVICES_QUERY[@]}" \
    --position_threshold "$POSITION_THRESHOLD" \
    --rotation_threshold "$ROTATION_THRESHOLD"

echo ""
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"