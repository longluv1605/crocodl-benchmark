#!/bin/bash

# Simple evaluation script for cross-device pose estimation
# Computes success rate matrices for LOCATION and OVERALL performance
# Based on benchmark script structure for consistency

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -z "$CAPTURE_DIR" ]; then
  CAPTURE_DIR="${PROJECT_ROOT}/capture_cv"
  echo "[INFO] Using default CAPTURE_DIR: $CAPTURE_DIR"
fi

# Configuration matching benchmark script
LOCATIONS=("HYDRO" "SUCCULENT")
BENCHMARK_DIR="benchmarking_all_ml_sp_lg"  # Fixed benchmark directory
OUTPUT_DIR="${PROJECT_ROOT}/evaluation_results"
DEVICES_REF=("ios" "hl" "spot")
DEVICES_QUERY=("ios" "hl" "spot")
POSITION_THRESHOLD=1.0
ROTATION_THRESHOLD=5.0
PYTHON_SCRIPT="${SCRIPT_DIR}/evaluate.py"

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Cross-device pose estimation evaluation with success rate matrix analysis.

OPTIONS:
    -d, --capture_dir DIR       Path to capture directory (default: ${CAPTURE_DIR})
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
        -d|--capture_dir)
            CAPTURE_DIR="$2"
            shift 2
            ;;
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

# Display configuration
echo "=============================================="
echo "Cross-device Pose Estimation Evaluation"
echo "=============================================="
echo "Capture directory: $CAPTURE_DIR"
echo "Benchmark directory: $BENCHMARK_DIR"
echo "Locations: ${LOCATIONS[@]}"
echo "Reference devices: ${DEVICES_REF[@]}"
echo "Query devices: ${DEVICES_QUERY[@]}"
echo "Output directory: $OUTPUT_DIR"
echo "Position threshold: $POSITION_THRESHOLD meters"
echo "Rotation threshold: $ROTATION_THRESHOLD degrees"
echo "=============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Running evaluation..."
python3 "$PYTHON_SCRIPT" \
    --capture_dir "$CAPTURE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --position_threshold "$POSITION_THRESHOLD" \
    --rotation_threshold "$ROTATION_THRESHOLD"

echo ""
echo "=============================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/$BENCHMARK_DIR"
echo "=============================================="