#!/bin/bash

# Flags and arguments:
# --scene : name of the scene, all capital leters
# --ref_id : name of the map session
# --query_id : name of the query session
# --retrieval : retrieval method
# --feature : feature extraction method
# --matcher : feature matcher method
# --capture : path to capture directory
# --outputs : path to the ouput directory
# --query_filename : name of the file keyframes list, in query_name/proc/query_filename.txt
# --is_rig : to be used with rig like query sessions, i.e. hololens and spot

# Consider writing output of this script in a file if you are using full configuration (all 18 configurations). 
# Output is too long, you will not be able to see all the recall results inside a CLI! Something like this:
# ./run_scripts/run_benchmarking.sh > location.txt 2>&1

# If you are saving to a .txt file you might use our run_scripts/run_read_benchmarking_output.sh script.
# This will print out confusion matrices of benchamrking results only of recall and map/query names.

if [ -z "$CAPTURE_DIR" ]; then
  echo "[ERROR] CAPTURE_DIR env var not set. Make sure to export CAPTURE_DIR=/path/to/data/root."
  exit 1
fi

# LOCATIONS=("SUCCULENT")
LOCATIONS=("HYDRO" "SUCCULENT")
OUTPUT_DIR="benchmarking_all_ml_sp_lg"
QUERIES_FILE="keyframes_original.txt"
LOCAL_FEATURE_METHOD="superpoint"
MATCHING_METHOD="lightglue"
GLOBAL_FEATURE_METHOD="megaloc"
# DEVICES_REF=("ios")
DEVICES_REF=("ios" "hl" "spot")
# DEVICES_QUERY=("ios")
DEVICES_QUERY=("ios" "hl" "spot")

echo "You are running with parameters: "
echo "  Capture: ${CAPTURE_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Locations: ${LOCATIONS[@]}"
echo "  Queries file: ${QUERIES_FILE}"
echo "  Local feature method: ${LOCAL_FEATURE_METHOD}"
echo "  Matching method: ${MATCHING_METHOD}"
echo "  Global feature method: ${GLOBAL_FEATURE_METHOD}"
echo "  Reference devices: ${DEVICES_REF[@]}"
echo "  Query devices: ${DEVICES_QUERY[@]}"

# read -p "Do you want to continue? (y/n): " answer

# if [[ ! "$answer" =~ ^[Yy]$ ]]; then
#     echo "Execution aborted."
#     exit 1
# fi

for LOCATION in "${LOCATIONS[@]}"; do

  CAPTURE="${CAPTURE_DIR}/${LOCATION}"
  OUTPUT_DIR_LOCATION="${CAPTURE}/${OUTPUT_DIR}"

  # Do not remove or change this line if you intend to use automatic recall reading tool.
  echo "Starting benchmarking for scene: $LOCATION and queries file: $QUERIES_FILE"

  for ref in "${DEVICES_REF[@]}"; do
    for query in "${DEVICES_QUERY[@]}"; do
      echo "Running with ref_id=${ref}_map and query_id=${query}_query ..."
      
      is_rig_flag=""
      if [[ "$query" == "hl" || "$query" == "spot" ]]; then
        is_rig_flag="--is_rig"
        echo "Run is using flag --is_rig due to ${query}_query"
      fi

      python -m lamar.run \
        --scene "$SCENE" \
        --ref_id "${ref}_map" \
        --query_id "${query}_query" \
        --retrieval "$GLOBAL_FEATURE_METHOD" \
        --feature "$LOCAL_FEATURE_METHOD" \
        --matcher "$MATCHING_METHOD" \
        --capture "$CAPTURE" \
        --outputs "$OUTPUT_DIR_LOCATION" \
        --query_filename "$QUERIES_FILE" \
        $is_rig_flag

      echo "Benchmarking completed for ref_id=${ref}_map and query_id=${query}_query"
      echo ""
    done
  done

  echo -e "Benchmarking completed for scene: $LOCATION and queries file: $QUERIES_FILE" 
  echo ""
  echo ""
done
