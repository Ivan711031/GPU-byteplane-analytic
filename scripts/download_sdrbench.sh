#!/bin/bash
set -euo pipefail
# SDRBench Data Download Script
# Target: Hurricane Isabel and CESM-ATM
DATA_DIR="datasets/scientific"
mkdir -p "$DATA_DIR"
echo "--- Downloading Hurricane Isabel (100x500x500) ---"
HURRICANE_URL="https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz"
if [ ! -f "$DATA_DIR/hurricane.tar.gz" ]; then
    wget -O "$DATA_DIR/hurricane.tar.gz.tmp" "$HURRICANE_URL"
    mv "$DATA_DIR/hurricane.tar.gz.tmp" "$DATA_DIR/hurricane.tar.gz"
fi
# Ensure extraction is successful
if [ ! -d "$DATA_DIR/SDRBENCH-Hurricane-ISABEL-100x500x500" ]; then
    tar -xzvf "$DATA_DIR/hurricane.tar.gz" -C "$DATA_DIR"
else
    echo "Hurricane Isabel unpacked directory already exists."
fi
echo "--- Downloading CESM-ATM (1800x3600) ---"
CESM_URL="https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/ATM/SDRBENCH-ATM-26x1800x3600.tar.gz"
if [ ! -f "$DATA_DIR/cesm.tar.gz" ]; then
    wget -O "$DATA_DIR/cesm.tar.gz.tmp" "$CESM_URL"
    mv "$DATA_DIR/cesm.tar.gz.tmp" "$DATA_DIR/cesm.tar.gz"
fi
# Ensure extraction is successful
if [ ! -d "$DATA_DIR/SDRBENCH-ATM-26x1800x3600" ]; then
    tar -xzvf "$DATA_DIR/cesm.tar.gz" -C "$DATA_DIR"
else
    echo "CESM-ATM unpacked directory already exists."
fi
echo "Download and extraction complete. Files are in $DATA_DIR"
