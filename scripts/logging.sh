#!/bin/bash

# Set the log file paths
GPU_LOG_FILE="../experiments/across_pdes/norm_mode/STUDY_OUT_burgers_1d_breed_Res_13_2_highres_fast_noval/gpu_memlog.log"
OTHER_LOG_FILE="../experiments/across_pdes/norm_mode/STUDY_OUT_burgers_1d_breed_Res_13_2_highres_fast_noval/stdout/openmpi.0.err"

# Set the interval in seconds (0.1 seconds)
INTERVAL=1.0

while true; do
    # Get GPU memory usage
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
    
    # Get the last line from the other log file
    LAST_LINE=$(tail -n 1 "$OTHER_LOG_FILE")
    
    # Log both to the combined log file
    echo "$(date +%H:%M:%S) - GPU Memory Used: $GPU_MEM" >> "$GPU_LOG_FILE"
    echo "Last Line: $LAST_LINE" >> "$GPU_LOG_FILE"
    
    # Sleep for the specified interval
    sleep "$INTERVAL"
done