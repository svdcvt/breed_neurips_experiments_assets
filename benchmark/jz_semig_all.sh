#!/bin/sh
#SBATCH --job-name=apebench-test
#SBATCH --output=std/%j.out
#SBATCH --error=std/%j.err
#SBATCH --time=02:00:00
#SBATCH --account=igf@cpu
#SBATCH --qos=qos_cpu-dev
##SBATCH --qos=qos_cpu-t3
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20 --cpus-per-task=2 --hint=nomultithread

module load cmake zeromq openmpi/4.1.5 python/3.10.4 cudnn/9.2.0.82-cuda

source $WORK/apebench/MELISSA/melissa_set_env.sh

export APEBENCH_ROOT="$WORK/apebench"
export TORCH_PATH=$APEBENCH_ROOT/MELISSA/install/torch
export PYTHONPATH=$PYTHONPATH:$APEBENCH_ROOT/test/common:$APEBENCH_ROOT/test:$TORCH_PATH


# Check if the folder path is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Recursively find JSON files, excluding those in paths containing STUDY_OUT_*
find "$1" -type f -name "*.json" ! -path "*/STUDY_OUT_*/*" | while read -r json_file; do
    echo "Processing file: $json_file"
    melissa-launcher --config_name "$json_file"
    # Wait for the current execution to finish before moving to the next
    if [ $? -ne 0 ]; then
        echo "Error while processing $json_file. Exiting."
        exit 1
    fi
done
