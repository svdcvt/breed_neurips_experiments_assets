#!/bin/bash

#OAR -n melissa-study-validation-3
#OAR -l /nodes=1/core=20,walltime=2:50:00
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh
singularity_container="/bettik/PROJECTS/pr-melissa/COMMON/containers/April23/melissa-active-sampling-with-apebench-cuda.sif"


singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=/home/dymchens-ext/apebench_test ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/config_offline_diff_ks__2w_x07_harder_max1_1d_x5_mpi.json
sleep 10
singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=/home/dymchens-ext/apebench_test ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/config_offline_diff_ks__3w_default_1d_x5_mpi.json
sleep 10

data_paths_pattern="/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/diff_*/trajectories/"
data_paths=$(ls -d ${data_paths_pattern})
for data_path in ${data_paths}
do
    echo "Processing data path: ${data_path}"
    # Merge samples for each data path
    python3 $HOME/apebench_test/scripts/utils/merge_samples.py ${data_path} --quiet
done
