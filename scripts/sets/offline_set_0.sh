#!/bin/bash

#OAR -n melissa-study-validation-3
#OAR -l /nodes=1/core=20,walltime=7:00:00
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh
singularity_container="/bettik/PROJECTS/pr-melissa/COMMON/containers/April23/melissa-active-sampling-with-apebench-cuda.sif"


singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=/home/dymchens-ext/apebench_test ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/config_offline_diff_ks_cons__3w_default_1d_x5_mpi.json
sleep 10
singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=/home/dymchens-ext/apebench_test ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/config_offline_diff_ks__3w_x07_harder_max1_1d_x5_mpi.json
sleep 10
singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=/home/dymchens-ext/apebench_test ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/config_offline_diff_ks__2w_x07_harder_1d_x5_mpi.json
sleep 10
singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=/home/dymchens-ext/apebench_test ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/config_offline_diff_ks__2w_default_max1_1d_x5_mpi.json
sleep 10
singularity exec  --bind /bettik:/bettik --env APEBENCH_ROOT=/home/dymchens-ext/apebench_test ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/config_offline_diff_kdv__2w_x10_easier_1d_x5_mpi.json
sleep 10
