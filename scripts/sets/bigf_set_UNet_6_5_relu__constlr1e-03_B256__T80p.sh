#!/bin/bash

#OAR -n melissa-study-UNet_6_5_relu__constlr1e-03_B256__T80p
#OAR -l /nodes=1/core=10/gpu=1,walltime=4:00:00
#OAR -p gpumodel='V100'
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh
singularity_container="/bettik/PROJECTS/pr-melissa/COMMON/containers/April23/melissa-active-sampling-with-apebench-cuda.sif"
dir_set="/home/dymchens-ext/apebench_test/experiments/set/diff_ks_cons__3w_x19_easier_1d_x5/100BUF_10WM__2TD_6CL/UNet_6_5_relu__constlr1e-03_B256__T80p"


singularity exec --nv --bind /bettik:/bettik --env APEBENCH_ROOT="$HOME/apebench_test" ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/diff_ks_cons__3w_x19_easier_1d_x5/100BUF_10WM__2TD_6CL/UNet_6_5_relu__constlr1e-03_B256__T80p/config_diff_ks_cons__3w_x19_easier_1d_x5__100BUF_10WM__2TD_6CL__UNet_6_5_relu__constlr1e-03_B256__T80p__precise_mpi.json
singularity exec --nv --bind /bettik:/bettik --env APEBENCH_ROOT="$HOME/apebench_test" ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/diff_ks_cons__3w_x19_easier_1d_x5/100BUF_10WM__2TD_6CL/UNet_6_5_relu__constlr1e-03_B256__T80p/config_diff_ks_cons__3w_x19_easier_1d_x5__100BUF_10WM__2TD_6CL__UNet_6_5_relu__constlr1e-03_B256__T80p__no_resampling_mpi.json
singularity exec --nv --bind /bettik:/bettik --env APEBENCH_ROOT="$HOME/apebench_test" ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/diff_ks_cons__3w_x19_easier_1d_x5/100BUF_10WM__2TD_6CL/UNet_6_5_relu__constlr1e-03_B256__T80p/config_diff_ks_cons__3w_x19_easier_1d_x5__100BUF_10WM__2TD_6CL__UNet_6_5_relu__constlr1e-03_B256__T80p__uniform_mpi.json
singularity exec --nv --bind /bettik:/bettik --env APEBENCH_ROOT="$HOME/apebench_test" ${singularity_container} melissa-launcher --config_name /home/dymchens-ext/apebench_test/experiments/set/diff_ks_cons__3w_x19_easier_1d_x5/100BUF_10WM__2TD_6CL/UNet_6_5_relu__constlr1e-03_B256__T80p/config_diff_ks_cons__3w_x19_easier_1d_x5__100BUF_10WM__2TD_6CL__UNet_6_5_relu__constlr1e-03_B256__T80p__broad_mpi.json
