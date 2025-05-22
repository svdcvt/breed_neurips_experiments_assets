#!/bin/bash


all_pdes=($(ls  $REPO_ROOT/experiments/set/ | grep diff))
echo "Found PDEs: ${all_pdes[@]}"

all_methods=(broad_0 soft_0 mixed_0 precise_0 uniform_0)

for pde in "${all_pdes[@]}"; do
    for method in "${all_methods[@]}"; do
        echo "Processing $pde with method $method"
        # Extract the last part of the path
        # Run the Python script with the extracted parameters
        python3 $REPO_ROOT/scripts/utils/plot_client_parameters.py --input-dir $REPO_ROOT/experiments/set/$pde/100BUF_10WM__2TD_8CL/UNet_6_5_relu__decaylr1e-3_1e-4_5000__B256__T75p/$method/client_scripts/ --validation-path $DATASET_ROOT/$pde/trajectories/input_parameters.npy
    done
done