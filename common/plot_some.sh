#!/bin/bash

chosenset=("diff_kdv__2w_x04_harder_max1_1d_x5")

for name in "${chosenset[@]}"; do
    python3 plot_model_predictions.py --study-paths ../experiments/set/${name}/100BUF_10WM__2TD_8CL/UNet_6_5_relu__constlr1e-03_B256__T75p/*/ --output-dir ../validation_results/validation_results_decay/${name}/best/ --ecdf-plot
done
