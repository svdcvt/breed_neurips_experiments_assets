#!/bin/bash


python3 plot_model_predictions.py \
    --study-paths  ../experiments/across_pdes/diff_mode/ks_cons/unet_breed_1 \
        ../experiments/across_pdes/diff_mode/ks_cons/unet_breed_2 \
        ../experiments/across_pdes/diff_mode/ks_cons/unet_breed_2500 \
        ../experiments/across_pdes/diff_mode/ks_cons/unet_breed_2500_const \
        ../experiments/across_pdes/diff_mode/ks_cons/unet_uniform \
        ../experiments/across_pdes/diff_mode/ks_cons/unet_uniform_1500 \
        ../experiments/across_pdes/diff_mode/ks_cons/unet_uniform_2500 \
        ../experiments/across_pdes/diff_mode/ks_cons/unet_uniform_2500_const \
    --output-dir ./validation_results_big/
