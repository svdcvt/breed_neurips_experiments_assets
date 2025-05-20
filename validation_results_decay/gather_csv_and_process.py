import os
import pandas as pd


def gather_csv_files(directory):
    csv_name = 'validation_results.csv'
    pde_dirs = [d for d in os.listdir(directory) if d.startswith('diff') and os.path.isdir(os.path.join(directory, d))]
    for pde_dir in pde_dirs:
        pde_path = os.path.join(directory, pde_dir)
        model_dirs = sorted([d for d in os.listdir(pde_path) if d.endswith('000') and os.path.isdir(os.path.join(pde_path, d))])
        for model_dir in model_dirs:
            model_path = os.path.join(pde_path, model_dir)
            # Check if the directory contains a CSV file
            if csv_name in os.listdir(model_path):
                csv_path = os.path.join(model_path, csv_name)
                csv = pd.read_csv(csv_path, header=0, index_col=False)
                csv['pde'] = pde_dir
                yield csv

all_csvs = pd.concat([csv for csv in gather_csv_files('../validation_results_decay/')], ignore_index=True)
all_csvs.to_csv('combined_validation_results.csv', index=False)
