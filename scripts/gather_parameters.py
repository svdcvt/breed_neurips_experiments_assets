#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_parameters(file_path):
    """Extract numeric parameters from the --ic-config argument in a client script."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Match the entire --ic-config parameter
    match = re.search(r'--ic-config="([^"]+)"', content)
    if not match:
        return None
    
    config_string = match.group(1)
    # Split by semicolon and extract parameters
    parts = config_string.split(';')
    
    # Get all numeric values (ignoring first part which is 'sine_sup' and last parts which are booleans)
    numeric_parts = []
    for part in parts[1:-2]:  # Skip first and last two parts
        try:
            numeric_parts.append(float(part))
        except ValueError:
            continue
    
    return numeric_parts

def gather_parameters(directory):
    """Gather parameters from all client script files in the directory."""
    parameters = []
    ids = []
    
    # Get all client.*.sh files and sort them
    files = sorted(Path(directory).glob("client.*.sh"), 
                  key=lambda x: int(x.stem.split('.')[1]))
    
    for file_path in files:
        client_id = int(file_path.stem.split('.')[1])
        params = extract_parameters(file_path)
        if params:
            ids.append(client_id)
            parameters.append(params)
    
    return np.array(ids), np.array(parameters)

def plot_parameters(ids, parameters, output_dir):
    """Create a scatter plot matrix of all parameters."""
    
    # Load additional data points
    additional_params_path = "/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/burgers_1d/high_res_faster_default/trajectories/all_parameters.npy"
    additional_parameters = np.load(additional_params_path)
    if additional_parameters.shape[1] < parameters.shape[1]:
        additional_parameters = np.concatenate((additional_parameters, np.full((additional_parameters.shape[0], parameters.shape[1] - additional_parameters.shape[1]), np.nan)), axis=1)
    elif additional_parameters.shape[1] > parameters.shape[1]:
        parameters = np.concatenate((parameters, np.full((parameters.shape[0], additional_parameters.shape[1] - parameters.shape[1]), np.nan)), axis=1)
    
    # Get number of parameters per data point
    n_params = parameters.shape[1]
    
    # Create a figure for the scatter plot matrix
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Parameter labels
    params = ['Amplitude', 'Phase']
    param_labels = [f'{params[i%2]} {i//2}' for i in range(n_params)]
    bins_amp = np.linspace(-0.7, 0.7, 20)
    bins_phase = np.linspace(0, 2*np.pi, 20)
    bins = [bins_amp, bins_phase]
    
    # Create the scatter plot matrix
    for i in range(n_params):
        for j in range(n_params):
            # Get the correct axis
            if n_params == 1:
                ax = axes
            else:
                ax = axes[i, j]
                
            # Diagonal: show histogram
            if i == j:
                ax.hist(parameters[:, i], bins=bins[i%2], density=True, alpha=1, color='blue', label='Training')
                ax.hist(additional_parameters[:, i], bins=bins[i%2], density=True, alpha=0.5, color='red', label='Validation')
                ax.set_title(f'{param_labels[i]} Distribution')
                if i == 0:  # Add legend only to first histogram
                    ax.legend()
            # Off-diagonal: show scatter plot
            else:
                # Plot current parameters in blue
                scatter1 = ax.scatter(parameters[:, j], parameters[:, i], 
                                   alpha=0.5, s=5, color='blue', label='Training')
                # Plot additional parameters in red
                scatter2 = ax.scatter(additional_parameters[:, j], additional_parameters[:, i], 
                                   alpha=0.9, s=10, color='red', label='Validation')
                
                # Add legend only to the first scatter plot
                if i == 1 and j == 0:
                    ax.legend()
            
            # Set labels only on the bottom and left edges
            if i == n_params-1:
                ax.set_xlabel(param_labels[j])
            if j == 0:
                ax.set_ylabel(param_labels[i])
                
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
    plt.suptitle('Parameter Scatter Plot Matrix', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the overall title
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'parameter_scatter_matrix.png'), dpi=150)
    plt.close()
    
    print(f"Scatter plot matrix saved to {os.path.join(output_dir, 'parameter_scatter_matrix.png')}")

def main():
    parser = argparse.ArgumentParser(description='Extract and plot parameters from client scripts.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing client.*.sh files')
    
    args = parser.parse_args()
    output_dir = os.path.dirname(args.input_dir.rstrip('/'))
    
    ids, parameters = gather_parameters(args.input_dir)
    
    if len(parameters) == 0:
        print("No valid parameters found in the client scripts.")
        return
    
    print(f"Found {len(parameters)} client scripts with parameters.")
    print(f"Each client has {parameters.shape[1]} parameters.")
    
    plot_parameters(ids, parameters, output_dir)


if __name__ == "__main__":
    main()