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
    timestamps = []
    
    # Get all client.*.sh files and sort them
    files = sorted(Path(directory).glob("client.*.sh"), 
                  key=lambda x: int(x.stem.split('.')[1]))
    
    for file_path in files:
        client_id = int(file_path.stem.split('.')[1])
        params = extract_parameters(file_path)
        if params:
            ids.append(client_id)
            parameters.append(params)
            timestamps.append(os.path.getctime(file_path))
    
    # Convert timestamps to values between 0 and 1
    min_time = min(timestamps)
    max_time = max(timestamps)
    normalized_times = [(t - min_time)/(max_time - min_time) if max_time != min_time else 0.5 for t in timestamps]
    
    return np.array(ids), np.array(parameters), np.array(normalized_times)

def plot_parameters(ids, parameters, timestamps, output_dir, agg=True):
    """Create a scatter plot matrix of all parameters."""
    
    # Load additional data points
    additional_params_path = "/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/burgers_1d/high_res_faster_default_5waves/trajectories/input_parameters.npy"
    additional_parameters = np.load(additional_params_path)
    if additional_parameters.shape[1] < parameters.shape[1]:
        additional_parameters = np.concatenate((additional_parameters, np.full((additional_parameters.shape[0], parameters.shape[1] - additional_parameters.shape[1]), np.nan)), axis=1)
    elif additional_parameters.shape[1] > parameters.shape[1]:
        parameters = np.concatenate((parameters, np.full((parameters.shape[0], additional_parameters.shape[1] - parameters.shape[1]), np.nan)), axis=1)
    
    def agg_parameters(params, fun='mean'):
        """Aggregate parameters by taking the mean of every two consecutive parameters."""
        if fun == 'mean':
            amplitude = params[:,::2].mean(axis=1, keepdims=True)
            phase = params[:,1::2].mean(axis=1, keepdims=True)
            return np.concatenate((amplitude, phase), axis=1)
        elif fun == 'std':
            amplitude = params[:,::2].std(axis=1, keepdims=True)
            phase = params[:,1::2].std(axis=1, keepdims=True)
        elif fun == 'mean+std':
            amplitude = params[:,::2].mean(axis=1, keepdims=True)
            phase = params[:,1::2].mean(axis=1, keepdims=True)
            amplitude_std = params[:,::2].std(axis=1, keepdims=True)
            phase_std = params[:,1::2].std(axis=1, keepdims=True)
            return np.concatenate((amplitude, phase, amplitude_std, phase_std), axis=1)
        else:
            raise ValueError("Unsupported aggregation function. Use 'mean' or 'std'.")
    
    if agg:
        parameters = agg_parameters(parameters, fun='mean+std')
        additional_parameters = agg_parameters(additional_parameters, fun='mean+std')
        min_params_all = np.array([-0.7, 0, 0, 0])
        max_params_all = np.array([0.7, 2 * np.pi, 0.7, np.pi])
    else:
        min_params_all = np.array([-0.7, 0] * (parameters.shape[1]//2))
        max_params_all = np.array([0.7, 2 * np.pi] * (parameters.shape[1]//2))
    # Get number of parameters per data point
    n_params = parameters.shape[1]
    
    # Create a figure for the scatter plot matrix
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Parameter labels
    params = ['Amplitude', 'Phase']
    nbins = 15
    if agg:
        agg = ['Mean', 'Std']
        param_labels = [f'{params[i%2]} {agg[i//2]}' for i in range(n_params)]
    else:
        param_labels = [f'{params[i%2]} {i//2}' for i in range(n_params)]
    bins = [np.linspace(min_params_all[i], max_params_all[i], nbins) for i in range(n_params)]
    # Check if we need to use timestamps for coloring
    use_timestamps = 'breed' in str(output_dir).lower()
    
    if use_timestamps:
        # Sort timestamps and find gaps
        sorted_times = np.sort(timestamps)
        time_diff = np.diff(sorted_times)
        
        # Find significant gaps (using mean + std as threshold)
        threshold = np.mean(time_diff) + 5 * np.std(time_diff)
        gap_indices = np.where(time_diff > threshold)[0]
        
        # Create time groups based on gaps
        time_edges = []  # Don't start with 0
        for idx in gap_indices:
            time_edges.append((sorted_times[idx] + sorted_times[idx + 1]) / 2)
        
        # Only add edges if we have gaps
        if len(time_edges) > 0:
            time_edges = [sorted_times[0]] + time_edges + [sorted_times[-1]]
        else:
            # If no significant gaps found, use a single group
            time_edges = [sorted_times[0], sorted_times[-1]]
        
        time_edges = np.array(time_edges)
        
        # Convert to normalized values between 0 and 1
        time_edges = (time_edges - time_edges[0]) / (time_edges[-1] - time_edges[0])
        time_edges[-1] += 0.01  # Ensure the last edge is slightly larger to include the last timestamp
        
        # Assign timestamps to groups
        time_groups = np.digitize(timestamps, time_edges) - 1
        # First, define the color mapping setup outside the loop
        n_intervals = len(np.unique(time_groups))

        print(f"Detected {n_intervals} time groups in the data")
        if n_intervals > 5:
            time_groups = (time_groups + 1 )// 2
            n_intervals = len(np.unique(time_groups))
            print(f"Reduced number of time groups to {n_intervals} to avoid too many colors.")
        
        
        # Create a custom colormap that spans from light to dark blue
        colors_scatter = plt.cm.winter_r(np.linspace(0.1, 1, n_intervals))  # Increased contrast
        custom_cmap_scatter = plt.matplotlib.colors.ListedColormap(colors_scatter)
        
        # Create bounds and normalization for discrete colors
        scatter_bounds = np.arange(n_intervals + 1) - 0.5
        norm_scatter = plt.matplotlib.colors.BoundaryNorm(scatter_bounds, n_intervals)
    
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
                if use_timestamps:
                    histmatrix = np.zeros((n_intervals + 1, len(bins[i])-1))
                    histmatrix[0] = np.histogram(additional_parameters[:, i], bins=bins[i], density=True)[0]
                    for k in range(n_intervals):
                        mask = time_groups == k
                        if np.any(mask):
                            histmatrix[k + 1] = np.histogram(parameters[mask, i], bins=bins[i])[0]
                    histmatrix = (histmatrix - histmatrix.min(axis=1, keepdims=True)) / (histmatrix.max(axis=1, keepdims=True) - histmatrix.min(axis=1, keepdims=True)) # Normalize
                    ax.pcolor(histmatrix, cmap='Oranges', vmin=0, vmax=1,
                            #    edgecolor='black', linewidth=0.1
                               )
                    ax.set_xticks((np.arange(len(bins[i])-1) + 0.5)[::2])
                    ax.set_xticklabels([f'{bins[i][k]:.2f}' for k in range(len(bins[i])-1)][::2], rotation=45)
                    ax.set_yticks(np.arange(n_intervals + 1))
                    ax.set_yticklabels(['Val'] + [f'R{k}' for k in range(n_intervals)])
                    ax.set_title(f'{param_labels[i]} Distribution over resmapling')
                    ax.grid(axis='y', color='white', linewidth=2)

                else:
                    ax.hist(parameters[:, i], bins=bins[i], density=True, 
                           alpha=1, color='blue', label='Training')
                    ax.hist(additional_parameters[:, i], bins=bins[i], density=True, 
                           alpha=0.5, color='red', label='Validation')
                ax.set_title(f'{param_labels[i]} Distribution')
                # Remove the legend from here - we'll add it later
            # Off-diagonal: show scatter plot
            else:
                # Plot additional parameters in red
                scatter2 = ax.scatter(additional_parameters[:, j], additional_parameters[:, i], 
                                   alpha=1, s=10, color='red', label='Validation')
                if use_timestamps:
                    scatter1 = ax.scatter(parameters[:, j], parameters[:, i], 
                                       c=time_groups,
                                       cmap=custom_cmap_scatter,
                                       norm=norm_scatter,  # Use the same norm as colorbar
                                       s=15, label='Training',
                                       alpha=0.5)
                else:
                    scatter1 = ax.scatter(parameters[:, j], parameters[:, i], 
                                       alpha=0.5, s=5, color='blue', label='Training')
                
                ax.set_xlim(min_params_all[j], max_params_all[j])
                ax.set_ylim(min_params_all[i], max_params_all[i])
                # Add legend only to the first scatter plot
                if i == 1 and j == 0:
                    ax.legend()
            
            # Set labels only on the bottom and left edges
            if i == n_params-1:
                ax.set_xlabel(param_labels[j])
            if j == 0:
                ax.set_ylabel(param_labels[i])
                
            # Add grid for better readability
            # ax.grid(True, linestyle='--', alpha=0.7)
    

    # Adjust the layout to make room for the colorbar/legend
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    # Add a single legend/colorbar for the entire figure
    if use_timestamps:
        # Create a new axes for the colorbar
        cax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=custom_cmap_scatter, norm=norm_scatter)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Resampling Group')
        cbar.set_ticks(np.arange(n_intervals) + 0.5)  # Center ticks in each interval
        cbar.set_ticklabels([f'R{i+1}' for i in range(n_intervals)])


        cax1 = fig.add_axes([0.95, 0.15, 0.02, 0.7])
        sm1 = plt.cm.ScalarMappable(cmap='Oranges')#, vmin=0, vmax=1)
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, cax=cax1)
        cbar1.set_label('Density')

    else:
        # Original legend for non-timestamp case
        handles = [plt.Rectangle((0,0),1,1, color='blue', alpha=0.7),
                  plt.Rectangle((0,0),1,1, color='red', alpha=0.5)]
        labels = ['Training', 'Validation']
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'parameter_scatter_matrix{"_agg" if agg else "" }.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plot matrix saved to {os.path.join(output_dir, f'parameter_scatter_matrix{"_agg" if agg else "" }.png')}")

def main():
    parser = argparse.ArgumentParser(description='Extract and plot parameters from client scripts.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing client.*.sh files')
    parser.add_argument('--agg', action='store_true', default=False,
                        help='Aggregation function to apply to parameters')
    
    args = parser.parse_args()
    output_dir = os.path.dirname(args.input_dir.rstrip('/'))
    
    ids, parameters, timestamps = gather_parameters(args.input_dir)
    
    if len(parameters) == 0:
        print("No valid parameters found in the client scripts.")
        return
    
    print(f"Found {len(parameters)} client scripts with parameters.")
    print(f"Each client has {parameters.shape[1]} parameters.")
    
    plot_parameters(ids, parameters, timestamps, output_dir, args.agg)


if __name__ == "__main__":
    main()