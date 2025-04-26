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
    r_ids = []
    is_bs = []
    c_ids = []
    timestamps = []
    
    # Get all client.*.sh files and sort them
    # client.X.Y.sh and cleint.XB.Y.sh are in the directory
    # Sort by the numeric part after 'client.' and by the numeric part after 'B' if present
    all_files = Path(directory).glob("client.*.*.sh")
    b_filter = lambda x: x.split('B')[0] if 'B' in x else x
    
    sorted_files = sorted(
        all_files,
        key=lambda x: (int(b_filter(x.stem.split('.')[1])), int(x.stem.split('.')[2]))
        )
    for file_path in sorted_files:
        resampling_id = file_path.stem.split('.')[1].split('B')[0]
        is_b = 'B' in file_path.stem
        client_id = int(file_path.stem.split('.')[2])
        params = extract_parameters(file_path)
        if params:
            print("From file", file_path.stem, resampling_id, is_b, client_id)
            r_ids.append(int(resampling_id))
            is_bs.append(is_b)
            c_ids.append(int(client_id))
            parameters.append(params)
            timestamps.append(os.path.getctime(file_path))
    
    return np.array(r_ids), np.array(is_bs), np.array(c_ids), np.array(parameters), np.array(timestamps)

def plot_parameters(r_ids, is_bs, c_ids, parameters, timestamps, output_dir, additional_params_path, agg=True, stack=False):
    """Create a scatter plot matrix of all parameters."""

    r_ids[~is_bs] = 0
    # r_ids is the color for resampling generation
    # c_ids is just the client id
    # is_bs is a boolean array indicating if the client is from proposal of not

    # Load additional data points
    additional_parameters = np.load(additional_params_path)
    if additional_parameters.shape[1] < parameters.shape[1]:
        additional_parameters = np.concatenate((additional_parameters, np.full((additional_parameters.shape[0], parameters.shape[1] - additional_parameters.shape[1]), np.nan)), axis=1)
    elif additional_parameters.shape[1] > parameters.shape[1]:
        parameters = np.concatenate((parameters, np.full((parameters.shape[0], additional_parameters.shape[1] - parameters.shape[1]), np.nan)), axis=1)
    
    def agg_parameters(params, fun='mean'):
        """Aggregate parameters by taking the mean of every two consecutive parameters."""
        amps = params[:,::2]
        phs = params[:,1::2]
        if fun == 'mean+std':
            amplitude = amps.mean(axis=1, keepdims=True)
            phase = phs.mean(axis=1, keepdims=True)
            amplitude_std = amps.std(axis=1, keepdims=True)
            phase_std = phs.std(axis=1, keepdims=True)
            return np.concatenate((amplitude, phase, amplitude_std, phase_std), axis=1)
        elif fun == 'norm':
            amplitude = np.linalg.norm(amps, axis=1, keepdims=True) / np.linalg.norm(np.array([1.0] * (params.shape[1]//2)))
            phase = np.linalg.norm(phs, axis=1, keepdims=True) / np.linalg.norm(np.array([2 * np.pi] * (params.shape[1]//2)))
            return np.concatenate((amplitude, phase), axis=1)
        elif fun == "ic_std_mean":
            # ic = sum_1..5 A sin(phase x)
            # A shape (N, 5), ph shape (N, 5), x shape (1, 100)
            # A * sin(ph * x) shape (N, 5, 100)
            # sum_1..5 A sin(phase x) shape (N, 100)
            # ic shape (N, 100)
            x = np.linspace(0, 1, 100).reshape(1, -1)
            ic = np.sum(amps[..., None] * np.sin(phs[..., None] * x), axis=1)
            ic_std = np.std(ic, axis=1, keepdims=True)
            ic_mean = np.mean(ic, axis=1, keepdims=True)
            return np.concatenate((ic_mean, ic_std), axis=1)
        else:
            raise ValueError("Unsupported aggregation function.")
    n_p = parameters.shape[1]//2

    params = ['Amplitude', 'Phase']

    if agg == 'mean+std':
        agg_label = ['Mean', 'Std']
        parameters = agg_parameters(parameters, fun='mean+std')
        additional_parameters = agg_parameters(additional_parameters, fun='mean+std')
        min_params_all = np.array([-1.0, 0, 0, 0])
        max_params_all = np.array([1.0, 2 * np.pi, 1.0, np.pi])
    elif agg == 'norm':
        agg_label = ['Norm']
        parameters = agg_parameters(parameters, fun='norm')
        additional_parameters = agg_parameters(additional_parameters, fun='norm')
        min_params_all = np.array([0, 0])
        max_params_all = np.array([1, 1])
    elif agg == 'ic_std_mean':

        params = ['IC_Mean', 'IC_Std']
        agg_label = [' ']
        parameters = agg_parameters(parameters, fun='ic_std_mean')
        additional_parameters = agg_parameters(additional_parameters, fun='ic_std_mean')
        a = np.concatenate([np.full((50,), 0.7*np.pi), np.full((50,), -0.7 * np.pi)]).std()
        print(a)
        min_params_all = np.array([-1, 0])
        max_params_all = np.array([1, a/1.5])
    else:
        min_params_all = np.array([-1.0, 0] * n_p)
        max_params_all = np.array([1.0, 2 * np.pi] * n_p)
    # Get number of parameters per data point
    n_params = parameters.shape[1]
    
    # Create a figure for the scatter plot matrix
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Parameter labels
    nbins = 10
    if agg is not None:
        param_labels = [f'{params[i%2]} {agg_label[i//2]}' for i in range(n_params)]
    else:
        param_labels = [f'{params[i%2]} {i//2}' for i in range(n_params)]
    bins = [np.linspace(min_params_all[i], max_params_all[i], nbins) for i in range(n_params)]

    n_resampling = len(np.unique(r_ids))
    n_intervals = n_resampling - 1 if n_resampling > 1 else 1

    colors_scatter = plt.cm.viridis(np.linspace(0., 1, n_intervals))  # Increased contrast
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
            if j > i:
                # Hide the upper triangle
                ax.axis('off')
                continue
            # Diagonal: show histogram
            elif i == j:
                if n_intervals > 1 and stack == True:
                    histmatrix = np.zeros((n_intervals + 1, len(bins[i])-1))
                    histmatrix[0] = np.histogram(additional_parameters[:, i], bins=bins[i], density=True)[0]
                    for k in range(n_intervals):
                        mask = r_ids == k
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
                    if is_bs.any():
                        
                        mask = (r_ids <= n_intervals) & is_bs
                        ledeux = [parameters[mask, i], parameters[~is_bs, i]]
                        ax.hist(ledeux, bins=bins[i], histtype='bar', color=['blue', 'silver'], label=['Proposal', 'Random'], alpha=0.5)
                        # ax.hist(parameters[mask, i], bins=bins[i], alpha=0.5, color='blue', label='Proposal')
                        # ax.hist(parameters[~is_bs, i], bins=bins[i], color='silver', label='Random')
                        ax.hist(parameters[r_ids>=n_intervals-1, i], bins=bins[i], alpha=0.7, color='yellow', label='Last 2')
                        vl = ax.get_ylim()[1]
                        ax.vlines([np.mean(parameters[mask, i])], 0, vl, color='blue', linestyle='--',)
                        ax.vlines([np.mean(parameters[~is_bs, i])], 0, vl, color='black', linestyle='--')
                    else:
                        ax.hist(parameters[~is_bs, i], bins=bins[i], color='silver', label='Random')
                        vl = ax.get_ylim()[1]
                        ax.vlines([np.mean(parameters[~is_bs, i])], 0, vl, color='black', linestyle='--')
                    ax.hist(additional_parameters[:, i], bins=bins[i], histtype='step', alpha=1.0, color='red', label='Validation')
                    ax.set_title(f'{param_labels[i]} Distribution')
                    if i == 0:
                        ax.legend(loc='upper right', fontsize='small')
                    # ax.set_xticks((np.arange(len(bins[i])-1) + 0.5)[::2])
                    # ax.set_xticklabels([f'{bins[i][k]:.2f}' for k in range(len(bins[i])-1)][::2], rotation=45)
            # Off-diagonal: show scatter plot
            else:
                # Plot additional parameters in red
                scatter2 = ax.scatter(additional_parameters[:, j], additional_parameters[:, i], 
                                   alpha=1, s=5, color='red', marker='v',label='Validation')
                scatter1 = ax.scatter(parameters[~is_bs, j], parameters[~is_bs, i], 
                                       c='silver',
                                       s=5, label='Random',
                                       alpha=0.5 if n_intervals > 1 else 0.9,
                                       )
                for rrr in range(1,n_intervals):
                    mask = r_ids == rrr
                    if np.any(mask):
                        ax.scatter(parameters[mask, j], parameters[mask, i], 
                                       c=r_ids[mask],
                                       cmap=custom_cmap_scatter,
                                        norm=norm_scatter,
                                       s=10, label=f'R{rrr+1}',
                                       edgecolor='black', linewidth=0.1)
                ax.set_xlim(min_params_all[j], max_params_all[j])
                ax.set_ylim(min_params_all[i], max_params_all[i])
            
            # Set labels only on the bottom and left edges
            if i == n_params-1:
                ax.set_xlabel(param_labels[j])
            if j == 0:
                ax.set_ylabel(param_labels[i])
                
            # Add grid for better readability
            # ax.grid(True, linestyle='--', alpha=0.7)
    

    # Adjust the layout to make room for the colorbar
    plt.tight_layout(rect=[0, 0, 0.95, 0.97])
    # Add a single colorbar for the entire figure
    cax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap_scatter, norm=norm_scatter)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Resampling Group')
    cbar.set_ticks(np.arange(n_intervals) + 0.5)  # Center ticks in each interval
    cbar.set_ticklabels([f'R{i+1}' for i in range(n_intervals)])

    
    if stack:
        cax1 = fig.add_axes([1.02, 0.15, 0.02, 0.7])
        sm1 = plt.cm.ScalarMappable(cmap='Oranges')#, vmin=0, vmax=1)
        sm1.set_array([])
        cbar1 = plt.colorbar(sm1, cax=cax1)
        cbar1.set_label('Density')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'parameter_scatter_matrix_{agg}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Scatter plot matrix saved to", os.path.join(output_dir, f'parameter_scatter_matrix_{agg}.png'))

def main():
    parser = argparse.ArgumentParser(description='Extract and plot parameters from client scripts.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing client.*.sh files')
    parser.add_argument('--agg', default='', type=str,
                        choices=['mean', 'std', 'mean+std', 'norm', 'ic_std_mean'],
                        help='Aggregation function to apply to parameters')
    parser.add_argument('--stack', action='store_true', default=False,
                        help='Stack histograms for resampling groups')
    parser.add_argument('--validation-path', type=str, default='',
                        help='Path to the validation parameters file')

    args = parser.parse_args()
    output_dir = os.path.dirname(args.input_dir.rstrip('/'))
    if args.agg == '':
        args.agg = None

    r_ids, is_bs, c_ids, parameters, timestamps = gather_parameters(args.input_dir)

    if len(parameters) == 0:
        print("No valid parameters found in the client scripts.")
        return

    print(f"Found {len(parameters)} client scripts with parameters.")
    print(f"Each client has {parameters.shape[1]} parameters.")
    # path = "/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/burgers_1d/high_res_faster_default_5waves/trajectories/input_parameters.npy"
    if not os.path.exists(args.validation_path) and args.validation_path != '':
        print(f"Validation parameters file not found at {args.validation_path}.")
        return

    plot_parameters(r_ids, is_bs, c_ids, parameters, timestamps, output_dir, args.validation_path, args.agg, args.stack)


if __name__ == "__main__":
    main()