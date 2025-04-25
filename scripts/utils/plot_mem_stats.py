import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

import os
import argparse
import json

def load_config(log_dir):
    """Load configuration from config.json"""
    config_path = os.path.join(log_dir, "config_mpi.json")
    try:
        # First, read and strip comments
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Remove JavaScript style comments
        import re
        content = re.sub(r'//.*?\n', '\n', content)  # Remove single line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # Remove multi-line comments
        
        # Parse the cleaned JSON
        config = json.loads(content)
        
        # Extract from study_options and dl_config
        study_opts = config.get('study_options', {})
        dl_config = config.get('dl_config', {})
        launcher_config = config.get('launcher_config', {})
        
        return {
            'model': study_opts.get('network_config', 'N/A'),
            'parameter_sweep_size': study_opts.get('parameter_sweep_size', 'N/A'),
            'nb_batches_update': dl_config.get('nb_batches_update', 'N/A'),
            'per_server_watermark': dl_config.get('per_server_watermark', 'N/A'),
            'buffer_size': dl_config.get('buffer_size', 'N/A'),
            'batch_size': dl_config.get('batch_size', 'N/A'),
            'valid_batch_size': dl_config.get('valid_batch_size', 'N/A'),
            'clear_freq': dl_config.get('clear_freq', ['N/A', 'N/A']),
            'job_limit' : launcher_config.get('job_limit', 'N/A'),
        }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}")
        return {
            'parameter_sweep_size': 'N/A',
            'nb_batches_update': 'N/A',
            'per_server_watermark': 'N/A',
            'buffer_size': 'N/A',
            'batch_size': 'N/A',
            'clear_freq': ['N/A', 'N/A'],
            'valid_batch_size': 'N/A',
            'model': 'N/A',
            'job_limit': 'N/A',
        }

def parse_log_file(filename):
    data = []
    ts_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    mem_pattern = (
        r'CPU_RSS:\s*(\d+\.?\d*)\s*Gb\s*\|\s*'
        r'CPU_MS:\s*(\d+\.?\d*)\s*Gb\s*\|\s*'  # This is VMS
        r'RAM_free:\s*(\d+\.?\d*)\s*Gb\s*\|\|\|\s*'
        r'JAX_mem\s*\(GPU\):\s*(\d+\.?\d*)\s*Gb\s*\|\s*'
        r'N_JAX_Arr\s*\(GPU\):\s*(\d+)\s*\|\s*'
        r'JAX_mem\s*\(CPU\):\s*(\d+\.?\d*)\s*Gb\s*\|\s*'
        r'N_JAX_Arr\s*\(CPU\):\s*(\d+)\s*\|\s*'
        r'N_Objs:\s*(\d+)'
    )
    mem_pattern_old = (
        r'CPU_RSS:\s+(\d+\.?\d*)\s+Gb\s+\|\s+'
        r'CPU_MS:\s+(\d+\.?\d*)\s+Gb\s+\|\s+'  # This is VMS
        r'RAM_free:\s+(\d+\.?\d*)\s+Gb\s+\|\|\|\s+'
        r'JAX_mem:\s+(\d+\.?\d*)\s+Gb\s+\|\s+'
        r'N_JAX_Arr:\s+(\d+)\s+\|\s+'
        r'N_Objs:\s+(\d+)'
    )
    
    current_timestamp = None
    event = None
    
    with open(filename, 'r') as f:
        for line in f:
            
            ts_match = re.search(ts_pattern, line)
            if ts_match: # it is "first" line (time, batch, event)
                current_timestamp = datetime.strptime(ts_match.group(1), 
                                                    '%Y-%m-%d %H:%M:%S')
                event = line.split('|')[-1].strip() if '|' in line else line.strip()
                
                info = {
                        'timestamp': current_timestamp,
                        'event': None,
                        'is_cache_clear': False,
                        'is_garbage_collect': False,
                        'training': False,
                        'validation': False
                    }
                
                if "preparing training attributes" in event.lower():
                    info['event'] = "Initialization end"
                elif "before validation start" in line.lower():
                    info['event'] = "Val Start"
                    info['validation'] = True
                elif "garbage" in line.lower():
                    info['event'] = "Garbage collection"
                    info['is_garbage_collect'] = True
                elif "end of validation" in line.lower():
                    info['event'] = "Val End"
                    info['validation'] = True
                elif "clearing cache" in line.lower():
                    info['event'] = "Cache cleared"
                    info['is_cache_clear'] = True
                
                if info['event'] is not None and "in training" in line.lower():
                    info['event'] += " in training"
                    info['training'] = True

            mem_match = re.search(mem_pattern, line)
            if mem_match and current_timestamp:
                
                info.update({
                    'timestamp': current_timestamp,
                    'cpu_rss': float(mem_match.group(1)),
                    'cpu_vms': float(mem_match.group(2)),
                    'ram_free': float(mem_match.group(3)),
                    'jax_mem_gpu': float(mem_match.group(4)),
                    'n_jax_arrays_gpu': int(mem_match.group(5)),
                    'jax_mem_cpu': float(mem_match.group(6)),
                    'n_jax_arrays_cpu': int(mem_match.group(7)),
                    'n_objects': int(mem_match.group(8))
                })
                data.append(info)
            else:
                mem_match = re.search(mem_pattern_old, line)
                if mem_match and current_timestamp:
                    
                    info.update({
                        'timestamp': current_timestamp,
                        'cpu_rss': float(mem_match.group(1)),
                        'cpu_vms': float(mem_match.group(2)),
                        'ram_free': float(mem_match.group(3)),
                        'jax_mem_gpu': float(mem_match.group(4)),
                        'n_jax_arrays_gpu': int(mem_match.group(5)),
                        'n_objects': int(mem_match.group(6))
                    })
                    data.append(info)
    
    return pd.DataFrame(data)

def parse_sim_completions(server_log_file):
    """Parse the melissa_server log file to extract simulation completion events."""
    sim_completions = []
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*sim-id=(\d+) has finished sending'
    
    with open(server_log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                timestamp = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S,%f')
                sim_id = int(match.group(2))
                sim_completions.append({'timestamp': timestamp, 'sim_id': sim_id})
    if len(sim_completions) == 0:
        print(f"Warning: No simulation completions found in {server_log_file}.")
        return None
    return pd.DataFrame(sim_completions)

def create_subplot(ax, df, sim_completions_df, y_cols, title, ylabel, colors):
    """Create a single subplot with metrics and annotations"""
    # Plot metrics
    if not isinstance(y_cols, list):
        y_cols = [y_cols]
    for y_col, color in zip(y_cols, colors):
        try:
            ax.plot(df['minutes'], df[y_col], 
                    label=y_col.replace('_', ' ').title(), linewidth=1,
                    color=color)
        except KeyError:
            print(f"Warning: Column '{y_col}' not found in DataFrame.")
    
    if sim_completions_df is not None and not sim_completions_df.empty:
        for _, row in sim_completions_df.iterrows():
            ax.axvline(x=row['minutes'], color='teal', alpha=0.15, linewidth=1, linestyle=':')
            ax.annotate(f"sim{row['sim_id']}", 
                      (row['minutes'], ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05),
                      rotation=90, 
                      color='teal',
                      fontsize=6,
                      verticalalignment='bottom')
    
    # Add validation markers
    vals = df[df['validation']]

    for _, row in vals.iterrows():
        ax.axvline(x=row['minutes'], color='green' if row['event'].endswith('Start') else 'red', alpha=0.2, linewidth=10)
        ax.annotate(row['event'], 
                   (row['minutes'], ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5),
                   rotation=90, 
                   color='green' if row['event'].endswith('Start') else 'red',
                   verticalalignment='top')
    
    if not df[df['event'] == "Initialization end"].empty:
        row = df[df['event'] == "Initialization end"].iloc[0]
        ax.axvline(x=row['minutes'], color='black', alpha=0.2, linewidth=10)
        ax.annotate(row['event'], 
                    (row['minutes'], ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5),
                    rotation=90, 
                    color='black',
                    verticalalignment='top')
    
    # Add cache clear markers
    cache_clears = df[df['is_cache_clear']]
    for _, row in cache_clears.iterrows():
        ax.axvline(x=row['minutes'], color='darkviolet', linestyle='--' if row['training'] else '-')
        ax.annotate('Cache', 
                   (row['minutes'] + 0.01, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95),
                   rotation=90,
                   color='darkviolet',
                   verticalalignment='top')
        
    cache_clears = df[df['is_garbage_collect']]
    for _, row in cache_clears.iterrows():
        ax.axvline(x=row['minutes'], color='darkviolet', linestyle='--' if row['training'] else '-')
        ax.annotate('Garbage', 
                   (row['minutes'] + 0.01, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.85),
                   rotation=90,
                   color='darkviolet',
                   verticalalignment='top')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='upper left')


def plot_all_stats(df, sim_completions_df, output_dir, config_info, fromm=None, until=None, comment=None):
    """Create figure with all memory statistics"""
    # Convert to relative time in minutes
    df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    min_timestamp = df['timestamp'].min()
    if sim_completions_df is not None and not sim_completions_df.empty:
        sim_completions_df['minutes'] = (sim_completions_df['timestamp'] - min_timestamp).dt.total_seconds() / 60
        
        # Filter by time range
        if fromm is not None:
            sim_completions_df = sim_completions_df[sim_completions_df['minutes'] >= fromm]
        if until is not None:
            sim_completions_df = sim_completions_df[sim_completions_df['minutes'] <= until]
    
    # Filter DataFrame based on time range
    if fromm is not None:
        df = df[df['minutes'] >= fromm]
    if until is not None:
        df = df[df['minutes'] <= until]
    
    # Create figure with subplots
    fig, axs = plt.subplots(6, 1, figsize=(15, 30))

    # Format comment text with line splitting
    if comment:
        # Split comment into lines of max 50 characters
        comment_lines = []
        words = comment.split()
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= 30:  # +1 for space
                current_line.append(word)
                current_length += len(word) + 1
            else:
                comment_lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            comment_lines.append(' '.join(current_line))
        
        comment_text = '\n'.join(comment_lines)
    else:
        comment_text = 'N/A'
    
    config_text = (
        f"Configuration:\n"
        f"parameter_sweep_size: {config_info['parameter_sweep_size']}\n"
        f"model: {config_info['model']}\n"
        f"batch_size: {config_info['batch_size']}\n"
        f"valid_batch_size: {config_info['valid_batch_size']}\n"
        f"nb_batches_update: {config_info['nb_batches_update']}\n"
        f"per_server_watermark: {config_info['per_server_watermark']}\n"
        f"buffer_size: {config_info['buffer_size']}\n"
        f"job_limit: {config_info['job_limit']}\n"
        f"clear_freq: {config_info['clear_freq']}"
        f"\n\n"
        f"Comment:\n{comment_text}\n"
    )
    if fromm is not None or until is not None:
        config_text += f"Time range: {fromm} to {until} minutes\n"

    # Define all plots configuration
    plots = [
        (['cpu_rss'], 'CPU Memory Usage RSS', 'Memory (GB)', ['blue']),
        (['cpu_vms'], 'CPU Memory Usage VMS', 'Memory (GB)', ['blue']),
        (['ram_free'], 'System Free RAM', 'Memory (GB)', ['blue']),
        (['jax_mem_gpu'], 'JAX Memory Usage (GPU)', 'Memory (GB)', ['blue']),
        (['n_jax_arrays_gpu'], 'JAX Arrays Count (GPU)', 'Count', ['blue']),
        (['n_objects'], 'Python Objects Count', 'Count', ['blue'])
    ]
    
    # Create each subplot
    for idx, (y_cols, title, ylabel, colors) in enumerate(plots):
        create_subplot(axs[idx], df, sim_completions_df, y_cols, title, ylabel, colors)
    
    fig.text(1.02, 0.98, config_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8), 
             verticalalignment='top')
    
    plt.tight_layout()
    if fromm is None and until is None:
        suffix = "_all"
    else:
        suffix = f"_f{int(fromm)}" if fromm is not None else ""
        suffix += f"_t{int(until)}" if until is not None else ""
    fig.savefig(f'{output_dir}/memory_analysis{suffix}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

def print_statistics(df):
    """Print summary statistics for memory metrics"""
    print("\nMemory Statistics Summary:")
    for col in ['cpu_rss', 'cpu_vms', 'ram_free', 'jax_mem_gpu']:
        print(f"\n{col.upper()}:")
        print(f"  Mean: {df[col].mean():.2f} GB")
        print(f"  Max:  {df[col].max():.2f} GB")
        print(f"  Min:  {df[col].min():.2f} GB")
        print(f"  Growth: {df[col].iloc[-1] - df[col].iloc[0]:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot memory statistics from log files.")
    parser.add_argument(
        "--log_dir", 
        type=str, 
        required=True, 
        help="Directory containing the memory stats log file."
    )
    parser.add_argument(
        "--until", 
        type=float,
        default=None,
        help="Plot data until this time in minutes (optional)"
    )
    parser.add_argument(
        "--from",
        dest="fromm",
        type=float,
        default=None,
        help="Plot data until this time in minutes (optional)"
    )
    parser.add_argument(
        "--comment", 
        type=str, 
        default=None, 
        help="Comment to add to the plot (optional)"
    )
    args = parser.parse_args()

    log_file = os.path.join(args.log_dir, "memory_stats.txt")
    server_log_file = os.path.join(args.log_dir, "melissa_server_0.log")
    output_dir = os.path.join(args.log_dir)
    config_info = load_config(args.log_dir)
    
    df = parse_log_file(log_file)
    sim_completions_df = parse_sim_completions(server_log_file)
    plot_all_stats(df, sim_completions_df, output_dir, config_info, fromm=args.fromm, until=args.until, comment=args.comment)
    print_statistics(df)
