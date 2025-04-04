import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

def parse_log_file(filename):
    data = []
    ts_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    mem_pattern = (
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
            # Look for cache clear events
            if "clearing cache" in line.lower():
                if current_timestamp:
                    data.append({
                        'timestamp': current_timestamp,
                        'event': 'Cache Cleared',
                        'is_cache_clear': True
                    })
                continue
                
            ts_match = re.search(ts_pattern, line)
            if ts_match:
                current_timestamp = datetime.strptime(ts_match.group(1), 
                                                    '%Y-%m-%d %H:%M:%S')
                event = line.split('|')[-1].strip() if '|' in line else line.strip()
                continue
            
            mem_match = re.search(mem_pattern, line)
            if mem_match and current_timestamp:
                data.append({
                    'timestamp': current_timestamp,
                    'event': event,
                    'is_cache_clear': False,
                    'cpu_rss': float(mem_match.group(1)),
                    'cpu_vms': float(mem_match.group(2)),
                    'ram_free': float(mem_match.group(3)),
                    'jax_mem': float(mem_match.group(4)),
                    'n_jax_arrays': int(mem_match.group(5)),
                    'n_objects': int(mem_match.group(6))
                })
    
    return pd.DataFrame(data)

def create_plot(df, y_cols, title, ylabel, colors=None):
    plt.figure(figsize=(15, 6))
    ax = plt.gca()
    
    # Convert to relative time in minutes
    df['minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
    if not isinstance(y_cols, list):
        y_cols = [y_cols]
    if not colors:
        colors = ['blue', 'green', 'red'][:len(y_cols)]
    
    # Plot each metric
    for y_col, color in zip(y_cols, colors):
        ax.plot(df['minutes'], df[y_col], 
                label=y_col.replace('_', ' ').title(), 
                color=color)
    
    # Add cache clear markers
    cache_clears = df[df['is_cache_clear']]
    for _, row in cache_clears.iterrows():
        ax.axvline(x=row['minutes'], color='purple', alpha=0.3, 
                   linestyle='--')#, label='Cache Clear')
        ax.annotate('Cache Clear', 
                   (row['minutes'], ax.get_ylim()[1]),
                   rotation=90, 
                   verticalalignment='top')
        
    # Add validation markers
    val_starts = df[df['event'].str.contains('Before validation start', na=False)]
    val_ends = df[df['event'].str.contains('End of validation', na=False)]
    
    for _, row in val_starts.iterrows():
        ax.axvline(x=row['minutes'], color='green', alpha=0.2, linestyle='-.')
        ax.annotate('Val Start', 
                   (row['minutes'], ax.get_ylim()[1] * 0.95),  # Slightly lower than cache clear
                   rotation=90, 
                   color='green',
                   verticalalignment='top')
    
    for _, row in val_ends.iterrows():
        ax.axvline(x=row['minutes'], color='red', alpha=0.2, linestyle='-.')
        ax.annotate('Val End', 
                   (row['minutes'], ax.get_ylim()[1] * 0.90),  # Even lower
                   rotation=90, 
                   color='red',
                   verticalalignment='top')
    
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    return plt.gcf()

def plot_all_stats(df, output_dir):
    # CPU Memory (RSS and VMS)
    fig = create_plot(
        df, 
        ['cpu_rss', 'cpu_vms'], 
        'CPU Memory Usage (RSS vs VMS)', 
        'Memory (GB)',
        ['blue', 'orange']
    )
    fig.savefig(f'{output_dir}/cpu_memory.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Other metrics
    plots = [
        ('ram_free', 'System Free RAM', 'Memory (GB)', 'green'),
        ('jax_mem', 'JAX Memory Usage', 'Memory (GB)', 'red'),
        ('n_jax_arrays', 'JAX Arrays Count', 'Count', 'blue'),
        ('n_objects', 'Python Objects Count', 'Count', 'purple')
    ]
    
    for y_col, title, ylabel, color in plots:
        fig = create_plot(df, y_col, title, ylabel, [color])
        fig.savefig(f'{output_dir}/{y_col}_over_time.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    import os
    
    log_file = "experiments/across_pdes/norm_mode/melissa-20250404T173708/memory_stats.txt"
    output_dir = "memory_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    df = parse_log_file(log_file)
    plot_all_stats(df, output_dir)
    
    # Print enhanced statistics
    print("\nMemory Statistics Summary:")
    for col in ['cpu_rss', 'cpu_vms', 'ram_free', 'jax_mem']:
        print(f"\n{col.upper()}:")
        print(f"  Mean: {df[col].mean():.2f} GB")
        print(f"  Max:  {df[col].max():.2f} GB")
        print(f"  Min:  {df[col].min():.2f} GB")
        print(f"  Growth: {df[col].iloc[-1] - df[col].iloc[0]:.2f} GB")
