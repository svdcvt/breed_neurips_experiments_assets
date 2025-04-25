import pandas as pd
import sys
import matplotlib.pyplot as plt

if __name__ == "__main__":    
    try:
        df = pd.read_csv('gpu_stats.csv', names=['timestamp', 'gpu_util', 'mem_util'], skiprows=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        for col in df.columns[1:]:
            df[col] = df[col].str.rstrip('%').astype('float') / 100.0
        
        plot = df.plot(x="timestamp", kind="line", title="GPU Utilization")
        plt.savefig("gpu_stats_plot.png")
        
    except Exception as e:
        print(f"Error: {e}")
