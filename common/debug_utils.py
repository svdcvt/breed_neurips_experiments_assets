import psutil
import jax
from datetime import datetime


class MemoryMonitor:
    def __init__(self, path='.'):
        self.process = psutil.Process()
        self.file = open(f"{path}/memory_stats.txt", "w")
        
    def get_stats(self):
        # CPU Memory
        mem = self.process.memory_info()
        cpu_rss = mem.rss / 1024 / 1024
        
        # System RAM
        sys_mem = psutil.virtual_memory()
        ram_available = sys_mem.available / 1024 / 1024
        
        # JAX arrays
        live_arrays = jax.live_arrays()
        jax_mem = sum(x.size for x in live_arrays) / 1024 / 1024
        
        return {
            'cpu_mb': cpu_rss,
            'ram_available_mb': ram_available,
            'jax_arrays_mb': jax_mem,
            'num_jax_arrays': len(live_arrays)
        }
        
    def print_stats(self, text):
        """Print current memory stats to the initialized file."""
        stats = self.get_stats()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Logging at:", text, file=self.file)
        print(f"Timestamp: {timestamp}", file=self.file)
        print(f"CPU Memory: {stats['cpu_mb']:.2f} MB", file=self.file)
        print(f"Available RAM: {stats['ram_available_mb']:.2f} MB", file=self.file)
        print(f"JAX Arrays: {stats['jax_arrays_mb']:.2f} MB", file=self.file)
        print(f"Number of JAX Arrays: {stats['num_jax_arrays']}", file=self.file
        )
        print("-" * 50, file=self.file)
