import gc
import psutil
import jax
from datetime import datetime

import tracemalloc
import numpy as np
import os

class MemoryMonitor:
    def __init__(self, path='.', off=False):
        self.process = psutil.Process()
        self.file = f"{path}/memory_stats.txt"
        self.off = off

    def get_stats(self):
        # CPU Memory
        mem = self.process.memory_info()
        cpu_rss = mem.rss / 1024 / 1024 / 1024
        cpu_vms = mem.vms / 1024 / 1024 / 1024

        # System RAM
        sys_mem = psutil.virtual_memory()
        ram_available = sys_mem.available / 1024 / 1024 / 1024

        # JAX arrays GPU
        live_arrays = jax.live_arrays(platform='gpu')
        jax_mem = sum(x.size for x in live_arrays) / 1024 / 1024 / 1024

        # JAX arrays CPU
        live_arrays_cpu = jax.live_arrays(platform='cpu')
        jax_mem_cpu = sum(x.size for x in live_arrays_cpu) / 1024 / 1024 / 1024

        # Garbage collection
        num_objects = len(gc.get_objects())

        return {
            'cpu_rss': cpu_rss,
            'cpu_vms': cpu_vms,
            'ram_available': ram_available,
            'jax_arrays_gpu': jax_mem,
            'num_jax_arrays_gpu': len(live_arrays),
            'jax_arrays_cpu': jax_mem_cpu,
            'num_jax_arrays_cpu': len(live_arrays_cpu),
            'num_objects': num_objects,
        }

    def print_stats(self, text=None, iteration=None, with_timestamp=False):
        """Print current memory stats to the initialized file."""
        if self.off:
            return
        stats = self.get_stats()
        to_print = ''
        if with_timestamp:
            to_print += datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' | '
        if iteration is not None:
            to_print += str(iteration) + ' : '
        if text is not None:
            to_print += text + '\n'

        to_print += (
            f"CPU_RSS: {stats['cpu_rss']: 3.3f} Gb | "
            f"CPU_MS: {stats['cpu_vms']: 3.3f} Gb | "
            f"RAM_free: {stats['ram_available']: 3.3f} Gb ||| "
            f"JAX_mem (GPU): {stats['jax_arrays_gpu']: 3.3f} Gb | "
            f"N_JAX_Arr (GPU): {stats['num_jax_arrays_gpu']: 6d} | "
            f"JAX_mem (CPU): {stats['jax_arrays_cpu']: 3.3f} Gb | "
            f"N_JAX_Arr (CPU): {stats['num_jax_arrays_cpu']: 6d} | "
            f"N_Objs: {stats['num_objects']: 6d}"
        )
        with open(self.file, 'a', encoding='utf-8') as f:
            f.write(to_print + '\n')

class MemoryTracer:
    """Tracks both general Python and NumPy-specific memory allocations."""
    
    def __init__(self, log_dir='.', off=False):
        self.log_file = os.path.join(log_dir, "memory_trace.log")
        self.off = off
        if not off:
            # Create NumPy domain filter
            self.np_domain = np.lib.tracemalloc_domain
            self.np_filter = tracemalloc.DomainFilter(
                inclusive=True,
                domain=self.np_domain
            )
            self.no_np_filter = tracemalloc.DomainFilter(
                inclusive=False,
                domain=self.np_domain
            )
            
            # Start tracing if not already started
            if not tracemalloc.is_tracing():
                tracemalloc.start(25)  # Track 25 frames
                
            # Initialize log file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(f"Memory Trace Log - Started at {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")


    def _format_size(self, size_bytes):
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def take_snapshot(self, label=""):
        """Takes a snapshot of current memory usage, both general and NumPy-specific."""
        if self.off:
            return
        snapshot = tracemalloc.take_snapshot()
        
        # Get general Python stats (excluding NumPy)
        python_snapshot = snapshot.filter_traces([self.no_np_filter])
        python_stats = python_snapshot.statistics('traceback')
        
        # Get NumPy-specific stats
        numpy_snapshot = snapshot.filter_traces([self.np_filter])
        numpy_stats = numpy_snapshot.statistics('traceback')
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            # Write header
            f.write(f"\n=== Snapshot: {label} at {timestamp} ===\n")
            
            # Write general Python stats
            total_python = sum(stat.size for stat in python_stats)
            total_python_count = sum(stat.count for stat in python_stats)
            f.write("\nGeneral Python Memory Usage:\n")
            f.write(f"Total: {self._format_size(total_python)} in {total_python_count} blocks\n")
            f.write("\nTop 10 Python allocation sites:\n")
            for stat in python_stats[:10]:
                f.write(f"  {self._format_size(stat.size)} ({stat.count} blocks) - {stat.traceback.format()[-1]}\n")
            
            # Write NumPy stats
            total_numpy = sum(stat.size for stat in numpy_stats)
            total_numpy_count = sum(stat.count for stat in numpy_stats)
            f.write("\nNumPy Memory Usage:\n")
            f.write(f"Total: {self._format_size(total_numpy)} in {total_numpy_count} blocks\n")
            f.write("\nTop 10 NumPy allocation sites:\n")
            for stat in numpy_stats[:10]:
                f.write(f"  {self._format_size(stat.size)} ({stat.count} blocks) - {stat.traceback.format()[-1]}\n")
            
            f.write("-" * 80 + "\n")
        
        tracemalloc.clear_traces()
        
        return {
            'python': {
                'total_bytes': total_python,
                'total_count': total_python_count,
                'top_allocs': [{
                    'size': stat.size,
                    'count': stat.count,
                    'location': stat.traceback.format()[-1]
                } for stat in python_stats[:5]]
            },
            'numpy': {
                'total_bytes': total_numpy,
                'total_count': total_numpy_count,
                'top_allocs': [{
                    'size': stat.size,
                    'count': stat.count,
                    'location': stat.traceback.format()[-1]
                } for stat in numpy_stats[:5]]
            }
        }

    def clear_traces(self):
        """Clears current traces but continues tracking."""
        tracemalloc.clear_traces()

    def __del__(self):
        """Ensures tracemalloc is stopped when the tracer is destroyed."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()