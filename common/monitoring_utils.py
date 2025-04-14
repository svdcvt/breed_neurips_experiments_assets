import gc
import psutil
import jax
from datetime import datetime

import tracemalloc
import numpy as np
import os

class MemoryMonitor:
    def __init__(self, mode="file+tb", path=None, tb_logger=None, unit='GB'):
        self.process = psutil.Process()
        self.unit = unit
        self.filepath = None
        self.tb_logger = None
        
        if mode == "off":
            if path is not None or tb_logger is not None:
                raise UserWarning("Path and TensorBoard logger should not be set for 'off' mode. Ignoring.")
        else:
            assert 'tb' in mode or 'file' in mode, "Invalid mode. Use 'off', 'file', 'tb', or 'file+tb'."

        if "tb" in mode:
            if tb_logger is None:
                print("TensorBoard logger must be set for 'on' mode.")
            else:
                self.tb_logger = tb_logger
        if "file" in mode:
            if path is None:
                path = os.getcwd()
                print("Path is not provided for 'on' mode. Using current working directory.")
            self.filepath = os.path.join(path, "memory_stats.txt")
            print(f"Memory stats will be logged to {self.filepath}.")
    
    def tb_log_stats(self, step):
        """Logs GPU and CPU memory stats to TensorBoard (if provided).
        The following is logged:
        - GPU memory in use
        - GPU memory limit
        - GPU peak allocation
        - GPU peak memory in use
        - GPU arrays sum
        - CPU RSS
        - CPU VMS
        - RAM available
        """
        if self.tb_logger is not None:
            self.tb_logger.log_scalars(f"GPU_stats_{self.unit}", self.get_jax_memstats(self.unit), step=step)
            self.tb_logger.log_scalars(f"CPU_stats_{self.unit}", self.get_cpu_memstats(self.unit), step=step)
    
    def convert_to_(self, mem, unit='GB'):
        """Convert memory stats to the specified unit."""
        if unit == 'GB':
            return mem / 1024 / 1024 / 1024
        elif unit == 'MB':
            return mem / 1024 / 1024
        elif unit == 'KB':
            return mem / 1024
        else:
            raise ValueError("Invalid unit. Use 'GB', 'MB', or 'KB'.")

    def get_jax_memstats(self, unit='GB'):
        gpu_mem = None
        for device in jax.local_devices():
            gpu_mem = device.memory_stats()
            try:
                live_arrays_sum = sum(x.size for x in device.live_arrays)
            except AttributeError:
                # If the device does not have live_arrays, we can use live_buffers
                # This is a workaround for older versions of JAX
                live_arrays_sum = sum(x.size for x in device.live_buffers())
            break        
    
        if gpu_mem is None:
            gpu_mem = {
                'gpu_in_use': 0,
                'gpu_limit': 0,
                'gpu_peak_alloc': 0,
                'gpu_peak_in_use': 0,
                'gpu_arrays_sum': 0
            }
        else:
            gpu_mem = {
                'gpu_in_use': self.convert_to_(gpu_mem['bytes_in_use'], unit),
                'gpu_limit': self.convert_to_(gpu_mem['bytes_limit'], unit),
                'gpu_peak_alloc': self.convert_to_(gpu_mem['largest_alloc_size'], unit),
                'gpu_peak_in_use': self.convert_to_(gpu_mem['peak_bytes_in_use'], unit),
                'gpu_arrays_sum': self.convert_to_(live_arrays_sum, unit)
            }
        return gpu_mem

    def get_cpu_memstats(self, unit='GB'):
        # CPU Memory
        mem = self.process.memory_info()
        cpu_rss = self.convert_to_(mem.rss, unit)
        cpu_vms = self.convert_to_(mem.vms, unit)

        # System RAM
        sys_mem = psutil.virtual_memory()
        ram_available = self.convert_to_(sys_mem.available, unit)

        cpu_mem = {
            'cpu_rss': cpu_rss,
            'cpu_vms': cpu_vms,
            'ram_available': ram_available
        }
        return cpu_mem

    def get_print_stats(self, unit='GB'):
        # CPU Memory
        cpu_mem = self.get_cpu_memstats(unit)

        # JAX arrays GPU
        live_arrays_gpu = jax.live_arrays(platform='gpu')
        jax_mem_gpu = self.convert_to_(sum(x.size for x in live_arrays_gpu), unit)
        # JAX arrays CPU
        live_arrays_cpu = jax.live_arrays(platform='cpu')
        jax_mem_cpu = self.convert_to_(sum(x.size for x in live_arrays_cpu), unit)

        # Garbage collection
        num_objects = len(gc.get_objects())
        
        return {
            'cpu_rss': cpu_mem['cpu_rss'],
            'cpu_vms': cpu_mem['cpu_vms'],
            'ram_available': cpu_mem['ram_available'],
            'jax_arrays_gpu': jax_mem_gpu,
            'num_jax_arrays_gpu': len(live_arrays_gpu),
            'jax_arrays_cpu': jax_mem_cpu,
            'num_jax_arrays_cpu': len(live_arrays_cpu),
            'num_objects': num_objects,
        }

    def write_stats(self, text=None, iteration=None, with_timestamp=False):
        """Print current memory stats to the initialized file."""
        stats = self.get_print_stats(self.unit)
        to_print = ''
        if with_timestamp:
            to_print += datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' | '
        if iteration is not None:
            to_print += str(iteration) + ' : '
        if text is not None:
            to_print += text + '\n'

        to_print += (
            f"CPU_RSS: {stats['cpu_rss']: 3.3f} {self.unit} | "
            f"CPU_MS: {stats['cpu_vms']: 3.3f} {self.unit} | "
            f"RAM_free: {stats['ram_available']: 3.3f} {self.unit} ||| "
            f"JAX_mem (GPU): {stats['jax_arrays_gpu']: 3.3f} {self.unit} | "
            f"N_JAX_Arr (GPU): {stats['num_jax_arrays_gpu']: 6d} | "
            f"JAX_mem (CPU): {stats['jax_arrays_cpu']: 3.3f} {self.unit} | "
            f"N_JAX_Arr (CPU): {stats['num_jax_arrays_cpu']: 6d} | "
            f"N_Objs: {stats['num_objects']: 6d}"
        )
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(to_print + '\n')
    
    def log_stats(self, text=None, iteration=None):
        if self.filepath is not None:
            self.write_stats(text=text, iteration=iteration, with_timestamp=True)
        if self.tb_logger is not None:
            self.tb_log_stats(iteration)


class MemoryTracer:
    """Tracks both general Python and NumPy-specific memory allocations."""
    
    def __init__(self, log_dir='.', on=False):
        self.log_file = os.path.join(log_dir, "memory_trace.log")
        self.on = on
        if self.on:
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
        if not self.on:
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