import gc
import psutil
import jax
from datetime import datetime


class MemoryMonitor:
    def __init__(self, path='.'):
        self.process = psutil.Process()
        self.file = f"{path}/memory_stats.txt"

    def get_stats(self):
        # CPU Memory
        mem = self.process.memory_info()
        cpu_rss = mem.rss / 1024 / 1024 / 1024
        cpu_vms = mem.vms / 1024 / 1024 / 1024

        # System RAM
        sys_mem = psutil.virtual_memory()
        ram_available = sys_mem.available / 1024 / 1024 / 1024

        # JAX arrays
        live_arrays = jax.live_arrays()
        jax_mem = sum(x.size for x in live_arrays) / 1024 / 1024 / 1024

        # Garbage collection
        num_objects = len(gc.get_objects())

        return {
            'cpu_rss': cpu_rss,
            'cpu_vms': cpu_vms,
            'ram_available': ram_available,
            'jax_arrays': jax_mem,
            'num_jax_arrays': len(live_arrays),
            'num_objects': num_objects,
        }

    def print_stats(self, text=None, iteration=None, with_timestamp=False):
        """Print current memory stats to the initialized file."""
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
            f"JAX_mem: {stats['jax_arrays']: 3.3f} Gb | "
            f"N_JAX_Arr: {stats['num_jax_arrays']: 6d} | "
            f"N_Objs: {stats['num_objects']: 6d}"
        )
        with open(self.file, 'a', encoding='utf-8') as f:
            f.write(to_print + '\n')
