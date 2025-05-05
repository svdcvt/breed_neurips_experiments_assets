import psutil
import os

def calculate_data_size(space_points, time_points, num_samples, precision=32):
    """Calculate size in bytes for given parameters"""
    bytes_per_number = 4 if precision == 32 else 8
    # Size for one sample (space points * time points * bytes per number)
    sample_size = space_points * time_points * bytes_per_number
    return sample_size * num_samples

def get_available_memory():
    """Get available system memory in bytes"""
    return psutil.virtual_memory().available

def bto(bytes_size, level=3):
    """Convert bytes to gigabytes"""
    return bytes_size / (1024 ** level)

def tob(bytes_size, level=3):
    """Convert to bytes"""
    return bytes_size * (1024 ** level)

def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
def get_next_dir(path: str) -> str:
    """Get the next directory name by incrementing the last number in the path"""
    # A/B/C/0/
    # A/B/C/1/
    # if A/B/C/ exists find the largest number and add 1
    path_ = path + "_0"
    if not os.path.exists(path_):
        print("Path is a new study")
        return path_
    else:
        print("Path is an existing study")
        i = max([int(p.split("_")[-1]) for p in os.listdir(os.path.dirname(path_)) if p.startswith(os.path.basename(path))]) + 1
        return path + "_" + str(i)
