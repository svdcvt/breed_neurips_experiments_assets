import os
import numpy as np
import argparse


def merge_npy_files(directory_path, output_file="all_trajectories.npy"):
    # Get a sorted list of all files matching the pattern sim{number}.npy
    files = sorted(
        [
            f
            for f in os.listdir(directory_path)
            if f.startswith("sim") and f.endswith(".npy")
        ]
    )

    if not files:
        print("No files found matching the pattern 'sim{number}.npy'")
        return

    print(f"Found {len(files)} files to merge")

    # Determine the total size of the merged array
    file_path = os.path.join(directory_path, files[0])
    array = np.load(file_path, mmap_mode="r", allow_pickle=False)
    total_rows = array.shape[0] * len(files)  # Total number of rows in the merged array
    sample_shape = array.shape[1:]  # Shape of a single sample

    output_path = os.path.join(directory_path, output_file)
    merged_array = np.empty((total_rows, *sample_shape), dtype=array.dtype)

    # Copy data from each file into the memory-mapped array
    current_index = 0
    for file in files:
        print(f"Reading file {file}...")
        file_path = os.path.join(directory_path, file)
        array = np.load(file_path, allow_pickle=False, mmap_mode="r")
        rows = array.shape[0]
        merged_array[current_index : current_index + rows] = array
        current_index += rows
        # Delete the read file
        os.remove(file_path)

    np.save(output_path, merged_array, allow_pickle=False)
    # Flush changes to disk
    print(
        f"Merged file saved to {output_path}, shape {merged_array.shape} and dtype {merged_array.dtype}\
         \nThe size is {merged_array.nbytes / 1024 / 1024 / 1024 :.2f} GB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge .npy files in a directory.")
    parser.add_argument(
        "directory",
        type=str,
        default="./",
        help="Directory containing the .npy files to merge.",
    )
    args = parser.parse_args()

    merge_npy_files(args.directory)
