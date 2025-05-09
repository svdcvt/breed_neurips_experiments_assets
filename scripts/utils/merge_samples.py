import os
import numpy as np
import argparse


def merge_npy_files(directory_path, output_file="all_trajectories.npy", test=False, quiet=False):
    # Get a sorted list of all files matching the pattern sim{number}.npy
    
    if not directory_path.rstrip('/').endswith("trajectories"):
        print(f"Directory {directory_path} does not end with 'trajectories', adding it.")
        directory_path = os.path.join(directory_path, "trajectories")
    files = sorted(
        [
            f
            for f in os.listdir(directory_path)
            if f.startswith("sim") and f.endswith(".npy")
        ],
        key=lambda x: int(x[3:-4])  # Extract the number from the filename
    )

    input_parameters_path = os.path.join(directory_path, "input_parameters.npy")
    if os.path.exists(input_parameters_path):
        print(f"Input parameters file found: {input_parameters_path}")
        input_parameters = np.load(input_parameters_path, mmap_mode="r", allow_pickle=False)
        print(f"Input parameters shape: {input_parameters.shape}")

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
    for i, file in enumerate(files):
        if not quiet:
            print(f"Reading file {file}...")
        file_path = os.path.join(directory_path, file)
        array = np.load(file_path, allow_pickle=False, mmap_mode="r")
        rows = array.shape[0]
        if not test:
            merged_array[current_index : current_index + rows] = array
            current_index += rows
            if not quiet:
                print(f"Deleting file {file_path}...")
            os.remove(file_path)
        else:
            try:
                merged_array[current_index : current_index + rows] = array
                current_index += rows
            except BaseException as e:
                print(f"Error while merging file {file_path}: {e}")
                return 1
            print(f"Test mode: File {file_path} would be deleted.")
    
    if not test:
        np.save(output_path, merged_array, allow_pickle=False)
        # Flush changes to disk
        print(
            f"Merged file saved to {output_path}, shape {merged_array.shape} and dtype {merged_array.dtype}\
            \nThe size is {merged_array.nbytes / 1024 / 1024 / 1024 :.2f} GB"
        )
    else:
        print(
            f"Test mode: Merged file would be saved to {output_path}, shape {merged_array.shape} and dtype {merged_array.dtype}\
            \nThe size is {merged_array.nbytes / 1024 / 1024 / 1024 :.2f} GB"
        )
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge .npy files in a directory.")
    parser.add_argument(
        "directory",
        type=str,
        default="./",
        help="Directory containing the .npy files to merge.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode, do not delete files after merging.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output messages.",
    )
    args = parser.parse_args()

    exit_code = merge_npy_files(args.directory, test=True, quiet=args.quiet)
    if exit_code == 1:
        print("An error occurred during the merging process.")
    elif exit_code == 0:
        if not args.test:
            print("Test mode: No errors. Starting the merging process.")
            merge_npy_files(args.directory, test=False, quiet=args.quiet)
        else:
            print("Test mode: No errors. No merging performed.")
