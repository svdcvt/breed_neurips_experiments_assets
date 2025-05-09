import numpy as np
import pandas as pd
import os
import glob


def get_metaids(length, num):
    meta_ids = np.linspace(0, length - 1, num - 2, dtype=int, endpoint=True)
    meta_ids = [meta_ids[0], meta_ids[0] + 10, *meta_ids[1:-1], meta_ids[-1] - 10, meta_ids[-1]]
    return meta_ids

main_directory = "/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/"
add_file_path = "diff_*/trajectories/all_trajectories.npy"
NUM = 5
validation_file_path = os.path.join(main_directory, add_file_path)
all_files = glob.glob(validation_file_path)
if len(all_files) == 0:
    raise FileNotFoundError(f"File not found: {validation_file_path}")

all_ids_diffs = []

for datapath in all_files:
    if os.path.exists(datapath):
        validation_dataset = np.load(datapath, allow_pickle=False)
        name = os.path.relpath(datapath, main_directory).split("/")[0]
        print(name)
        # shape is (NUM*nb_timesteps (101 or 201), mesh size)
        if validation_dataset.shape[0] > (1500 * 101):
            nb_time_steps = 201
        else:
            nb_time_steps = 101
        num_samples = validation_dataset.shape[0] // nb_time_steps
        validation_dataset = validation_dataset.reshape(num_samples, nb_time_steps, -1)
        print("bad values:", (~np.isfinite(validation_dataset)).sum())
        print("percentage of bad values:", (~np.isfinite(validation_dataset)).sum() / validation_dataset.size)
        # drop rows with bad values
        difference = validation_dataset[:, 1:] - validation_dataset[:, :-1]
        diff_std = np.nanstd(difference, axis=-1)
        diff_std_std = np.nanstd(diff_std, axis=-1)
        diff_std_mean = np.nanmean(diff_std, axis=-1)
        chosen_ids = np.argsort(diff_std_std)[get_metaids(num_samples, NUM)]
        assert len(chosen_ids) == NUM
        row = [name, ] + [int(i) for i in chosen_ids] + [float(d) for d in diff_std_std[chosen_ids]]
        assert len(row) == 2 * NUM + 1
        # print(row)
        all_ids_diffs.append(row)
    else:
        print(f"File not found: {datapath}")
        continue
# Create a DataFrame from the list of lists
df = pd.DataFrame(all_ids_diffs)
# Save the DataFrame to a CSV file
output_file = os.path.join(main_directory, "validation_ids.csv")
df.to_csv(output_file, index=False)
print(f"Validation IDs and differences saved to {output_file}")
