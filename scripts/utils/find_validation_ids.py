import numpy as np

validation_file_path = "/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/ks_cons_1d/800_3waves_easy/trajectories/all_trajectories.npy"
validation_dataset = np.load(validation_file_path, allow_pickle=False)
# shape is (NUM*nb_timesteps (101), mesh size)
nb_time_steps = 101
num_samples = validation_dataset.shape[0] // nb_time_steps
print(validation_dataset.shape, num_samples)
diff_stds = []
for i in range(num_samples):
    sample = validation_dataset[i * nb_time_steps: (i + 1) * nb_time_steps]
    diff = (sample[1:] - sample[:-1])
    diff_std = np.std(np.std(diff, axis=-1))
    diff_stds.append(diff_std)

sort_ids = np.argsort(diff_stds)
NUM = 10
# i want 5 ids across the range of diff_stds
# should definitely include the max and min
meta_ids = np.linspace(0, len(sort_ids) - 1, NUM - 2, dtype=int, endpoint=True)
meta_ids = [meta_ids[0], meta_ids[0] + 10, *meta_ids[1:-1], meta_ids[-1] - 10, meta_ids[-1]]

ids = sort_ids[meta_ids]
print("valid_ids_to_predict = [", end="")
print(*ids, sep=", ", end="")
print("]\nvalid_diffs = [", end="")
print(*(f"{diff_stds[i]:.3f}" for i in ids), sep=", ", end="")
print("]\n")
