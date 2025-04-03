import logging
import os
import os.path as osp
import numpy as np


logger = logging.getLogger("melissa")


class AutoregressiveTrajectoryDataset:
    """Loads trajectories stored in numpy files, specific for validation.

    Method `get_pairs` :
        Returns necessary time steps data for the given batch range.
    Method `get_rollout` :
        Returns the rollout timesteps for the given batch range of trajectory samples.
        """

    def __init__(
        self,
        data_dir: str,
        output_shape: tuple[int, ...],
        nb_time_steps: int
    ) -> None:
        self.nb_time_steps: int = nb_time_steps

        file_list: list[str] = [
            f for f in sorted(os.listdir(data_dir))
            if f.endswith(".npy") and "sim" in f
        ][:15]
        self.num_samples: int = len(file_list)
        self.num_pairs: int = self.num_samples * (self.nb_time_steps - 1)

        # reading dataset to CPU memory
        self.validation_dataset: np.ndarray = np.empty(
            (len(file_list) * self.nb_time_steps, *output_shape), dtype=np.float32
        )
        logger.info(f"Loading validation dataset from {data_dir}...")
        for i, f in enumerate(file_list):
            self.validation_dataset[i * self.nb_time_steps: (i + 1) * self.nb_time_steps] = \
                np.load(
                osp.join(data_dir, f),
                mmap_mode="r"
                )
            logger.info(f"Loaded {f}.")
        logger.info("Validation dataset loaded.")

    def __len__(self):
        return len(self.validation_dataset)
    
    def __getitem__(self, idx: int | list | slice) -> np.ndarray:
        return self.validation_dataset[idx]

    def get_pairs(self, batch_l: int, batch_r: int) -> tuple[list[int], list[int], list[int]]:
        '''
        Returns necessary time steps data for the given batch range.
        This function is optimized, as it does not load timesteps twice for input and ouput.
        For large datasets, instead of loading a whole trajectory for each simulation sample, it treats
        time steps pairs as an item of a batch and loads only necessary time steps for the given batch range.
        Use as follows: 
        ```
        u_prev_indices, u_next_indices = dataset.get_pairs(i, i + BATCH_SIZE)
        u_prev = dataset[u_prev_indices].to(device)
        u_pred = model(u_prev).detach().cpu()
        valid_loss = error(u_pred, dataset[u_next_indices])
        ```
        Returns:
            - indices of the previous time step
            - indices of the next time step
            - simulation indices
        '''
        def convert_to_array_index(batch_id):
            return min(batch_id + (batch_id // (self.nb_time_steps - 1)), self.num_pairs - 2)
        
        u_prev_indices: list[int] = []
        u_next_indices: list[int] = []
        for i in range(convert_to_array_index(batch_l), convert_to_array_index(batch_r) + 1):
            if not i % (self.nb_time_steps - 1) == 0:
                u_prev_indices.append(i)
                u_next_indices.append(i + 1)

        return u_prev_indices, u_next_indices
    
    def get_rollout(self, batch_l: int, batch_r: int, rollout_size: int | None = None) -> tuple[list[int], list[list[int]]]:
        '''
        Returns the rollout timesteps for the given batch range of trajectory samples.
        Use as follows:
        ```
        ic_idx, rollout_idx = dataset.get_rollout(i, i + BATCH_SIZE, rollout_size=ROLL_SIZE)
        u_step = dataset[ic_idx].to(device)
        rollout_losses = []
        for roll_idx in rollout_idx:
            u_step = model(u_step)
            rollout_losses.append(error(u_step.detach().cpu(), dataset[roll_idx]))
        ```
        This way avoids unnecessary memory usage by predicting the rollout step by step instead of the whole rollout at once.
        '''

        if rollout_size is None:
            rollout_size = self.nb_time_steps
        assert 0 < rollout_size < self.nb_time_steps, f"Rollout size must be at least 1 and not more than {self.nb_time_steps - 1}, but got {rollout_size}."
        assert batch_l >= 0, f"Rollout index must be non negative {batch_l} < 0."
        assert batch_r <= self.num_samples, f"Rollout index must be not more than the number of samples, but got {batch_r} > {self.num_samples}."
        
        ic_idx = [i * self.nb_time_steps for i in range(batch_l, batch_r)]
        rollout_idx = []
        for r in range(1, rollout_size + 1):
            rollout_idx.append([i + r for i in ic_idx])
        return ic_idx, rollout_idx

def batch_generator(dataset, batch_size, rollout_size=-1):
    if dataset is None:
        return None
    n = dataset.num_pairs if rollout_size == -1 else dataset.num_samples
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        if rollout_size == -1:
            yield dataset.get_pairs(i, j)
        else:
            yield dataset.get_rollout(i, j, rollout_size)


def load_validation_dataset(validation_dir,
                            nb_time_steps,
                            output_shape):
    if validation_dir is None:
        return None, None

    valid_dataset = None
    valid_parameters = None
    validation_dir = osp.expandvars(validation_dir)

    if osp.exists(validation_dir):
        valid_dataset = AutoregressiveTrajectoryDataset(
            data_dir=validation_dir,
            output_shape=output_shape,
            nb_time_steps=nb_time_steps
        )

        params_path = osp.join(validation_dir, "input_parameters.npy")
        if osp.exists(params_path):
            valid_parameters = np.load(params_path)
        logger.info("Validation set loaded.")
    else:
        logger.warning("Validation set not found. "
                       "Please set validation_directory in configuration.")

    return valid_dataset, valid_parameters