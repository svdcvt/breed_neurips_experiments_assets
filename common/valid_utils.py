import logging
import os
import os.path as osp
import numpy as np


logger = logging.getLogger("melissa")


class TrajectoryDataset:
    """Loads trajectories stored in numpy files, specific for validation."""

    def __init__(
        self,
        seed,
        data_dir,
        output_shape,
        solver_time_steps,
        nb_time_steps,
        only_trajectory=False,
        rollout_size=-1
    ):
        np.random.seed(seed)
        self.data_dir = data_dir
        self.output_shape = output_shape
        self.nb_time_steps = nb_time_steps
        self.offset = solver_time_steps // nb_time_steps
        logger.info(f"Validation will be set to t -> t + {self.offset}*dt")

        self.only_trajectory = only_trajectory
        self.rollout_size = rollout_size

        self.file_list = [
            f for f in os.listdir(data_dir)
            if f.endswith(".npy") and "sim" in f
        ]

        self.trajectory_size = self.nb_time_steps - self.offset
        self.time_step_ids = np.tile(
            np.arange(0, self.nb_time_steps - self.offset),
            (len(self.file_list), 1)
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            idx = [idx]

        trajectories = []
        prepos_data = []
        pos_data = []

        for i in idx:
            path = osp.join(self.data_dir, self.file_list[i])
            trajectory = np.load(
                path,
                mmap_mode="r"
            ).astype(np.float32)

            if self.only_trajectory:
                trajectories.append(trajectory[:self.rollout_size + 1])
            else:
                prepos_data.append(trajectory[self.time_step_ids[i]])
                pos_data.append(
                    trajectory[self.time_step_ids[i] + self.offset]
                )

        sim_ids = np.array(idx)
        if self.only_trajectory:
            trajectories = np.array(trajectories)
            if trajectories.shape[0] == 1:
                trajectories = trajectories.squeeze(axis=0)
            return trajectories, sim_ids

        u_prev = np.array(prepos_data)
        u_next = np.array(pos_data)
        if u_prev.shape[0] == 1:
            u_prev = u_prev.squeeze(axis=0)
            u_next = u_next.squeeze(axis=0)

        return u_prev, u_next, sim_ids


def basic_batch_generator(dataset, batch_size):

    n = len(dataset)
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        yield dataset[i:j]


def load_validation_data(validation_dir,
                         seed,
                         valid_batch_size,
                         nb_time_steps,
                         output_shape,
                         only_trajectories_dataset=False,
                         rollout_size=-1):
    if validation_dir is None:
        return None, None, None

    valid_dataset = None
    valid_dataloader = None
    valid_parameters = None
    validation_dir = osp.expandvars(validation_dir)

    if osp.exists(validation_dir):
        if only_trajectories_dataset:
            return TrajectoryDataset(
                seed=seed,
                data_dir=validation_dir,
                output_shape=output_shape,
                solver_time_steps=nb_time_steps,
                nb_time_steps=nb_time_steps,
                only_trajectory=True,
                rollout_size=rollout_size
            )

        valid_dataset = TrajectoryDataset(
            seed=seed,
            data_dir=validation_dir,
            output_shape=output_shape,
            solver_time_steps=nb_time_steps,
            nb_time_steps=nb_time_steps,
            only_trajectory=False
        )

        valid_dataloader = basic_batch_generator(
            valid_dataset,
            valid_batch_size
        )

        params_path = osp.join(validation_dir, "input_parameters.npy")
        if osp.exists(params_path):
            valid_parameters = np.load(params_path)
        logger.info("Validation set loaded.")
    else:
        logger.warning("Validation set not found. "
                       "Please set validation_directory in configuration.")

    return valid_dataset, valid_dataloader, valid_parameters
