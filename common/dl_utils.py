import logging
import os.path as osp
import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

logger = logging.getLogger("melissa")

def loss_fn(model, x, y):
    y_pred = jax.vmap(model)(x)
    return jnp.mean(jnp.square(y_pred - y))


@eqx.filter_jit(donate='all')
def update_fn(model, optimizer, x, y, opt_state):
    loss, grads = eqx.filter_value_and_grad(
            loss_fn
        )(model, x, y)

    updates, new_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return (
        new_model,
        new_state,
        loss
    )

# def get_grads_stats(grads):
#     grads_flat, _ = jax.tree_util.tree_flatten(eqx.filter(grads, eqx.is_array))
#     flat_grads = [g.flatten() for g in grads_flat]
#     grads_concat = jnp.concatenate(flat_grads)
#     total_norm = 0.0
#     for g in flat_grads:
#         total_norm += jnp.linalg.norm(g, ord=2) ** 2
#     total_norm = jnp.sqrt(total_norm)
#     mean = jnp.mean(grads_concat)
#     variance = jnp.var(grads_concat)
#     return {
#         "l2-norm": total_norm.item(),
#         "mean": mean.item(),
#         "var": variance.item()
#     }
# 
# def loss_fn(model, x, y):
#     y_pred = jax.vmap(model)(x)
#     mse_per_sample = jax.vmap(
#         ex.metrics.MSE
#     )(y_pred, y)
#     batch_mse = jnp.mean(mse_per_sample)
#     return batch_mse, mse_per_sample
# 
# @eqx.filter_jit(donate='all')
# def update_fn(model, optimizer, x, y, opt_state):
#     eval_grad = eqx.filter_value_and_grad(
#         loss_fn,
#         has_aux=True
#     )
#     (loss, loss_per_sample), grads = eval_grad(model, x, y)
#     updates, new_state = optimizer.update(grads, opt_state, model)
#     new_model = eqx.apply_updates(model, updates)
#     return (
#         new_model,
#         new_state,
#         loss,
#         loss_per_sample
# )
# 
# def rollout_loss_fn(model, x, n=5):
#     """x is the batch of trajectories (batch/sim, tsteps, channel, *dims)"""
#     ics = x[:, 0]
#     y = x[:, 1:n+1]
#     y_pred = jax.vmap( # lax.scan
#         ex.rollout(
#             model,
#             n,
#             include_init=False,
#         )
#     )(ics)
#     mse_per_traj = jax.vmap(
#         ex.metrics.nRMSE,
#         in_axes=1
#     )(y_pred, y)
#     return jnp.mean(mse_per_traj), mse_per_traj, y_pred


class AutoregressiveTrajectoryDataset:
    """Loads trajectories stored in numpy files, specific for validation.

    Method `get_pairs` :
        Returns necessary time steps data for the given batch range.
    Method `get_rollout` :
        Returns the rollout timesteps for the given batch range of trajectory samples.
        """

    def __init__(
        self,
        data_path: str,
        nb_time_steps: int
    ) -> None:
        self.nb_time_steps = nb_time_steps
        logger.info(f"Loading validation dataset from {data_path}...")
        self.validation_dataset = np.load(data_path)#, mmap_mode="r")
        
        self.num_samples = self.validation_dataset.shape[0] // self.nb_time_steps
        self.num_pairs = self.num_samples * (self.nb_time_steps - 1)
        logger.info(f"Validation dataset loaded {self.num_samples} {self.num_pairs} {self.validation_dataset.shape}.")

    def __len__(self):
        return self.validation_dataset.shape[0]
    
    def __getitem__(self, idx):
        return np.array(self.validation_dataset[idx], dtype=np.float32, copy=False)

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
    
    def _prepare_batch_generator(self, batch_size: int, rollout_size: int):
        self.batched_indices = []
        for i in range(0, self.num_pairs, batch_size):
            j = min(self.num_pairs, i + batch_size)
            self.batched_indices.append(self.get_pairs(i, j))
        if rollout_size != -1:
            self.batched_indices_rollout = []
            for i in range(0, self.num_samples, batch_size):
                j = min(self.num_samples, i + batch_size)
                self.batched_indices_rollout.append(self.get_rollout(i, j, rollout_size))

    def batch_generator(self, rollout_size: int = -1):
        if rollout_size != -1: # want rollout batches
            return self.batched_indices_rollout
        else: # want 1to1 batches
            return self.batched_indices


def load_validation_dataset(validation_dir,
                            nb_time_steps,
                            batch_size,
                            rollout_size=-1):
    if validation_dir is None:
        return None, None

    valid_dataset = None
    valid_parameters = None
    valid_dataloader = None
    valid_dataloader_rollout = None
    validation_dir = osp.expandvars(validation_dir)
    # traj_path = osp.join(validation_dir, "all_trajectories.npy")
    traj_path = osp.join(validation_dir, "15_trajectories.npy")

    if osp.exists(traj_path):
        valid_dataset = AutoregressiveTrajectoryDataset(
            data_path=traj_path,
            nb_time_steps=nb_time_steps
        )
        valid_dataset._prepare_batch_generator(
            batch_size=batch_size,
            rollout_size=rollout_size
        )
        
        valid_dataloader = valid_dataset.batch_generator()
        if rollout_size != -1:
            valid_dataloader_rollout = valid_dataset.batch_generator(rollout_size=rollout_size)

        params_path = osp.join(validation_dir, "all_parameters.npy")
        if osp.exists(params_path):
            valid_parameters = np.load(params_path)
        logger.info(f"Validation set loaded. Size: {valid_dataset.num_samples} trajectories, {valid_dataset.num_pairs} samples, {len(valid_dataloader)} batches.")
    else:
        logger.warning("Validation set not found. "
                       "Please set validation_directory in configuration.")
    
    return valid_dataset, valid_parameters, valid_dataloader, valid_dataloader_rollout