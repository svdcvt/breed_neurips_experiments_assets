# # optax.schedules.warmup_cosine_decay_schedule(init_value: float, peak_value: float, warmup_steps: int, decay_steps: int, end_value: float = 0.0, exponent: float = 1.0) → base.Schedule
# [source]

# Linear warmup followed by cosine decay.

# Parameters:

#         init_value – Initial value for the scalar to be annealed.

#         peak_value – Peak value for scalar to be annealed at end of warmup.

#         warmup_steps – Positive integer, the length of the linear warmup.

#         decay_steps – Positive integer, the total length of the schedule. Note that this includes the warmup time, so the number of steps during which cosine annealing is applied is decay_steps - warmup_steps.

#         end_value – End value of the scalar to be annealed.

#         exponent – The default decay is 0.5 * (1 + cos(pi t/T)), where t is the current timestep and T is decay_steps. The exponent modifies this to be (0.5 * (1 + cos(pi * t/T))) ** exponent. Defaults to 1.0.

# Returns:

#     schedule

#         A function that maps step counts to values

import numpy as np
import matplotlib.pyplot as plt

# Assuming the warmup_cosine_decay_schedule function is implemented
def warmup_cosine_decay_schedule(init_value, peak_value, warmup_steps, decay_steps, end_value=0.0, exponent=1.0):
    def schedule(step):
        if step < warmup_steps:
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        elif step < decay_steps:
            t = (step - warmup_steps) / (decay_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * t))
            return (peak_value - end_value) * (cosine_decay ** exponent) + end_value
        else:
            return end_value
    return schedule

# Define parameters for the schedule
init_value = 1.0e-3
peak_value = 1.0e-4
warmup_steps = 5000
decay_steps = 20000
end_value = 0.0
exponent = 1.0

# Generate the schedule
schedule = warmup_cosine_decay_schedule(init_value, peak_value, warmup_steps, decay_steps, end_value, exponent)

# Generate steps and corresponding values
steps = np.arange(0, 10000)
values = [schedule(step) for step in steps]

# Plot the schedule
plt.figure(figsize=(10, 6))
plt.plot(steps, values, label="Learning Rate Schedule")
plt.xscale("linear")
plt.yscale("log")
# plt.yscale("linear")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.title("Warmup Cosine Decay Learning Rate Schedule")
plt.legend()
plt.grid(True)
plt.savefig("warmup_cosine_decay_schedule.png")
