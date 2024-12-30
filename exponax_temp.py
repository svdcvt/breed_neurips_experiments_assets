import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import exponax as ex


DOMAIN_EXTENT = 1.0
NUM_POINTS = 20

grid = ex.make_grid(
    1,
    DOMAIN_EXTENT,
    NUM_POINTS,
)

ic_fun = lambda x: jnp.sin(2 * jnp.pi * x / DOMAIN_EXTENT)  # noqa
ic = ic_fun(grid)


VELOCITY = 1.0
DT = 0.2

advection_stepper = ex.stepper.Advection(
    1,
    DOMAIN_EXTENT,
    NUM_POINTS,
    DT,
    velocity=VELOCITY,
)

print(advection_stepper)

full_grid = ex.make_grid(1, DOMAIN_EXTENT, NUM_POINTS, full=True)
full_ic = ex.wrap_bc(ic)

# plt.plot(full_grid[0], ex.wrap_bc(ic)[0], label="ic")
# u = ic
# for i in range(3):
#     u_next = advection_stepper(u)
#     plt.plot(full_grid[0], ex.wrap_bc(u_next)[0], label=f"{i+1} step")
#     u = u_next
# plt.xlim(-0.1, 1.1)
# plt.grid()
# plt.legend()
# plt.show()

SMALLER_DT = 0.01
slower_advection_stepper = ex.stepper.Advection(
    1, DOMAIN_EXTENT, NUM_POINTS, SMALLER_DT, velocity=VELOCITY
)
longer_rollout_advection_stepper = ex.rollout(
    slower_advection_stepper, 200, include_init=True
)
s = time.time()
longer_trajectory = longer_rollout_advection_stepper(ic)
print(f"Took {time.time() - s:.2f} seconds")

longer_trajectory_wrapped = jax.vmap(ex.wrap_bc)(longer_trajectory)
print(longer_trajectory_wrapped.shape)
plt.imshow(
    longer_trajectory_wrapped[:, 0, :].T,
    origin="lower",
    cmap="RdBu",
    vmin=-1,
    vmax=1,
    extent=[0, 200 * SMALLER_DT, 0, DOMAIN_EXTENT],
)
plt.colorbar()
plt.xlabel("time")
plt.ylabel("space")
plt.title("advection")
plt.show()

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# values = longer_trajectory_wrapped[:, 0, :].T
# time = jnp.linspace(0, 2.0, longer_trajectory_wrapped.shape[0])
# space = jnp.linspace(0, DOMAIN_EXTENT, longer_trajectory_wrapped.shape[2])
# time_grid, space_grid = jnp.meshgrid(time, space)

# surf = ax.plot_surface(
#     time_grid, space_grid, values,
#     cmap="RdBu", vmin=-1, vmax=1, edgecolor='none'
# )

# cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
# cbar.set_label('Amplitude')

# ax.set_xlabel("Time")
# ax.set_ylabel("Space")
# ax.set_zlabel("Amplitude")
# ax.set_title("Advection in 3D")

# plt.show()
