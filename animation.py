import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulation import Simulation

sim = Simulation(n=5)

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor("black")

# --- 3D particle view ---
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_zlim(0, 1)
ax1.set_facecolor("black")
scatter = ax1.scatter([], [], [], s=20, c="white")

# --- Energy graph ---
ax2 = fig.add_subplot(122)
energy_line, = ax2.plot([], [])
ax2.set_xlabel("Frame")
ax2.set_ylabel("Total Kinetic Energy")
energy_history = []


def update(frame):
    positions = sim.next_frame()
    scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

    total_ke = 0.5 * np.sum(sim.mass_arr[:, None] * sim.vel_arr**2)
    energy_history.append(total_ke)

    energy_line.set_data(range(len(energy_history)), energy_history)
    ax2.relim()
    ax2.autoscale_view()

    return scatter, energy_line


ani = animation.FuncAnimation(
    fig, update, frames=sim.frames, interval=0.001, blit=False
)

plt.show()