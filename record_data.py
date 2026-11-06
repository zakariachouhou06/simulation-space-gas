import csv
import json
import numpy as np
from simulation import Simulation

ENERGY_EVERY = 10    # energy recorded every this many steps (positions: every step)
FLUSH_EVERY = 1000   # force data onto disk this often, so a crash loses little

sim = Simulation()
step = getattr(sim, "step", sim.next_frame)  # fast step() if you added it to simulation.py
dt = float(sim.dt)


def get_pos():
    return np.asarray(sim.get_positions()).reshape(-1, 3).astype(np.int32)


def get_energy():
    return np.atleast_1d(np.asarray(sim.energy(), dtype=np.float64)).ravel()


first = get_pos()
n = first.shape[0]
with open("meta.json", "w") as fh:
    json.dump({"n": n, "dt": dt}, fh)

pos_f = open("pos.bin", "wb")                 # raw int32, appended every step
en_f = open("energy.csv", "w", newline="")
w = csv.writer(en_f)

e0 = get_energy()
w.writerow(["t"] + [f"E{i}" for i in range(len(e0))])
w.writerow([0.0, *e0])
first.tofile(pos_f)

k = 0
try:
    while True:
        step()
        k += 1
        get_pos().tofile(pos_f)
        if k % ENERGY_EVERY == 0:
            w.writerow([k * dt, *get_energy()])
        if k % FLUSH_EVERY == 0:
            pos_f.flush()
            en_f.flush()
        if k % 5000 == 0:
            print(k, "steps, t =", k * dt)
except KeyboardInterrupt:
    print("stopped")
    
finally:
    pos_f.close()
    en_f.close()
    sim.close()
    with open("meta.json", "w") as fh:
        json.dump({"n": n, "dt": dt, "steps": k, "total_time": k * dt}, fh)
    print("saved", k + 1, "position frames to pos.bin, energy to energy.csv")
    print("simulated time:", k * dt, "s  (", k, "steps x dt =", dt, ")")