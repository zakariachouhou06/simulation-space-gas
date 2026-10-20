
import mojo.importer
import sim
import numpy as np

class Simulation:
    def __init__(
        self,
        n = 4000,
        c1 = 1000,
        epsilon = 500000,
        g = 0.3*2147483647.0**3,
        theta = 0.004,
        interaction_range_agr = 1,
        radius = 1000000.0,
        cof_of_restitution = 0.95,
        dt = 0.0004
    ):
        # Pack parameters into the tuple/dict structure expected by Mojo
        args = (n, c1, epsilon, g, theta, interaction_range_agr, radius, cof_of_restitution, dt)
        self.radius = radius
        self.dt = dt
        kwargs = {}
        self.ptr = sim.create_simulation(args, kwargs)

    def next_frame(self):
        """Advances the simulation and returns the latest positions array."""
        # Wrap self.ptr in a tuple so Mojo receives it correctly as args[0]
        sim.step((self.ptr,))
        return sim.get_positions((self.ptr,))

    def get_positions(self):
        return sim.get_positions((self.ptr,))
    def close(self):
        """Explicitly frees the heap-allocated simulation memory in Mojo."""
        if getattr(self, "ptr", None) is not None:
            sim.destroy_simulation((self.ptr,))
            self.ptr = None

    def energy(self):
        return sim.energy((self.ptr,))

    def collision(self):
        return sim.collision((self.ptr,))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
