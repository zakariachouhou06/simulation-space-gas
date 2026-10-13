
import mojo.importer
import sim
import numpy as np

class Simulation:
    def __init__(
        self,
        n = 3000,
        c1 = 0,
        epsilon = 500000,
        g = 0.003*2147483647.0**3,
        theta = 0.04,
        interaction_range_agr = 0.9,
        radius = 1000.0,
        cof_of_restitution = 0.5,
        dt = 0.000004
    ):
        # Pack parameters into the tuple/dict structure expected by Mojo
        args = (n, c1, epsilon, g, theta, interaction_range_agr, radius, cof_of_restitution, dt)
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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
