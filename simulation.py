
import mojo.importer
import sim
import numpy as np

mass = (10**15)*1.6735575e-27
class Simulation:
    def __init__(
        self,
        n = 1000,
        c1 = 10**10,
        epsilon = 500000,
        g = 6.67430*(10**-11)*mass,
        theta = 0.0000004,
        interaction_range_agr = 1,
        radius = 7.94*(10**-11),
        cof_of_restitution = 0.9,
        dt = 0.0000004,
        spawn_radius = 1,
        side = 2
        ):

        # Pack parameters into the tuple/dict structure expected by Mojo
        args = (n, c1, epsilon, g, theta, interaction_range_agr, radius, cof_of_restitution, dt, side, spawn_radius)
        self.radius = radius
        self.dt = dt
        self.side = side
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
    
    def distance_2_to_center_of_mass(self):
        return np.asarray(sim.distance_2_to_center_of_mass((self.ptr,)))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
