from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64,random_si64
from std.memory import Pointer, ArcPointer, alloc, Layout
from tree import tree_node, build_tree
from std.math import sqrt
from spatial import spatial_maker, extract_octant_bits, ParticleKey
from std.os import abort
from std.python.bindings import PythonModuleBuilder

struct Simulation:
    # State variables
    var n: Int
    var c1: Float32
    var t: Float32
    var np: PythonObject
    var pos: List[SIMD[DType.int32, 4]]
    var vel: List[SIMD[DType.float32, 4]]
    var acc: List[SIMD[DType.float32, 4]]

    # Physics coefficients / configuration parameters
    var epsilon: Float32
    var g: Float32
    var theta: Float32
    var interaction_range: Float32
    var radius: Float32
    var cof_of_restitution: Float32
    var dt: Float32

    def __init__(
        out self,
        n: Int,
        c1: Float32,
        epsilon: Float32 = 400.0,
        g: Float32 = 2147483647.0,
        theta: Float32 = 0.04,
        interaction_range: Int32 = 0x00000FFF,
        radius: Float32 = 0.001,
        cof_of_restitution: Float32 = 0.5,
        dt: Float32 = 0.4
    ) raises:
        seed()
        self.n = n
        self.c1 = c1
        self.t = 0.0

        # Assign configuration parameters
        self.epsilon = epsilon
        self.g = g
        self.theta = theta
        self.interaction_range = Float32(interaction_range)
        self.radius = radius
        self.cof_of_restitution = cof_of_restitution
        self.dt = dt

        # Positions
        self.pos = create_pos_vector_int(n)

        # Velocities
        try:
            self.np = Python.import_module("numpy")
        except:
            print("numpy doesn't work you stupid")
            self.np = PythonObject()
        var vel_initial = c1 * (self.np.random.rand(n, 3) - 0.5)
        self.vel = create_vel_vector_float(n, vel_initial)

        # Accelerations
        self.acc = List[SIMD[DType.float32, 4]]()
        self.acc.resize(
            n,
            SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
        )

    def step(mut self):
        self.run_timestep()

    def run_timestep(mut self):
        var keys = spatial_maker(self.pos)
        var tree = build_tree(keys)
        var interaction_pairs = List[Tuple[Int, Int]]()

        # Updating accelerations
        for i in range(len(keys)):
            var p_vec = keys[i].vector # Note index 3 is the id
            var id = Int(p_vec[3])
            self.acc[id] = self.compute_interactions(tree, 0, i, keys, interaction_pairs)

        # Updating velocities
        for i in range(len(self.vel)):
            self.vel[i] += self.dt * self.acc[i]

        # Updating velocities from collisions
        for pair in interaction_pairs:
            var result = self.handle_interaction(self.pos[pair[0]], self.pos[pair[1]], self.vel[pair[0]], self.vel[pair[1]])
            self.vel[pair[0]] = result[0]
            self.vel[pair[1]] = result[1]

        # Updating positions and time
        self.t += self.dt
        for i in range(len(self.pos)):
            self.pos[i] += (self.dt * self.vel[i]).cast[DType.int32]()

    def calculate_gravitational_acceleration(
        self,
        m1: Float32,
        pos1: SIMD[DType.int32, 4],
        pos2: SIMD[DType.int32, 4],
        ) -> SIMD[DType.float32, 4]:
        var mask = SIMD[DType.int32, 4](1, 1, 1, 0)
        var r_unmask_vec = pos2 - pos1
        var r_vec = (r_unmask_vec * mask).cast[DType.float32]()
        var r = sqrt((r_vec * r_vec).reduce_add()) + self.epsilon
        var inv_re = 1.0 / r
        var direction = r_vec * inv_re
        var scale = (self.g * inv_re) * (m1 * inv_re)
        var acceleration = scale * direction
        return acceleration

    def compute_interactions(
        self,
        mut tree: List[tree_node],
        node_index: Int,
        particle_id_keys: Int,
        keys: List[ParticleKey],
        mut interaction_pairs: List[Tuple[Int, Int]]
        ) -> SIMD[DType.float32, 4]:
        var particle_position = keys[particle_id_keys].vector
        var particle_id = Int(particle_position[3])
        var s = Float32(0xFFFFFFFF >> tree[node_index].level)
        var d = self.norm(tree[node_index].center, particle_position)

        if (s / d) < self.theta:
            return self.calculate_gravitational_acceleration(
                tree[node_index].total_mass,
                particle_position,
                tree[node_index].center_of_mass
            )
        else:
            if d < self.interaction_range:
                for i in range(tree[node_index].number_of_particles):
                    var p2_id = Int(keys[tree[node_index].index_start + i].vector[3])
                    if p2_id < particle_id:
                        interaction_pairs.append((p2_id, particle_id))

            var acc_sum = SIMD[DType.float32, 4](0, 0, 0, 0)
            if tree[node_index].child_initialized == True:
                for child in tree[node_index].children:
                    acc_sum += self.compute_interactions(tree, child, particle_id_keys, keys, interaction_pairs)
            else:
                for i in range(tree[node_index].number_of_particles):
                    if Int(tree[node_index].index_start + i) != particle_id_keys:
                        acc_sum += self.calculate_gravitational_acceleration(
                            1.0,
                            particle_position,
                            keys[tree[node_index].index_start + i].vector
                        )
            return acc_sum

    def handle_interaction(
        self,
        r1: SIMD[DType.int32, 4],
        r2: SIMD[DType.int32, 4],
        v1: SIMD[DType.float32, 4],
        v2: SIMD[DType.float32, 4]
        ) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4]]:
        var maskint = SIMD[DType.int32, 4](1, 1, 1, 0)
        var masked_r1 = r1 * maskint
        var masked_r2 = r2 * maskint
        var dr = (masked_r2 - masked_r1).cast[DType.float32]()
        var dv = v2 - v1
        var dvdotdr = (dv * dr).reduce_add()
        var dv_mag_sq = (dv * dv).reduce_add()

        if dv_mag_sq == 0.0:
            return (v1, v2)

        var t_min = dvdotdr / dv_mag_sq
        if (t_min < self.dt) & (t_min > 0.0):
            var min_r = dr + dv * t_min
            var min_distance = sqrt((min_r * min_r).reduce_add())
            if min_distance < self.radius:
                var new_v1 = (v1 + v2) / 2.0 + self.cof_of_restitution * dv
                var new_v2 = (v1 + v2) / 2.0 - self.cof_of_restitution * dv
                return (new_v1, new_v2)
            else:
                return (v1, v2)
        else:
            return (v1, v2)

    def get_positions(self) raises -> PythonObject:
        var result = self.np.empty([self.n, 3], dtype=self.np.int32)
        for i in range(self.n):
            result[i, 0] = self.pos[i][0]
            result[i, 1] = self.pos[i][1]
            result[i, 2] = self.pos[i][2]
        return result

    def get_velocity(self) raises -> PythonObject:
        var result = self.np.empty([self.n, 3], dtype=self.np.float32)
        for i in range(self.n):
            result[i, 0] = self.vel[i][0]
            result[i, 1] = self.vel[i][1]
            result[i, 2] = self.vel[i][2]
        return result

    @staticmethod
    def calculate_center_mass(pos: List[SIMD[DType.int32, 4]], mass: List[Float32]) -> SIMD[DType.float32, 4]:
        var total_moment = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
        var total_mass: Float32 = 0.0
        for i in range(len(pos)):
            var p_float = pos[i].cast[DType.float32]()
            var m = mass[i]
            total_moment += p_float * m
            total_mass += m
        if total_mass == 0.0:
            return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
        return total_moment / total_mass

    @staticmethod
    def norm(r1: SIMD[DType.int32, 4], r2: SIMD[DType.int32, 4]) -> Float32:
        var mask = SIMD[DType.int32, 4](1, 1, 1, 0)
        var masked_r1 = r1 * mask
        var masked_r2 = r2 * mask
        var f1 = masked_r1.cast[DType.float32]()
        var f2 = masked_r2.cast[DType.float32]()
        var diff = f1 - f2
        var distsq = (diff * diff).reduce_add()
        return sqrt(distsq)

    @staticmethod
    def norm(r1: SIMD[DType.float32, 4], r2: SIMD[DType.float32, 4]) -> Float32:
        var diff = r1 - r2
        var distsq = (diff * diff).reduce_add()
        return sqrt(distsq)

def create_vel_vector_float(n: Int, py_array: PythonObject) raises -> List[SIMD[DType.float32, 4]]:
    """Helper function to convert a NumPy array into a Mojo SIMD vector list."""
    var vec = List[SIMD[DType.float32, 4]]()
    vec.resize(n, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))

    for i in range(n):
        vec[i] = SIMD[DType.float32, 4](
            Float32(py=py_array[i][0]),
            Float32(py=py_array[i][1]),
            Float32(py=py_array[i][2]),
            0.0
        )
    return vec^

def create_pos_vector_int(n: Int) -> List[SIMD[DType.int32, 4]]:
    """Helper function to generate a Mojo SIMD vector list with random uint32 values."""
    # We need 4 values for each SIMD vector (3 random + 1 zero)
    var total_values = n * 4
    var raw_data = List[Int32](capacity=total_values)
    raw_data.resize(total_values, 0)

    # Fill the list with random Int32 values within the specified bounds
    for i in range(total_values):
        raw_data[i] = Int32(random_si64(-1073741824, 1073741824))

    var vec = List[SIMD[DType.int32, 4]]()
    vec.resize(n, SIMD[DType.int32, 4](0, 0, 0, 0))

    for i in range(n):
        var base = i * 4
        vec[i] = SIMD[DType.int32, 4](
            raw_data[base],
            raw_data[base + 1],
            raw_data[base + 2],
            0
        )
        vec[i][3] = Int32(i)

    return vec^

# --- PYTHON MODULE BUILDER INTEGRATION ---

@export
def PyInit_sim() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("sim")
        m.def_function[create_simulation]("create_simulation")
        m.def_function[step]("step")
        m.def_function[get_positions]("get_positions")
        m.def_function[destroy_simulation]("destroy_simulation")
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))

def create_simulation(args: PythonObject, kwargs: PythonObject) raises -> PythonObject:
    # Ensure all required positional arguments are provided
    if len(args) < 9:
        raise Error("create_simulation requires 9 positional arguments: n, c1, epsilon, g, theta, interaction_range, radius, cof_of_restitution, dt")

    var n = Int(py=args[0])
    var c1 = Float32(py=args[1])
    var epsilon = Float32(py=args[2])
    var g = Float32(py=args[3])
    var theta = Float32(py=args[4])
    var interaction_range = Int32(py=args[5])
    var radius = Float32(py=args[6])
    var cof_of_restitution = Float32(py=args[7])
    var dt = Float32(py=args[8])

    # Allocate memory, leak ownership so it persists for Python, then write to it
    var allocation = alloc(Layout[Simulation].single())
    var ptr = allocation^.unsafe_leak()
    ptr.unsafe_write(
        Simulation(
            n=n, c1=c1, epsilon=epsilon, g=g, theta=theta,
            interaction_range=interaction_range, radius=radius,
            cof_of_restitution=cof_of_restitution, dt=dt
        )
    )
    return PythonObject(Int(ptr))

def step(args: PythonObject) raises -> PythonObject:
    var handle = Int(py=args[0])
    var ptr = Pointer[Simulation, MutUntrackedOrigin](unsafe_from_address=handle)
    ptr[].step()
    return PythonObject()

def get_positions(args: PythonObject) raises -> PythonObject:
    var handle = Int(py=args[0])
    var ptr = Pointer[Simulation, MutUntrackedOrigin](unsafe_from_address=handle)
    return ptr[].get_positions()

def destroy_simulation(args: PythonObject) raises -> PythonObject:
    var handle = Int(py=args[0])
    var ptr = Pointer[Simulation, MutUntrackedOrigin](unsafe_from_address=handle)

    # Safely deinitialize the struct and free the heap memory
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()

    return PythonObject()
