from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64,random_si64
from std.memory import Pointer, ArcPointer, alloc, Layout
from tree import tree_node, build_tree
from std.math import sqrt
from spatial import spatial_maker, extract_octant_bits, ParticleKey
from std.os import abort
from std.python.bindings import PythonModuleBuilder
from std.benchmark import run, keep
from std.time import global_perf_counter_ns
comptime int_minhalf = Int32(-1073741824)
comptime int_maxhalf = Int32(1073741823)

struct timer:
    var time: UInt64
    var last_time: UInt64
    def __init__(out self):
        self.time = global_perf_counter_ns()
        self.last_time = self.time
    def click(mut self):
        self.time = global_perf_counter_ns()
        self.last_time = self.time
    def tak(mut self,declaration:String):
        self.time = global_perf_counter_ns()
        print("{} : {}".format(declaration, Int(self.time) - Int(self.last_time)))
        self.last_time = self.time


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
    var theta_rcp_sq: Int64
    var interaction_range_agr: Float32
    var interaction_range: Int64
    var interaction_range_sq: Int64
    var radius: Float32
    var radius_sq: Float32
    var cof_of_restitution: Float32
    var dt: Float32

    def __init__(
        out self,
        n: Int,
        c1: Float32,
        epsilon: Float32 = 400.0,
        g: Float32 = Float32(0.3*2147483647.0**3),
        theta: Float32 = 0.04,
        interaction_range_agr: Float32 = 0.9, #number of the max theoretical range low = better preformance but more missed collisions
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
        self.theta_rcp_sq = Int64(1/(theta*theta))
        self.interaction_range_agr =interaction_range_agr
        self.radius = radius
        self.radius_sq = radius*radius
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

        #initialize interaction range
        self.interaction_range= Int64(self.dt* c1 * interaction_range_agr)
        self.interaction_range_sq = Int64(self.dt*self.dt*c1 * interaction_range_agr * c1 * interaction_range_agr)

        #initialize vel
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

        #make a sorted pos list keys
        var keys = spatial_maker(self.pos)
        var tree = build_tree(keys)
        #initialize
        var posible_collisions_pairs = List[Tuple[Int, Int]]()
        var collision_pairs = List[Tuple[Int, Int, SIMD[DType.float32, 4], SIMD[DType.float32, 4]]]() # intermidate step because updating the velocities for each particle is expensive
        # Updating accelerations
        for i in range(len(keys)):
            var p_vec = keys[i].vector # Note index 3 is the id
            var id = keys[i].id
            self.acc[id] = self.compute_interactions(tree, 0, i,id,p_vec, keys, posible_collisions_pairs)

        # Updating velocities
        for i in range(len(self.vel)):
            self.vel[i] += self.dt * self.acc[i]
        # Updating velocities from collisions
        for pair in posible_collisions_pairs:
            self.handle_interaction(keys[pair[0]].vector, keys[pair[1]].vector, self.vel[pair[0]], self.vel[pair[1]], collision_pairs, pair)
        #print("collision_pairs:")
        #print(len(collision_pairs))
        for collision_pair in collision_pairs:
            print("collision",collision_pair[0],collision_pair[1] )
            self.vel[collision_pair[0]]=collision_pair[2]
            self.vel[collision_pair[1]]=collision_pair[3]
        print("norm:",self.norm(self.pos[0],self.pos[1]), "radius", self.radius)
        #timer.tak("interactions")
        # Updating positions and time
        posible_collisions_pairs.clear()
        collision_pairs.clear()

        var max_val2: Float32 = 0.0

        for i in range(len(self.vel)):
            var magnitude2 = (self.vel[i]*self.vel[i]).reduce_add()
            if magnitude2 > max_val2:
                max_val2 = magnitude2

        self.interaction_range_sq = Int64(self.dt* self.dt * max_val2 * max_val2 * self.interaction_range_agr * self.interaction_range_agr)
        self.interaction_range = Int64(sqrt(self.dt* max_val2 * self.interaction_range_agr))

        self.pos_step()
        self.t += self.dt


    def calculate_gravitational_acceleration(
        self,
        m1: Float32,
        pos1: SIMD[DType.int32, 4],
        pos2: SIMD[DType.int32, 4],
        ) -> SIMD[DType.float32, 4]:
        var r_vec = (pos2 - pos1).cast[DType.float32]()
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
        particle_id: Int,
        particle_position: SIMD[DType.int32, 4],
        keys: List[ParticleKey],
        mut interaction_pairs: List[Tuple[Int, Int]]) -> SIMD[DType.float32, 4]:

        """This is a function that calculate the real stuff. For every block it checks if conditions are met and if not it breaks it self up and calls itself."""
        var s = Int64(0xFFFFFFFF >> tree[node_index].level) #with of a block
        var diff = tree[node_index].center-particle_position #difference between particle and non particle
        var d_2 = (diff.cast[DType.int64]()*diff.cast[DType.int64]()).reduce_add() # we have to cast to int 64 because it can overflow

        if s*s * self.theta_rcp_sq > d_2: #rewriting of the equation to avoid division s*s * self.theta_rcp_sq > d_2

                var acc_sum = SIMD[DType.float32, 4](0, 0, 0, 0)
                if tree[node_index].child_initialized == True:
                    for child in tree[node_index].children:
                        acc_sum += self.compute_interactions(tree, child, particle_id_keys,particle_id, particle_position, keys, interaction_pairs)
                else:
                    for i in range(tree[node_index].number_of_particles):
                        if Int(tree[node_index].index_start + i) != particle_id_keys:
                            var p2_id = Int(keys[tree[node_index].index_start + i].id)
                            if p2_id < particle_id:
                                interaction_pairs.append((p2_id, particle_id))
                            acc_sum += self.calculate_gravitational_acceleration(
                                1.0,
                                particle_position,
                                keys[tree[node_index].index_start + i].vector
                            )
                return acc_sum
        else:
            return self.calculate_gravitational_acceleration(
                tree[node_index].total_mass,
                particle_position,
                tree[node_index].center_of_mass)

    def handle_interaction(
        self,
        r1: SIMD[DType.int32, 4],
        r2: SIMD[DType.int32, 4],
        v1: SIMD[DType.float32, 4],
        v2: SIMD[DType.float32, 4],
        mut collision_pairs : List[Tuple[Int, Int, SIMD[DType.float32, 4], SIMD[DType.float32, 4]]],
        pair: Tuple[Int, Int]
        ):
        var dv = v2 - v1
        var dr = (r2 - r1).cast[DType.float32]()
        var dv_mag_sq = (dv * dv).reduce_add()
        var dvdotdr = (dv * dr).reduce_add()

        if (- dvdotdr  <= self.dt * dv_mag_sq) & (-dvdotdr  > 0.0) & (dv_mag_sq != 0.0): #checks if minimum time is in the dt frame
            var dr_mag_sq = (dr * dr).reduce_add()
            if dr_mag_sq * dv_mag_sq - (dvdotdr * dvdotdr)  < self.radius_sq*dv_mag_sq: #write out the checks and do some algebra it will make sense
                var new_v1 = 0.5 * ((v1 + v2) + self.cof_of_restitution * dv )
                var new_v2 = 0.5 * ((v1 + v2) - self.cof_of_restitution * dv )
                collision_pairs.append((pair[0],pair[1],new_v1, new_v2))

    def pos_step(mut self):
        var next_pos = self.pos.copy()
        for i in range(len(self.pos)):
            next_pos[i] = self.pos[i] + (self.vel[i] * self.dt).cast[DType.int32]()
            for j in range(3):
                if int_minhalf >= next_pos[i][j] or next_pos[i][j] >= int_maxhalf:
                    self.vel[i][j] = -self.vel[i][j]
                    next_pos[i] = self.pos[i]
        self.pos = next_pos^


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
    def energy(self) raises -> PythonObject:
        return 0


    def collision(self) raises -> PythonObject:
        return 0

    @staticmethod
    def norm(r1: SIMD[DType.int32, 4], r2: SIMD[DType.int32, 4]) -> Float32:
        var f1 = r1.cast[DType.float32]()
        var f2 = r2.cast[DType.float32]()
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
        raw_data[i] = Int32(random_si64(-1073741, 1073741)) #-10737418, 10737418

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
        m.def_function[energy]("energy")
        m.def_function[collision]("collision")
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
    var interaction_range_agr = Float32(py=args[5])
    var radius = Float32(py=args[6])
    var cof_of_restitution = Float32(py=args[7])
    var dt = Float32(py=args[8])
    print(g)
    # Allocate memory, leak ownership so it persists for Python, then write to it
    var allocation = alloc(Layout[Simulation].single())
    var ptr = allocation^.unsafe_leak()
    ptr.unsafe_write(
        Simulation(
            n=n, c1=c1, epsilon=epsilon, g=g, theta=theta,
            interaction_range_agr=interaction_range_agr, radius=radius,
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

def energy(args: PythonObject) raises -> PythonObject:
    var handle = Int(py=args[0])
    var ptr = Pointer[Simulation, MutUntrackedOrigin](unsafe_from_address=handle)
    return ptr[].energy()
def collision(args: PythonObject) raises -> PythonObject:
    var handle = Int(py=args[0])
    var ptr = Pointer[Simulation, MutUntrackedOrigin](unsafe_from_address=handle)
    return ptr[].collision()
def destroy_simulation(args: PythonObject) raises -> PythonObject:
    var handle = Int(py=args[0])
    var ptr = Pointer[Simulation, MutUntrackedOrigin](unsafe_from_address=handle)

    # Safely deinitialize the struct and free the heap memory
    ptr.unsafe_deinit_pointee()
    ptr.unsafe_free()

    return PythonObject()

#def main() raises:
#    print("Initializing simulation...")
#
#    # Create a simulation with 100 particles and a velocity scale of 1.0
#    var sim = Simulation(n=1000, c1=1.0)
#
#    print("Running simulation steps...")
#
#    # Run for 5 steps to test timing output and execution
#    for i in range(5):
#        print("\n--- Step", i, "---")
#        sim.step()
#        print(sim.pos[0])
#        print(sim.vel[0])
#        print(sim.acc[0])
