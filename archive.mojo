from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64
from std.memory import Pointer, ArcPointer
from physics import physics_sim
from std.random import random_si64
comptime MAX_MULTIPOLE_TERMS = 9

struct Simulation:
    var n: Int
    var c1: Float32
    var t: Float32
    var np: PythonObject
    var pos: List[SIMD[DType.int32, 4]]
    var vel: List[SIMD[DType.float32, 4]]
    var acc: List[SIMD[DType.float32, 4]]

    def __init__(out self, n: Int, c1: Float32) raises:
        seed()
        self.n = n
        self.c1 = c1
        self.t = 0.0

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
        run_timestep(self.t, self.pos, self.vel, self.acc)
        # If run_timestep advances t itself, don't do this.
        self.t += 1.0

    def get_positions(self) raises -> PythonObject:
        var result = self.np.empty([self.n, 3], dtype=self.np.float32)

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
