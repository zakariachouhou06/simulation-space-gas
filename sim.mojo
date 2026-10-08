from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64
from std.memory import Pointer, ArcPointer
from physics import run_timestep
from std.random import random_si64
comptime MAX_MULTIPOLE_TERMS = 9

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

def main() raises:
    seed()
    # 1. Import NumPy
    var np = Python.import_module("numpy")

    # 2. Simulation Parameters
    var n = 1000        # Number of particles
    var c1 = 0.01      # Velocity scaling coefficient
    var t : Float32= 0


    # 3. Initialize Positions and Velocities using helper function
    var pos = create_pos_vector_int(n)

    var vel_initial = c1 * (np.random.rand(n, 3) - 0.5)
    var vel = create_vel_vector_float(n, vel_initial)

    # 4. Initialize Accelerations
    var acc = List[SIMD[DType.float32, 4]]()
    acc.resize(n, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    for _ in range(100):
        run_timestep(t, pos, vel, acc)
        print(pos[0])

    #creating the tree:
