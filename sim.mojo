from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64

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

def create_pos_vector_int(n: Int) -> List[SIMD[DType.uint32, 4]]:
    """Helper function to generate a Mojo SIMD vector list with random uint32 values."""
    # We need 4 values for each SIMD vector (3 random + 1 zero)
    var total_values = n * 4
    var raw_data = List[UInt32](capacity=total_values)
    raw_data.resize(total_values, 0)

    # Fill the span with fully scrambled uint32 values across the full 32-bit range
    rand(raw_data, min=0, max=4294967295)

    var vec = List[SIMD[DType.uint32, 4]]()
    vec.resize(n, SIMD[DType.uint32, 4](0, 0, 0, 0))

    for i in range(n):
        var base = i * 4
        vec[i] = SIMD[DType.uint32, 4](
            raw_data[base],
            raw_data[base + 1],
            raw_data[base + 2],
            0
        )
        vec[i][3] = UInt32(i)

    return vec^

struct ParticleKey(Copyable, ImplicitlyCopyable):
    var key: UInt64
    var vector: SIMD[DType.uint32, 4]

    def __init__(out self, key: UInt64, vector: SIMD[DType.uint32, 4]):
        self.key = key
        self.vector = vector


def get_key(v: SIMD[DType.uint32, 4]) -> UInt64:
    """Interleaves the first 3 components (x, y, z) into a single code.
    x = 1011, y = 1010, z = 0110 -> 110 001
    note: rest gets truncated
    note: this is a example with type byte to 2 bytes so small scale demo.
    """
    var x = UInt64(v[0]) << 32
    var y = UInt64(v[1]) << 32
    var z = UInt64(v[2]) << 32
    var result = UInt64(0)
    for i in range(20):
        var shift = UInt64(i)
        var bit = (x << shift) & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000
        result |= bit >> 3*shift
        #this part of the function is to grab the ith element and move it to the posion i*3 for y and z there is a offset the big number is a mask.
        bit = (y << shift) & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000
        result |= bit >> (3*shift + 1)
        bit = (z << shift) & 0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000
        result |= bit >> (3*shift + 2)
    return result

def get_bit(v: UInt64, shift: UInt64) -> Bool:
    var x :UInt64 = v^
    return Bool((x >> 63 - shift) & 0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0001)

def radix_sort_u64(mut keys: List[ParticleKey]):
    var n = len(keys)
    if n <= 1:
        return
    var temp = List[UInt64](capacity=n)
    for _ in range(n):
        temp.append(0)

    # Process each byte (8 passes total for 64 bits)
    for shift in range(0, 64, 8):
        var count = List[Int](capacity=256)
        for _ in range(256):
            count.append(0)

        # 1. Count frequencies of each byte value
        for i in range(n):
            var byte_val = UInt64((keys[i].key >> UInt64(shift)) & 0xFF)
            count[byte_val] += 1

        # 2. Compute prefix sums (cumulative counts)
        var total = 0
        for i in range(256):
            var count_val = count[i]
            count[i] = total
            total += count_val

        # 3. Place elements into temporary array stably
        for i in range(n):
            var byte_val = UInt64((keys[i].key >> UInt64(shift)) & 0xFF)
            temp[count[byte_val]] = keys[i].key
            count[byte_val] += 1

        # 4. Copy back to the original array for the next pass
        for i in range(n):
            keys[i].key = temp[i]

struct CoarseEntry(Copyable, ImplicitlyCopyable):
    var start_idx: Int
    var end_idx: Int
    def __init__(out self, start_idx: Int, end_idx: Int):
            self.start_idx = start_idx
            self.end_idx = end_idx

def extract_octant_bits(key: UInt64, level: UInt64) -> UInt64:
    var bit = (key[0] >> 63 - 3*level) & 0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0111
    return bit



def populate_coarse_entries(keys: List[ParticleKey], depth: Int = 4) -> List[CoarseEntry]:
    var n = len(keys)
    var lookup_table = List[CoarseEntry]()
    var size :Int=0
    for i in range(depth):
        size += 8*(i+1)
    lookup_table.reserve(size)
    for level in range(depth):
        for octant in range(8):
            var start_idx = -1
            var end_idx = -1

            for j in range(n):
                var key = keys[j].key
                var current_octant = extract_octant_bits(key, UInt64(level))

                if current_octant == UInt64(octant):
                    if start_idx == -1:
                        start_idx = j
                    end_idx = j + 1

            if start_idx == -1:
                start_idx = 0
                end_idx = 0

            lookup_table.append(CoarseEntry(start_idx=start_idx, end_idx=end_idx))
    return lookup_table^
def treemaker(
    pos: List[SIMD[DType.uint32, 4]]):
    var n = len(pos)
    var keys = List[ParticleKey]()
    keys.reserve(n)
    for i in range(n):
        var key = get_key(pos[i])
        keys.append(ParticleKey(key, pos[i]))
    radix_sort_u64(keys)
    var lookup_table = populate_coarse_entries(keys)

def main() raises:
    seed()
    # 1. Import NumPy
    var np = Python.import_module("numpy")

    # 2. Simulation Parameters
    var n = 100        # Number of particles
    var c1 = 0.01      # Velocity scaling coefficient
    var dt = 0.1       # Time step
    var epsilon = 0.1  # Softening factor
    var G = 0.003      # Gravitational constant

    # 3. Initialize Positions and Velocities using helper function
    var pos = create_pos_vector_int(n)

    var vel_initial = c1 * (np.random.rand(n, 3) - 0.5)
    var vel = create_vel_vector_float(n, vel_initial)

    # 4. Initialize Accelerations
    var acc = List[SIMD[DType.float32, 4]]()
    acc.resize(n, SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0))
    treemaker(pos)

    #creating the tree:
