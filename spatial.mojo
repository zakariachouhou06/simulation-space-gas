from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64
from std.memory import Pointer, ArcPointer, bitcast

struct ParticleKey(Copyable, ImplicitlyCopyable):
    var key: UInt64
    var vector: SIMD[DType.int32, 4]

    def __init__(out self, key: UInt64, vector: SIMD[DType.int32, 4]):
        self.key = key
        self.vector = vector

def get_key(v: SIMD[DType.int32, 4]) -> UInt64:
    """Interleaves the first 3 components (x, y, z) into a single code.
    fist cast x form int to uint (by ofsetting to min int limit)
    x = 1011, y = 1010, z = 0110 -> 110 001
    note: rest gets truncated
    note: this is a example with type byte to 2 bytes so small scale demo.
    """
    # 1. Bitcast the whole SIMD vector to uint32 and toggle the sign bit
    var u = bitcast[DType.uint32](v) ^ 0x80000000

    # 2. Extract components as UInt64 and align for interleaving
    var x = UInt64(u[0]) << 32
    var y = UInt64(u[1]) << 32
    var z = UInt64(u[2]) << 32
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
    return Bool((v >> 63 - shift) & 0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0001)

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

def extract_octant_bits(key: UInt64, level: UInt32) -> UInt64:
    var bit = (key[0] >> 64 - 3*(UInt64(level)+1)) & 0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0111
    return bit

struct CoarseEntry(Copyable, ImplicitlyCopyable):
    var start_idx: Int
    var end_idx: Int
    def __init__(out self, start_idx: Int, end_idx: Int):
            self.start_idx = start_idx
            self.end_idx = end_idx

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
                var current_octant = extract_octant_bits(key, UInt32(level))

                if current_octant == UInt64(octant):
                    if start_idx == -1:
                        start_idx = j
                    end_idx = j + 1

            if start_idx == -1:
                start_idx = 0
                end_idx = 0

            lookup_table.append(CoarseEntry(start_idx=start_idx, end_idx=end_idx))
    return lookup_table^

def spatial_maker(
    pos: List[SIMD[DType.int32, 4]]) -> List[ParticleKey]:
    var n = len(pos)
    var keys = List[ParticleKey]()
    keys.reserve(n)
    for i in range(n):
        var key = get_key(pos[i])
        keys.append(ParticleKey(key, pos[i]))
    radix_sort_u64(keys)
    #var lookup_table = populate_coarse_entries(keys)
    return keys^

def print_binary(val: UInt64):
    """Function for testing."""
    var s = String("")
    for i in range(64):
        # Extract bits from MSB (bit 63) down to LSB (bit 0)
        var bit = (val >> (63 - UInt64(i))) & 1
        s += String(Int(bit))
    print(s)
#test
#def main():
#        # Test a vector with positive, negative, and zero coordinates: (x, y, z, padding)
#        var test_vector = SIMD[DType.int32, 4](-2147483648, 0, 1073741825, 0)
#
#        var key = get_key(test_vector)
#
#        print("Coordinates: x=0, y=-2,147,483,648, z=-2,147,483,648")
#        print("Generated Morton Key (UInt64):")
#        print_binary(key)
