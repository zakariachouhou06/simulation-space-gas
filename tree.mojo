#Libs
from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64
from std.memory import Pointer, ArcPointer, UnsafePointer
from std.collections import List
#file structure
import spatial
from spatial import CoarseEntry, ParticleKey, get_key, get_bit, extract_octant_bits

comptime MAX_MULTIPOLE_TERMS = 9

struct tree_node:
    var children: List[Int]  # Stores indices of child nodes in the main node list
    var parent: Int          # Index of the parent node (-1 if root)

    # Center and bounds
    var min_corner: SIMD[DType.int32, 4]
    var max_corner: SIMD[DType.int32, 4]
    var center: SIMD[DType.int32, 4]
    var id: Int              # Consistent as Int

    # Multipole arrays
    var l: List[Float32]
    var m: List[Float32]
    var l_local: List[Float32]
    var m_local: List[Float32]
    #center mass
    var center_of_mass: SIMD[DType.int32, 4]
    var total_mass: Float32
    #index
    var index_start: UInt64
    var index_end: UInt64
    var level: UInt32
    var number_of_particles: UInt64 # Kept as UInt64
    #flags
    var center_of_mass_valid: Bool
    var is_leaf: Bool
    var child_initialized: Bool

    def __init__(
        out self: Self,
        parent: Int,
        level: UInt32,
        index_start: UInt64,
        index_end: UInt64,
        min_corner: SIMD[DType.int32, 4],
        max_corner: SIMD[DType.int32, 4],
    ):
        self.parent = parent
        self.level = level
        self.index_start = index_start
        self.index_end = index_end
        self.min_corner = min_corner
        self.max_corner = max_corner
        self.children = List[Int]()
        self.center_of_mass_valid = False
        self.is_leaf = False
        self.child_initialized = False
        self.children.reserve(8)

        self.l = List[Float32]()
        self.m = List[Float32]()
        self.l_local = List[Float32]()
        self.m_local = List[Float32]()
        self.center_of_mass = SIMD[DType.int32, 4]()
        self.total_mass = Float32()
        self.number_of_particles = index_end - index_start
        self.id = 0
        self.center = calculate_center(self.min_corner, self.max_corner)

    def calculate_center_of_mass(mut self, positions: List[SIMD[DType.int32, 4]], mass: List[Float32]) -> Tuple[SIMD[DType.int32, 4], Float32]:
        var total_mass: Float32 = 0
        var center_of_mass = SIMD[DType.float32, 4](0)
        for i in range(len(mass)):
            total_mass += mass[i]
            var r = positions[i] - self.min_corner
            center_of_mass += r.cast[DType.float32]() * mass[i]
        center_of_mass = center_of_mass / total_mass
        return (center_of_mass.cast[DType.int32]() + self.min_corner, total_mass)

    def divide(mut self, self_index: Int, keys: List[ParticleKey]) -> List[tree_node]:
            var new_children = List[tree_node]()
            if self.number_of_particles < 9:
                self.is_leaf = True
                var pos_particles = List[SIMD[DType.int32, 4]]()
                var masses = List[Float32]()
                masses.resize(unsafe_uninit_length=Int(self.number_of_particles))
                pos_particles.resize(unsafe_uninit_length=Int(self.number_of_particles))
                for i in range(self.number_of_particles):
                    masses[i] = 1.0
                    pos_particles[i] = keys[self.index_start + i].vector
                self.center_of_mass, self.total_mass = self.calculate_center_of_mass(pos_particles, masses)
                self.center_of_mass_valid = True
                return new_children^

            # calculate the child begin and end index
            var child_index_start: List[UInt64] = List[UInt64]()
            child_index_start.append(self.index_start)
            var child_index_end: List[UInt64] = List[UInt64]()
            var last_child_index : UInt64 = 0

            # Cast UInt64 to Int for range()
            for offset in range(Int(self.number_of_particles)):
                var key = keys[offset + Int(self.index_start)].key
                var child_index = extract_octant_bits(key, self.level)
                var difference = child_index - last_child_index
                for _ in range(difference):
                    child_index_start.append(UInt64(offset) + self.index_start)
                    child_index_end.append(UInt64(offset) + self.index_start)
                last_child_index = child_index
            child_index_end.append(self.index_end)

            #pad so that it always reaches 8 so you don't get any nasty out of bound.
            while len(child_index_start) < 8:
                child_index_start.append(self.index_end)
                child_index_end.append(self.index_end)

            for ix in range(2):
                for iy in range(2):
                    for iz in range(2):
                        var mask_vector = SIMD[DType.int32, 4](Int32(ix), Int32(iy), Int32(iz), Int32(0))
                        var min_corner_child = self.min_corner + self.center * mask_vector
                        var max_corner_child = self.center + self.center * mask_vector

                        # Instantiate child
                        var child = tree_node(
                            self_index,
                            self.level + 1,
                            index_start=child_index_start[ix*4+iy*2+iz],
                            index_end=child_index_end[ix*4+iy*2+iz],
                            min_corner=min_corner_child,
                            max_corner=max_corner_child
                        )

                        new_children.append(child^)
            self.child_initialized = True
            return new_children^


def calculate_center(min_corner: SIMD[DType.int32, 4], max_corner: SIMD[DType.int32, 4]) -> SIMD[DType.int32, 4]:
    return min_corner + ((max_corner - min_corner) >> 1)

def build_tree(mut keys: List[ParticleKey]) -> List[tree_node]:
    var root = tree_node(
        parent=-1,
        level=0,
        index_start=UInt64(0),
        index_end=UInt64(len(keys)),
        min_corner=SIMD[DType.int32, 4](-2147483648),
        max_corner=SIMD[DType.int32, 4](2147483647)
    )

    var all_nodes = List[tree_node]()
    all_nodes.append(root^)

    var search_nodes = List[Int]()
    search_nodes.append(0)

    while len(search_nodes) > 0:
        var current_id = search_nodes.pop(0)

        # Call divide cleanly without passing the lists
        var children = all_nodes[current_id].divide(current_id, keys)

        for _ in range(len(children)):
            var placeholder = children.pop(0)
            all_nodes.append(placeholder^)
            var child_index = len(all_nodes) - 1
            all_nodes[current_id].children.append(child_index)
            search_nodes.append(child_index)
    #initalize center of mass
    var current_node = 0
    while all_nodes[0].center_of_mass_valid == False:
        var counter = 0
        var n = 8
        for child in all_nodes[current_node].children:
            if all_nodes[child].center_of_mass_valid == False:
                current_node = child
                break
            else:
                counter += 1
        if counter == n:
            var pos_center_mass_child = List[SIMD[DType.int32, 4]]()
            var masses = List[Float32]()
            masses.resize(unsafe_uninit_length=n)
            pos_center_mass_child.resize(unsafe_uninit_length=n)
            for i in range(n):
                var child = all_nodes[current_node].children[i]
                masses[i] = all_nodes[child].total_mass
                pos_center_mass_child[i] = all_nodes[child].center_of_mass

            var result = all_nodes[current_node].calculate_center_of_mass(pos_center_mass_child, masses)
            # Assign the individual tuple elements to the node's fields
            all_nodes[current_node].center_of_mass = result[0]
            all_nodes[current_node].total_mass = result[1]
            all_nodes[current_node].center_of_mass_valid = True
            if current_node == 0:
                break
            current_node = all_nodes[current_node].parent

    return all_nodes^
