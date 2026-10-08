from std.python import Python, PythonObject
from std.random import seed, rand, random_ui64
from std.memory import Pointer, ArcPointer
from tree import tree_node, build_tree
from std.math import sqrt
from spatial import spatial_maker, extract_octant_bits, ParticleKey

comptime epsilon = 400  # Softening factor
comptime G = 400000000      # Gravitational constant
comptime theta = 0.04
comptime interaction_range = 0x00000FFF
comptime radius: Float32 = 0.001
comptime cof_of_restitution: Float32 = 0.5
comptime dt: Float32=0.4

def calculate_gravitational_acceleration(
    m1: Float32,
    pos1: SIMD[DType.float32, 4],
    pos2: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:

    # 1. Vector distance: r = pos2 - pos1
    var r_vec = pos2 - pos1

    # 2. Calculate squared distance using only the 3 spatial lanes (x, y, z)
    # Zero out index 3 just in case the ID subtraction caused a weird non-zero value
    r_vec[3] = 0.0

    var r_sq = (r_vec[0] * r_vec[0]) + (r_vec[1] * r_vec[1]) + (r_vec[2] * r_vec[2])

    # 3. Add epsilon squared for softening (no need for a hard r_sq == 0 check anymore!)
    var r_sq_eps = r_sq + (epsilon * epsilon)

    # 4. Compute softened r and r^3
    var r = sqrt(r_sq_eps)
    var r_cubed = r_sq_eps * r

    # 5. Scaling factor: (G * m1) / (r^2 + eps^2)^(3/2)
    var scale = (G * m1) / r_cubed

    # 6. Multiply the direction vector by the scale factor (lane 3 remains 0)
    var acc = r_vec * scale
    return acc

def calculate_gravitational_acceleration(
    m1: Float32,
    pos1: SIMD[DType.int32, 4],
    pos2: SIMD[DType.int32, 4],) -> SIMD[DType.float32, 4]:

    # 1. Vector distance: r = pos2 - pos1
    var mask = SIMD[DType.int32, 4](1, 1, 1, 0)

    # 2. Apply the mask to both vectors
    var masked_pos1 = pos1 * mask
    var masked_pos2 = pos2 * mask

    var f1 = masked_pos1.cast[DType.float32]()
    var f2 = masked_pos2.cast[DType.float32]()

    var r_vec = f1 - f2

    var r_sq = (r_vec[0] * r_vec[0]) + (r_vec[1] * r_vec[1]) + (r_vec[2] * r_vec[2])

    # 3. Add epsilon squared for softening (no need for a hard r_sq == 0 check anymore!)
    var r_sq_eps = r_sq + (epsilon * epsilon)

    # 4. Compute softened r and r^3
    var r = sqrt(r_sq_eps)
    var r_cubed = r_sq_eps * r

    # 5. Scaling factor: (G * m1) / (r^2 + eps^2)^(3/2)
    var scale = (G * m1) / r_cubed

    # 6. Multiply the direction vector by the scale factor (lane 3 remains 0)
    var acc = r_vec * scale
    return acc

def calculate_center_mass(pos: List[SIMD[DType.int32, 4]], mass: List[Float32]) -> SIMD[DType.float32, 4]:
    # Initialize our numerator (moment vector) and denominator (total mass)
    var total_moment = SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)
    var total_mass: Float32 = 0.0

    # Loop through each position and corresponding mass
    for i in range(len(pos)):
        # Cast the uint32 SIMD vector to float32 element-wise
        var p_float = pos[i].cast[DType.float32]()
        var m = mass[i]

        # Accumulate weighted position and total mass
        total_moment += p_float * m
        total_mass += m

    # Guard against division by zero
    if total_mass == 0.0:
        return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 0.0)

    # Final division: Center of Mass = Total Moment / Total Mass
    return total_moment / total_mass

def norm( r1: SIMD[DType.int32, 4], r2: SIMD[DType.int32, 4]) -> Float32:
    # 1. Create the mask to zero out the ID at index 3
    var mask = SIMD[DType.int32, 4](1, 1, 1, 0)

    # 2. Apply the mask to both vectors
    var masked_r1 = r1 * mask
    var masked_r2 = r2 * mask

    var f1 = masked_r1.cast[DType.float32]()
    var f2 = masked_r2.cast[DType.float32]()

    var diff = f1 - f2
    var distsq = (diff*diff).reduce_add()
    return sqrt(distsq)

def norm( r1: SIMD[DType.float32, 4], r2: SIMD[DType.float32, 4]) -> Float32:
    var diff = r1 - r2
    var distsq = (diff*diff).reduce_add()
    return sqrt(distsq)

def compute_interactions(mut tree: List[tree_node],node_index:Int, particle_id_keys : Int, keys: List[ParticleKey], mut interaction_pairs: List[Tuple[Int, Int]]) -> SIMD[DType.float32, 4]:

    var particle_position = keys[particle_id_keys].vector
    var particle_id = Int(particle_position[3])
    var s = Float32(0xFFFFFFFF >> tree[node_index].level)
    var d = norm(tree[node_index].center, particle_position)
    if (s / d)< theta:
        return calculate_gravitational_acceleration(tree[node_index].total_mass, tree[node_index].center, particle_position)
    else:
        if d < interaction_range:
            for i in range(tree[node_index].number_of_particles):
                var p2_id = Int(keys[tree[node_index].index_start + i].vector[3])
                if p2_id < particle_id:
                    interaction_pairs.append((p2_id, particle_id))
        var acc = SIMD[DType.float32, 4](0, 0, 0, 0)
        if tree[node_index].child_initialized == True:
            for child in tree[node_index].children:
                acc += compute_interactions(tree, child, particle_id, keys, interaction_pairs)
        else:
            for i in range(tree[node_index].number_of_particles):
                acc = calculate_gravitational_acceleration(1,keys[tree[node_index].index_start + i].vector, particle_position)
        return acc

def handle_interaction(r1: SIMD[DType.int32, 4], r2: SIMD[DType.int32, 4], v1: SIMD[DType.float32, 4], v2: SIMD[DType.float32, 4], dt: Float32) -> Tuple[SIMD[DType.float32, 4], SIMD[DType.float32, 4]]:
    var maskint = SIMD[DType.int32, 4](1, 1, 1, 0)
    var masked_r1 = r1 * maskint
    var masked_r2 = r2 * maskint
    var dr = (masked_r2 - masked_r1).cast[DType.float32]()
    var dv = v2 - v1
    var dvdotdr = (dv*dr).reduce_add()
    var t_min = dvdotdr / (dv*dv).reduce_add()
    if (t_min < dt) & (t_min > 0):
        var min_r = dr + dv * t_min
        var min_distance = sqrt((min_r*min_r).reduce_add())
        if min_distance < radius:
            new_v1 = (v1 + v2) / 2 + cof_of_restitution * dv
            new_v2 = (v1 + v2) / 2 - cof_of_restitution * dv
            return (new_v1, new_v2)
        else:
            return (v1, v2)
    else:
        return (v1, v2)



def convert_list_int(float_list: List[SIMD[DType.float32, 4]]) -> List[SIMD[DType.int32, 4]]:
    return [x.cast[DType.int32]() for x in float_list]

def run_timestep(mut t: Float32,mut pos: List[SIMD[DType.int32, 4]],mut vel: List[SIMD[DType.float32, 4]],mut acc: List[SIMD[DType.float32, 4]]) -> None:
    var keys = spatial_maker(pos)
    var tree = build_tree(keys)
    var interaction_pairs = List[Tuple[Int, Int]]()

    #updating acc
    for i in range(len(keys)):
        var pos = keys[i].vector #note index 3 is the id
        var id = Int(pos[3])
        acc[id] = compute_interactions(tree, 0, i, keys, interaction_pairs)
    #updating v
    for i in range(len(vel)):
        vel[i] += dt * acc[i]
    #update v collisions
    for pair in interaction_pairs:
        var result = handle_interaction(pos[pair[0]], pos[pair[1]], vel[pair[0]], vel[pair[1]], dt)
        vel[pair[0]] = result[0]
        vel[pair[1]] = result[1]
    #update pos
    t += dt
    for i in range(len(pos)):
        pos[i] += (dt * vel[i]).cast[DType.int32]()

        #var current_node = 0
        #var level: UInt32 = 0
        #while tree[current_node].is_leaf == False:
        #    var next_node = extract_octant_bits(key,level)
        #    current_node = tree[current_node].children[Int(next_node)]
        #    level += 1
        #current_node is the leaf node index
