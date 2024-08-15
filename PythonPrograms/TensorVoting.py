import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_segment_end1():
    #generate segment from point to point
    start = np.array([1,1,1])
    end = np.array([8,8,8])

    num_points = 10000

    x_vals = np.linspace(start[0], end[0], num_points)
    y_vals = np.linspace(start[1], end[1], num_points)
    z_vals = np.linspace(start[2], end[2], num_points)

    segment = np.stack((x_vals, y_vals, z_vals), axis = -1)
    voxels = np.round(segment).astype(int)

    voxels = np.unique(voxels, axis = 0)

    grid_size = (9,9,9)
    vox_grid = np.zeros(grid_size, dtype=bool)
    
    for voxel in voxels:
        vox_grid[voxel[0], voxel[1], voxel[2]] = True
    
    return vox_grid

def test_segment_end2():
    grid_size = (8,8,8)
    vox_grid = np.zeros(grid_size, dtype=bool)
    p1 = np.array([1,1,1])
    p2 = np.array([1,1,2])
    p3 = np.array([1,2,3])
    p4 = np.array([1,2,4])
    p5 = np.array([1,2,5])
    p6 = np.array([2,3,6])
    p7 = np.array([2,4,6])

    voxels = np.array([p1, p2, p3, p4, p5, p6, p7])
    for voxel in voxels:
        vox_grid[voxel[0], voxel[1], voxel[2]] = True
    return vox_grid

def test_island1():
    grid_size = (8,8,8)
    vox_grid = np.zeros(grid_size, dtype=bool)
    p1 = np.array([1,1,1])
    p2 = np.array([1,1,2])
    p3 = np.array([1,2,3])
    p4 = np.array([1,2,4])
    p5 = np.array([1,2,5])
    p6 = np.array([2,3,6])
    p7 = np.array([2,4,6])

    voxels = np.array([p1, p2, p3])
    for voxel in voxels:
        vox_grid[voxel[0], voxel[1], voxel[2]] = True
    return vox_grid

def test_segment1():
    #generate segment from point to point
    start = np.array([0,0,0])
    end = np.array([8,8,8])

    num_points = 10000

    x_vals = np.linspace(start[0], end[0], num_points)
    y_vals = np.linspace(start[1], end[1], num_points)
    z_vals = np.linspace(start[2], end[2], num_points)

    segment = np.stack((x_vals, y_vals, z_vals), axis = -1)
    voxels = np.round(segment).astype(int)

    voxels = np.unique(voxels, axis = 0)

    grid_size = (9,9,9)
    vox_grid = np.zeros(grid_size, dtype=bool)
    
    for voxel in voxels:
        vox_grid[voxel[0], voxel[1], voxel[2]] = True
    
    return vox_grid

def test_segment2():
    grid_size = (8,8,8)
    vox_grid = np.zeros(grid_size, dtype=bool)
    p0 = np.array([0,0,0])
    p1 = np.array([1,1,1])
    p2 = np.array([1,1,2])
    p3 = np.array([1,2,3])
    p4 = np.array([1,2,4])
    p5 = np.array([1,2,5])
    p6 = np.array([2,3,6])
    p7 = np.array([2,4,6])

    voxels = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    for voxel in voxels:
        vox_grid[voxel[0], voxel[1], voxel[2]] = True
    return vox_grid

def test_reconnection1():
    grid_size = (2,2,14)
    vox_grid = np.zeros(grid_size, dtype=bool)
    p1 = np.array([1,1,1])
    p2 = np.array([1,1,2])
    p3 = np.array([1,1,3])
    p4 = np.array([1,1,4])
    p5 = np.array([1,1,5])
    p6 = np.array([1,1,6])

    s1 = np.array([1,1,8])
    s2 = np.array([1,1,9])
    s3 = np.array([1,1,10])
    s4 = np.array([1,1,11])
    s5 = np.array([1,1,12])
    s6 = np.array([1,1,13])

    voxels = np.array([p1, p2, p3, p4, p5, p6, s1, s2, s3, s4, s5, s6])

    for voxel in voxels:
        vox_grid[voxel[0], voxel[1], voxel[2]] = True
    return vox_grid


def load(file_path):
    img = nib.load(file_path)
    return img.get_fdata(), img.affine, img.header
    #return test_reconnection1(), img.affine #testing purposes

def save(data, affine, file_path):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, file_path)

def initialize():
    N = {}
    E = []
    S = []
    return N, E, S

def is_extremity(It, P):
    neighbors = np.sum(It[P[0] - 1:P[0]+2 , P[1]-1:P[1]+2,P[2]-1:P[2]+2]) - It[P]
    return neighbors == 1

def traverse(It, P):
    visited = set()
    stack = [P]
    segment = []

    while stack:
        voxel = stack.pop()
        if voxel not in visited:
            visited.add(voxel)
            segment.append(voxel)
            neighbors = [(voxel[0]+i, voxel[1]+j, voxel[2]+k)
                         for i in range(-1,2)
                         for j in range(-1,2)
                         for k in range(-1,2)
                         if (i,j,k) != (0,0,0)]
            for n in neighbors:
                if 0 <= n[0] < It.shape[0] and 0<= n[1] < It.shape[1] and 0<= n[2] < It.shape[2] and It[n]:
                    stack.append(n)
    return segment

def prune(It):
    N, E, S = initialize()
    It_copy = It.copy()

    for x in range (It.shape[0]):
        for y in range (It.shape[1]):
            for z in range (It.shape[2]):
                p = (x,y,z)
                if It_copy[p]:
                    if is_extremity(It_copy, p):
                        segment = traverse(It, p)
                        S.append(segment)
                        for voxel in segment:
                            It_copy[voxel] = 0
                        
                        first_extrem = segment[0]
                        second_extrem = segment[-1]

                        if is_extremity(It, first_extrem):
                            E.append(first_extrem)
                        else:
                            D = tuple(first_extrem)
                            if D not in N:
                                N[D] = []
                            N[D].append(len(S) - 1)
                        
                        if is_extremity(It, second_extrem):
                            E.append(second_extrem)
                        else:
                            D = tuple(second_extrem)
                            if D not in N:
                                N[D] = []
                            N[D].append(len(S) - 1)
    
    return N, E, S

def classify_tokens(N, E, S):
    beta = 6
    segment_ends = []
    islands = []
    segments = []

    for segment in S:
    
        segment_length = len(segment)
        connections = 0
        if(segment[0] in N):
            connections += 1
        if(segment[-1] in N):
            connections +=1

        if connections == 0 and segment_length > beta:
            segment_ends.append(segment)
        elif connections == 0 and segment_length <= beta:
            islands.append(segment)
        elif ((connections == 0 and segment_length >= 2*beta) or
              (connections == 1 and segment_length > beta) or
              connections == 2 and segment_length > 3):
            segments.append(segment)


    return segment_ends, islands, segments

def initialize_fields(shape):
    tensor_field = np.zeros(shape + (3,3))
    scalar_field = np.zeros(shape)
    return tensor_field, scalar_field

def express_tokens(tensor_field, scalar_field, segment_ends, islands, segments):
    L = 10
    theta = np.pi/4
    
    sig = L
    c = L**3/(4*np.sin(theta)**2)
    segment_ends_tensor(segment_ends, sig, c, tensor_field)
    islands_tensor(islands, sig, tensor_field)
    segments_tensor(segments, sig, tensor_field)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def clamp(value, minv = -1, maxv = 1):
    return max(min(value, maxv), minv)

def segment_ends_tensor(segment_ends, sig, c, tensor_field):
    for segment in segment_ends:
        O = np.array(segment[-1])
        V = np.array(segment[-1]) - np.array(segment[-2])
        V = normalize(V)
        
        for point in segment:
            P = np.array(point)
            OP = P - O
            OP = normalize(OP)
            W = 2*OP*clamp(np.dot(OP,V)) - V
            W = normalize(W)

            r = np.linalg.norm(P - O)
            phi = np.arccos(clamp(np.dot(OP, V)))

            weight = np.exp(-(r**2 + c* phi**2)/sig**2) * W

            TE = np.outer(weight, weight)

            tensor_field[tuple(P)] += TE
    
    for segment in segment_ends:
        O = np.array(segment[0])
        V = np.array(segment[0]) - np.array(segment[1])
        V = normalize(V)
        
        for point in segment:
            P = np.array(point)
            OP = P - O
            OP = normalize(OP)
            W = 2*OP*clamp(np.dot(OP,V)) - V
            W = normalize(W)

            r = np.linalg.norm(P - O)
            phi = np.arccos(clamp(np.dot(OP, V)))

            weight = np.exp(-(r**2 + c* phi**2)/sig**2) * W

            TE = np.outer(weight, weight)

            tensor_field[tuple(P)] += TE

def islands_tensor(islands, sig, tensor_field):
    for island in islands:
        C = np.array(island[int(len(island)/2)])

        neighborhood = [C + np.array([dx,dy,dz]) 
                        for dx in range(-1,2)
                        for dy in range(-1,2)
                        for dz in range(-1,2)]
        
        for point in neighborhood:
            P = np.array(point)
            CP = P - C
            CP = normalize(CP)
            
            weight = np.exp(-(np.linalg.norm(CP)**2)/ sig**2) * CP

            TI = np.outer(weight, weight)
            
            tensor_field[tuple(P)] += TI

def vec_rad(segment):
    vec = {}
    rad = {}

    for i in range(len(segment) - 1):
        point = tuple(segment[i])
        next_point = segment[i+1]
        vec[point] = np.array(next_point) - np.array(point)
        rad[point] = np.linalg.norm(vec[point])
        vec[point] = normalize(vec[point])
    
    return vec, rad

def segments_tensor(segments, sig, tensor_field):
    for segment in segments:
        vec, rad = vec_rad(segment)

        for point in segment:
            P = np.array(point)

            for vfp, V in vec.items():
                distance_vec = P -np.array(vfp)
                distance = np.linalg.norm(distance_vec)

                if distance < rad[vfp]:
                    weight = np.exp(-((distance - rad[vfp]**2)/sig**2)*normalize(distance_vec))

                    TS = np.outer(weight, weight)

                    tensor_field[tuple(P)] += TS


def saliency_map(tensor_field):
    # this is the unoptimized function, do not use
    map = np.zeros(tensor_field.shape[:3])

    it = np.nditer(tensor_field[...,0 ,0], flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        tensor = tensor_field[idx]

        if not np.all(tensor == 0):
            eigs = np.linalg.eigvalsh(tensor)
            eigs = np.sort(eigs)[::-1]
            l1, l2, _ = eigs
            map[idx] = l1 - l2
        it.iternext()

    return map

def saliency_map_optimized(tensor_field):
    T_flatten = tensor_field.reshape(-1, 3, 3)
    saliency_flat = np.zeros(T_flatten.shape[0])

    eigs = np.linalg.eigvalsh(T_flatten)

    eigs_sort = np.sort(eigs, axis = 1)[:, ::-1]

    l1 = eigs_sort[:, 0]
    l2 = eigs_sort[:, 1]
    saliency_flat = l1 - l2

    map = saliency_flat.reshape(tensor_field.shape[:3])

    return map

def find_neighborhood(point, shape):
    x, y, z = point
    neighbors = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue #avoid original pt

                nx, ny, nz = x + dx, y + dy, z + dz

                if 0 <= nx < shape[0] and 0<= ny < shape[1] and 0 <= nz < shape[2]:
                    neighbors.append((nx, ny, nz))

    return neighbors

def in_segments(point, segments):
    for segment in segments:
        if point in segment:
            return True
    return False

def merge(segment_ends, islands, segments, map, threshold, max_length):
    junctions = set()
    directions = {}
    paths = {}

    for segment in segment_ends:
        O = tuple(segment[-1])
        V = np.array(segment[-1]) - np.array(segment[-2])
        directions[O] = normalize(V)
    
    def create_path(start):
        """
        junctions = set()
        directions = {}
        paths = {}
        end_points = get_segment_end_points(segment_ends)
        for segment in segment_ends:
            O = tuple(segment[-1])
            V = np.array(segment[-1]) - np.array(segment[-2])
            directions[O] = normalize(V)
        
        def create_path(start_point):
            path = [start_point]
            current_point = start_point
            current_direction = directions[start_point]
            current_direction = directions[start_point]

            while True:
                neighbors = find_neighborhood(current_point, map.shape)
                next_point = None
                max_saliency = -np.inf
                for neighbor in neighbors:
                    if np.dot(np.array(neighbor) - np.array(current_point), current_direction > 0):     
                        if map[neighbor] > max_saliency:
                            max_saliency =map[neighbor]
                            next_point = neighbor
                        
                    if next_point is None or max_saliency < threshold or len(path)>max_length:
                        break

                    if next_point in path or next_point in segment:
                        break

                    path.append(next_point)
                    current_direction = normalize(np.array(next_point) - np.array(current_point))
                    current_point = next_point

                    if next_point in end_points:
                        end_points.remove(next_point)
                        paths[start_point] = path
                        junctions.add((start_point, next_point))
                        return
                    
                    if in_segments(next_point, segments):
                        nearest_element = min([p for seg in segments for p in seg], key = lambda p: np.linalg.norm(np.array(p)-np.array(next_point)))
                        paths[start_point] = path + [nearest_element]
                        junctions.add((start_point, nearest_element))
                        return
                    if in_segments(next_point, islands):
                        islands.remove([token for token in islands if next_point in token][0])
                        new_segment_end = path[-1]
                        directions[new_segment_end] = current_direction
                        paths[start_point] = path
                        junctions.add((start_point, new_segment_end))
                        create_path(new_segment_end)
                        return
                    
        
        for segment in segment_ends:
            start_point = tuple(segment[-1])
            create_path(start_point)

        return paths, junctions
        """
        path = [start]
        current = start
        current_direction = directions[start]

        while True:
            neighbors = find_neighborhood(current, map.shape)
            next_point = None
            max_saliency = -np.inf

            for neighbor in neighbors:
                if(np.dot(np.array(neighbor) - np.array(current), current_direction)) > 0:
                    if map[neighbor] > max_saliency:
                        max_saliency = map[neighbor]
                        next_point = neighbor

            if next_point is None or max_saliency < threshold or len(path) > max_length:
                break

            path.append(next_point)
            current_direction = normalize(np.array(next_point)- np.array(current))
            current = next_point

            if in_segments(next_point, segment_ends):
                segment_ends.remove([seg for seg in segment_ends if next_point in seg][0])
                paths[start] = path
                junctions.add((start, next_point))
                return
            
            if in_segments(next_point, segments):
                nearest_element = min([p for seg in segments for p in seg], 
                                      key = lambda p: np.linalg.norm(np.array(p) - np.array(next_point)))
                paths[start] = path + [nearest_element]
                junctions.add((start, nearest_element))
                return
            
            if in_segments(next_point, islands):
                #islands.remove([island for island in islands if next_point in island][0])
                new_segment_end = path[-1]
                directions[new_segment_end] = current_direction

                paths[start] = path
                junctions.add((start, new_segment_end))
                create_path(new_segment_end)
                return
            

    for segment in segment_ends:
        start = tuple(segment[-1])
        create_path(start)

    return paths, junctions

def change_radius(skeleton, paths):
    for path in paths.values():
        start_radius = skeleton[tuple(path[0])]
        end_radius = skeleton[tuple(path[-1])]
        if start_radius == 0:
            min_radius = end_radius
        elif end_radius == 0:
            min_radius = end_radius
        else:
            min_radius = min(start_radius, end_radius)

        for point in path:
            skeleton[tuple(point)] = min_radius
    
    return skeleton

def dense_vol_reconstruction(skeleton, paths, sx, sy, sz):
    volume_reconstructed = np.zeros_like(skeleton)
    
    for path in paths.values():
        for point in path:
            point = tuple(point)
            radius = skeleton[point]
            radx = max(radius/sx,1)
            rady = max(radius/sy,1)
            radz = max(radius/sz,1)

            
            z, y, x = np.ogrid[-point[0]:skeleton.shape[0]-point[0],
                               -point[1]:skeleton.shape[1]-point[1],
                               -point[2]:skeleton.shape[2]-point[2]]
            mask = (x/radx)**2 + (y/rady)**2 + (z/radz)**2 <= 1

            volume_reconstructed[mask] = 1
    
    return volume_reconstructed

def denoise(segments, alpha=2):
    return [sublist for sublist in segments if len(sublist) > alpha]


def main(in_path, out_path, rad_path):
    data, affine, header = load(in_path)
    skeleton, affine1, header1 = load(rad_path)
    sx, sy, sz = header.get_zooms()

    N, E, S = prune(data)
    segment_ends, islands, segments = classify_tokens(N,E,S)
    islands = denoise(islands)
    tensor_field, scalar_field = initialize_fields(data.shape)
    #for y in segment_ends: print(y)
    express_tokens(tensor_field, scalar_field, segment_ends, islands, segments)
    saliency = saliency_map_optimized(tensor_field)
    paths, junctions = merge(segment_ends.copy(), islands.copy(), segments.copy(), saliency, 0, 20)
    #visualize(segment_ends, segments, islands, paths)
    radii = change_radius(skeleton, paths)
    reconstructed = dense_vol_reconstruction(radii, paths, sx, sy, sz)

    save(reconstructed, affine, out_path)

    

def print_nonzero(tensor_field):
    it = np.nditer(tensor_field[..., 0, 0], flags = ['multi_index'])
    while not it.finished:
        idx = it.multi_index
        tensor = tensor_field[idx]
        if not np.all(tensor == 0):
            print(f"at {idx}")
            print(tensor)
        it.iternext()

def visualize(segment_ends, segments, islands, paths):
    fig = plt.figure(figsize = (15, 15))
    ax = fig.add_subplot(111, projection = '3d')

    

    for segment in segment_ends:
        print(segment)
        segment = np.array(segment)
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], 'bo-')
    
    for segment in islands:
        segment = np.array(segment)
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], 'go-')

    for segment in segments:
        segment = np.array(segment)
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], 'ro-')

    for path in paths.values():
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'k--')

    #ax.set_zlim([0,20])
    #ax.set_xlim([0,15])
    #ax.set_ylim([0,15])
    plt.show()


if __name__ == "__main__":
    main("/rsrch1/ip/bmi/Downloads/bmi.vesselthinned.nii.gz", "/rsrch1/ip/bmi/Downloads/bmi.reconstructed.nii.gz",
         "/rsrch1/ip/bmi/Downloads/bmi.radius.nii.gz")


 