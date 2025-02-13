# flood_fill.pyx

# Disable bounds checking and wraparound for performance
# These directives must be placed at the top of the file
# Alternatively, they can be placed above the function definition
# using decorators as shown in the original code.

from libc.stdlib cimport malloc, free
from libc.string cimport memset
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def flood_fill_cython(tuple dims, list voxels, list seed):
    """
    Perform flood fill on a 3D voxel grid using Cython for performance.

    Parameters:
    - dims: Tuple of ints (dimx, dimy, dimz)
    - voxels: List of lists, each sublist is [x, y, z] of active voxel
    - seed: List of 3 ints, the [x, y, z] seed position

    Returns:
    - connected_voxels: List of [x, y, z] in the same connected component as seed
    """
    cdef int dimx = dims[0]
    cdef int dimy = dims[1]
    cdef int dimz = dims[2]
    cdef long total_size = dimx * dimy * dimz

    # Allocate memory for the grid and initialize to 0
    cdef unsigned char *grid = <unsigned char *>malloc(total_size * sizeof(unsigned char))
    if grid == NULL:
        raise MemoryError("Unable to allocate memory for the grid.")
    memset(grid, 0, total_size * sizeof(unsigned char))


    cdef int i, x, y, z, idx

    # Mark active voxels in the grid
    for i in range(len(voxels)):
        x = voxels[i][0]
        y = voxels[i][1]
        z = voxels[i][2]
        if (0 <= x < dimx) and (0 <= y < dimy) and (0 <= z < dimz):
            idx = x * dimy * dimz + y * dimz + z
            grid[idx] = 1  # Active voxel

    # Initialize seed
    x = seed[0]
    y = seed[1]
    z = seed[2]

    # Check if seed is within bounds
    if not (0 <= x < dimx and 0 <= y < dimy and 0 <= z < dimz):
        free(grid)
        raise ValueError("Seed position is out of bounds.")

    idx = x * dimy * dimz + y * dimz + z

    # Check if seed voxel is active
    if grid[idx] != 1:
        free(grid)
        return []

    # Initialize queue
    # Preallocate queue with maximum possible size
    cdef long max_queue_len = total_size
    cdef long *queue = <long *>malloc(max_queue_len * 3 * sizeof(long))
    if queue == NULL:
        free(grid)
        raise MemoryError("Unable to allocate memory for the queue.")

    cdef long queue_start = 0
    cdef long queue_end = 0

    # Enqueue the seed voxel
    queue[queue_end * 3 + 0] = x
    queue[queue_end * 3 + 1] = y
    queue[queue_end * 3 + 2] = z
    queue_end += 1
    grid[idx] = 2  # Mark as visited

    # Define neighbor offsets for 6-connectivity
    cdef int neighbor_offsets[6][3]
    neighbor_offsets[0][:] = [-1, 0, 0]
    neighbor_offsets[1][:] = [1, 0, 0]
    neighbor_offsets[2][:] = [0, -1, 0]
    neighbor_offsets[3][:] = [0, 1, 0]
    neighbor_offsets[4][:] = [0, 0, -1]
    neighbor_offsets[5][:] = [0, 0, 1]

    # Initialize list for connected voxels
    cdef list connected_voxels = []
    connected_voxels.append([x, y, z])

    cdef int n, nx, ny, nz, neighbor_idx
    cdef int dx, dy, dz

    # Perform BFS
    while queue_start < queue_end:
        # Dequeue the current voxel
        x = queue[queue_start * 3 + 0]
        y = queue[queue_start * 3 + 1]
        z = queue[queue_start * 3 + 2]
        queue_start += 1

        # Explore all 6-connected neighbors
        for n in range(6):
            dx = neighbor_offsets[n][0]
            dy = neighbor_offsets[n][1]
            dz = neighbor_offsets[n][2]

            nx = x + dx
            ny = y + dy
            nz = z + dz

            # Check boundaries
            if (0 <= nx and nx < dimx) and (0 <= ny and ny < dimy) and (0 <= nz and nz < dimz):
                neighbor_idx = nx * dimy * dimz + ny * dimz + nz
                if grid[neighbor_idx] == 1:
                    # Enqueue the neighbor
                    if queue_end < max_queue_len:
                        queue[queue_end * 3 + 0] = nx
                        queue[queue_end * 3 + 1] = ny
                        queue[queue_end * 3 + 2] = nz
                        queue_end += 1
                        grid[neighbor_idx] = 2  # Mark as visited
                        connected_voxels.append([nx, ny, nz])

    # Free allocated memory
    free(grid)
    free(queue)

    return connected_voxels
