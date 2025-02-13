import os
import numpy as np
import time
import trimesh
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from shapely.geometry import Polygon, Point
import PIL.Image as Image
import pymeshlab
import random

from scipy.spatial import KDTree
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN
from shapely.prepared import prep
import matplotlib.pyplot as plt
import binvox_rw_customized as binvox_rw
from scipy.spatial import cKDTree
import igl
from omegaconf import OmegaConf
from flood_fill import flood_fill_cython
import glob

def poisson_blending3(joint_v, joint_t, corresponding_loop_e, corresponding_vertices, boundary_weight=10.0, force_correspondence=True):
    '''
    Performs Poisson blending on a 3D mesh, using edge lengths as weights and allowing customization of the boundary weight.

    :param joint_v: (N x 3) NumPy array of vertex positions.
    :param joint_t: (M x 3) NumPy array of triangle indices.
    :param corresponding_loop_e: List of boundary loops (each a list of vertex indices).
    :param corresponding_vertices: List of target vertex positions for the boundary loops.
    :param boundary_weight: Weight applied to the boundary conditions (default is 10.0).
    :param force_correspondence: If True, enforces strict correspondence between loops and their vertices.
    '''
    corresponding_loop_v = []
    corresponding_loop_targetv = []
    for i in range(len(corresponding_loop_e)):
        if corresponding_loop_e[i][-1] < 0:
            if force_correspondence:
                corresponding_loop_v.extend(corresponding_loop_e[i][:-1])
                corresponding_loop_targetv.extend(corresponding_vertices[i])
        else:
            corresponding_loop_v.extend(corresponding_loop_e[i])
            corresponding_loop_targetv.extend(corresponding_vertices[i])

    # Convert lists to NumPy arrays for efficient indexing
    corresponding_loop_v = np.array(corresponding_loop_v, dtype=np.int32)
    corresponding_loop_targetv = np.array(corresponding_loop_targetv, dtype=np.float32)

    # Poisson blending for each coordinate channel (X, Y, Z)
    for channel_id in range(3):
        print('Poisson blending -- channel', str(channel_id))

        # Prepare the edges from the triangles
        vi0 = np.concatenate([joint_t[:, 0], joint_t[:, 1], joint_t[:, 2]])
        vi1 = np.concatenate([joint_t[:, 1], joint_t[:, 2], joint_t[:, 0]])

        # Compute the differences for all edges
        dist = joint_v[vi1, channel_id] - joint_v[vi0, channel_id]

        num_edges = len(vi0)
        num_boundary = len(corresponding_loop_v)
        num_equations = num_edges + num_boundary

        # Compute edge weights as edge lengths
        edge_vectors = joint_v[vi1] - joint_v[vi0]  # Shape: (num_edges, 3)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        edge_weights = edge_lengths

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        edge_weights = edge_weights + epsilon

        # Apply weights to the edge equations
        sqrt_edge_weights = np.sqrt(edge_weights)
        weighted_dist = dist * sqrt_edge_weights

        # Prepare data arrays for the sparse matrix
        data_edges = np.empty(num_edges * 2, dtype=np.float32)
        row_edges = np.empty(num_edges * 2, dtype=np.int32)
        col_edges = np.empty(num_edges * 2, dtype=np.int32)

        edge_indices = np.arange(num_edges, dtype=np.int32)

        # Populate the data for edges
        data_edges[0::2] = sqrt_edge_weights
        row_edges[0::2] = edge_indices
        col_edges[0::2] = vi1

        data_edges[1::2] = -sqrt_edge_weights
        row_edges[1::2] = edge_indices
        col_edges[1::2] = vi0

        b_edges = weighted_dist

        # Prepare data for boundary conditions
        sqrt_boundary_weight = np.sqrt(boundary_weight)

        data_boundary = np.full(num_boundary, sqrt_boundary_weight, dtype=np.float32)
        row_boundary = np.arange(num_edges, num_edges + num_boundary, dtype=np.int32)
        col_boundary = corresponding_loop_v

        b_boundary = corresponding_loop_targetv[:, channel_id] * sqrt_boundary_weight

        # Concatenate edge and boundary data
        data = np.concatenate([data_edges, data_boundary])
        row = np.concatenate([row_edges, row_boundary])
        col = np.concatenate([col_edges, col_boundary])
        b = np.concatenate([b_edges, b_boundary])

        # Construct the sparse matrix A
        A = csr_matrix((data, (row, col)), shape=(num_equations, len(joint_v)))

        print('Computing least squares solution...')
        # Solve the least squares problem
        solution = lsqr(A, b)[0] #atol=1e-04, btol=1e-04
        print("done")

        # Update the vertex positions for the current channel
        joint_v[:, channel_id] = solution
    


def load_object_trimesh(obj_path):
    mesh = trimesh.load(obj_path)
    if not isinstance(mesh, trimesh.base.Trimesh):
        meshes = [geometry for geometry in mesh.geometry.values()]
        mesh = trimesh.util.concatenate(meshes)    
    return mesh



def normalize_points(points_unnormalized):
    bbox_scale = 1.1
    bbox_center = (points_unnormalized.min(0).values + points_unnormalized.max(0).values) / 2.
    bbox_len = (points_unnormalized.max(0).values - points_unnormalized.min(0).values).max()
    points_normalized = (points_unnormalized - bbox_center) * (2 / (bbox_len * bbox_scale))
    return points_normalized





def poisson_reconstruction(mesh):
    """
    Samples points from a mesh and performs Poisson surface reconstruction, returning a trimesh object.

    Args:
        mesh_file (str): Path to the mesh file.
        num_samples (int): Number of points to sample from the mesh.
        depth (int): Depth of the Poisson reconstruction octree (controls level of detail).

    Returns:
        reconstructed_trimesh (trimesh.Trimesh): The Poisson reconstructed mesh as a trimesh object.
    """
    


    # Sample points on the mesh
    points, face_indices = mesh.sample(num_samples, return_index=True)

    # Extract normals for the sampled points
    normals = mesh.face_normals[face_indices]
    #normals = normal_correction(points,normals)


    # Create an Open3D point cloud from sampled points and normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Poisson reconstruction
    print("starting poissson reconstruction")
    mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Convert Open3D mesh to numpy arrays
    vertices = np.asarray(mesh_poisson.vertices)
    faces = np.asarray(mesh_poisson.triangles)

    # Create a trimesh object from the vertices and faces
    output_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    #output_mesh.export("a.obj")
    #trimesh.Trimesh(vertices=points).export("points.obj")

    if largest_component:
        components = output_mesh.split(only_watertight=False)
        output_mesh = max(components, key=lambda c: len(c.faces))
    return output_mesh

def find_boundary_loops_numpy(verts, faces):
    """
    Find all boundary loops in a triangular mesh using NumPy.

    Args:
        faces (np.ndarray): A NumPy array of shape (m x 3) representing the faces of the mesh, where each face is a triplet of vertex indices.

    Returns:
        boundary_loop (np.ndarray): The longest boundary loop as a NumPy array of vertex indices.
    """

    # Step 1: Extract edges from triangles
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])

    # Step 2: Sort edges so that (i, j) and (j, i) are treated the same
    sorted_edges = np.sort(edges, axis=1)

    # Step 3: Flatten the sorted edges and count occurrences
    num_vertices = np.max(faces) + 1
    edge_indices = sorted_edges[:, 0] * num_vertices + sorted_edges[:, 1]
    
    # Count occurrences of each edge
    edge_counts = np.zeros(num_vertices * num_vertices, dtype=np.int16)
    np.add.at(edge_counts, edge_indices, 1)

    # Step 4: Find boundary edges (edges that appear exactly once)
    boundary_edge_mask = (edge_counts == 1)
    boundary_edges = sorted_edges[boundary_edge_mask[edge_indices]]

    # Step 5: Form multiple boundary loops by connecting boundary edges
    boundary_loops = []
    visited_edges = set()

    while len(visited_edges) < len(boundary_edges):
        # Find a new starting edge that hasn't been visited yet
        for edge in boundary_edges:
            edge_tuple = tuple(edge.tolist())
            if edge_tuple not in visited_edges:
                current_edge = edge_tuple
                break

        # Start a new loop
        boundary_loop = []
        boundary_loop.append(current_edge[0])
        boundary_loop.append(current_edge[1])
        visited_edges.add(current_edge)

        # Walk along the boundary to form the loop
        while True:
            last_vertex = boundary_loop[-1]

            found_next_edge = False
            for edge in boundary_edges:
                edge_tuple = tuple(edge.tolist())
                if edge_tuple not in visited_edges:
                    if edge_tuple[0] == last_vertex:
                        boundary_loop.append(edge_tuple[1])
                        visited_edges.add(edge_tuple)
                        found_next_edge = True
                        break
                    elif edge_tuple[1] == last_vertex:
                        boundary_loop.append(edge_tuple[0])
                        visited_edges.add(edge_tuple)
                        found_next_edge = True
                        break

            # If no more connected edges are found, the loop is complete
            if not found_next_edge:
                break

        # Add the completed boundary loop to the list
        boundary_loops.append(np.array(boundary_loop))

    # Return the longest boundary loop
    boundary_loops = sorted(boundary_loops, key=lambda x: x.shape[0], reverse=True)

    return boundary_loops[0]




def create_plane_mesh(min_x, max_x, min_y, max_y, resolution_x=50, resolution_y=50,scale = 1.2):
    """
    Creates a plane mesh in the xy-plane (z=0) with given dimensions and resolution.

    Args:
        min_x (float): Minimum x-coordinate.
        max_x (float): Maximum x-coordinate.
        min_y (float): Minimum y-coordinate.
        max_y (float): Maximum y-coordinate.
        resolution_x (int): Number of subdivisions along x-axis.
        resolution_y (int): Number of subdivisions along y-axis.

    Returns:
        vertices (numpy.ndarray): Array of vertices of the plane mesh.
        faces (numpy.ndarray): Array of faces (triangles) of the plane mesh.
    """
    min_x = min_x * scale
    min_y = min_y * scale
    max_x = max_x * scale
    max_y = max_y * scale

    # Create a grid of points in the xy-plane
    x = np.linspace(min_x, max_x, resolution_x)
    y = np.linspace(min_y, max_y, resolution_y)
    xx, yy = np.meshgrid(x, y)

    # Flatten the grid and create vertices with z=0
    vertices = np.vstack([xx.ravel(), yy.ravel()]).T

    # Create faces (triangles) for the plane
    faces = []
    for i in range(resolution_x - 1):
        for j in range(resolution_y - 1):
            # Get the indices of the four vertices of the current quad
            v0 = i * resolution_y + j
            v1 = (i + 1) * resolution_y + j
            v2 = (i + 1) * resolution_y + (j + 1)
            v3 = i * resolution_y + (j + 1)

            # Create two triangles for the quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = np.array(faces)
    return vertices, faces


def fit_rectangle_rotating_calipers(points):
    """
    Fit the best rectangle to a looped sequence of 2D points using the Rotating Calipers algorithm,
    and compute the 2x2 rotation matrix that aligns the rectangle to the axes.

    Parameters:
    points (ndarray): An Nx2 array of points (x, y), forming a loop (polygon).

    Returns:
    rectangle (ndarray): An array of shape (5, 2) representing the corners of the rectangle (closed loop).
    rotation_matrix (ndarray): A 2x2 array representing the rotation matrix that aligns the rectangle to the axes.
    """

    # Ensure the polygon is closed
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Create a Shapely polygon
    polygon = Polygon(points)

    # Get the minimum area rectangle (oriented minimum bounding box)
    min_rect = polygon.minimum_rotated_rectangle

    # Extract the rectangle's coordinates
    x, y = min_rect.exterior.coords.xy
    rectangle = np.column_stack([x, y])

    # Compute the rotation angle from the rectangle's first edge
    dx = rectangle[1, 0] - rectangle[0, 0]
    dy = rectangle[1, 1] - rectangle[0, 1]
    theta = np.arctan2(dy, dx)

    # Compute the rotation matrix to align the rectangle to the axes
    rotation_matrix = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta),  np.cos(-theta)]
    ])

    return rectangle, rotation_matrix

def boundary_vertices_projection(mesh):
    """
    Manually find the boundary of a 3D mesh by detecting edges that belong to only one face,
    fit a plane to the boundary vertices, project the vertices onto the plane,
    and return the indices of the boundary vertices and their projected 3D coordinates.

    Args:
        mesh (trimesh.Trimesh): Input 3D mesh.

    Returns:
        boundary_indices (numpy.ndarray): Indices of the boundary vertices in the original mesh.
        projected_boundary_vertices (numpy.ndarray): 3D coordinates of the boundary vertices after projection onto the plane.
    """

    faces = np.array(mesh.faces)
    verts = np.array(mesh.vertices)
    ###find boundary loop

    #boundary_indices = find_boundary_loops_pytorch(verts,faces)
    boundary_indices = find_boundary_loops_numpy(verts,faces)

    # Step 2: Extract boundary vertices
    boundary_vertices = verts[boundary_indices]

    # Step 3: Fit a plane using PCA (Principal Component Analysis)
    pca = PCA(n_components=2)
    pca.fit(boundary_vertices)
    
    # The normal of the plane is the third component from PCA
    plane_normal = np.cross(pca.components_[0], pca.components_[1]).reshape(1,3)
    plane_origin = pca.mean_.reshape(1,3)
    
    # Step 4: Project boundary vertices onto the plane
    vector = boundary_vertices - plane_origin
    distance = (vector * plane_normal).sum(axis = 1).reshape(-1,1)
    projected_vertices = boundary_vertices - distance * plane_normal
    boundary_2d = pca.transform(projected_vertices)

    # Step 5: Fit the best 2D rectangle in least squares sense to the 2D projected points
    rectangle_vertex, rotation_matrix = fit_rectangle_rotating_calipers(boundary_2d)



    # Step 5: generate the plane 
    aa_rectangle_vertex = np.matmul(rectangle_vertex,rotation_matrix.T)
    min_x, min_y = np.min(aa_rectangle_vertex, axis=0)
    max_x, max_y = np.max(aa_rectangle_vertex, axis=0)
    verts,faces = create_plane_mesh(min_x, max_x, min_y, max_y)
    verts = np.matmul(verts,rotation_matrix) #inverse the rotation matrix
    verts = pca.inverse_transform(verts)


    # Step 5: Return boundary vertex indices and their projected 3D coordinates
    return boundary_indices, projected_vertices,verts,faces


def align_plane_to_xy(plane_points,mesh_verts,boundary_verts,points_in_world):
    """
    Aligns a 3D plane mesh to the (x, y, 0) plane, centers it at the origin,
    aligns its axes with the global x and y axes, and scales it to fit within [-1, 1].

    Parameters:
    points (ndarray): An Nx3 array of 3D points.

    Returns:
    transformed_points (ndarray): An Nx3 array of transformed points.
    transformation_matrix (ndarray): A 4x4 transformation matrix (homogeneous coordinates).
    """
    # Step 1: Compute the centroid of the points
    points = plane_points
    centroid = np.mean(points, axis=0)

    # Step 2: Compute the normal vector of the plane using SVD
    centered_points = points - centroid
    _, _, Vt = np.linalg.svd(centered_points)
    normal = Vt[2, :]

    # Step 3: Compute rotation matrix to align normal to z-axis
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(normal, z_axis)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm != 0:
        rotation_axis /= rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    else:
        R = np.eye(3)

    # Step 4: Apply rotation to the centered points
    rotated_points = centered_points @ R.T

    # Step 5: Perform PCA on the rotated points to align with x and y axes
    xy_points = rotated_points[:, :2]
    mean_xy = np.mean(xy_points, axis=0)
    centered_xy = xy_points - mean_xy
    _, _, Vt_xy = np.linalg.svd(centered_xy)
    pc1 = Vt_xy[0, :]
    angle_z = np.arctan2(pc1[1], pc1[0])

    # Build rotation matrix around z-axis
    c = np.cos(-angle_z)
    s = np.sin(-angle_z)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    # Step 6: Apply rotation around z-axis
    rotated_points = rotated_points @ Rz.T

    # Step 7: Combine rotations into a single rotation matrix
    total_rotation = Rz @ R

    # Step 8: Build the homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = total_rotation
    transformation_matrix[:3, 3] = -total_rotation @ centroid

    # Step 9: Apply the transformation to the original points
    transformed_points = (points @ total_rotation.T) + transformation_matrix[:3, 3]

    mesh_verts = np.concatenate([mesh_verts,np.ones((mesh_verts.shape[0],1))],axis=1)
    mesh_verts = np.matmul(transformation_matrix,mesh_verts.T).T[:,:3]


    # Step 10: Compute scaling factor to fit within [-1, 1] without distorting proportions
    # Compute the ranges in x and y after alignment
    # min_coords = np.min(transformed_points[:, :2], axis=0)
    # max_coords = np.max(transformed_points[:, :2], axis=0)
    # ranges = max_coords - min_coords
    # max_range = np.max(ranges)

    min_coords = np.min(mesh_verts, axis=0)
    max_coords = np.max(mesh_verts, axis=0)
    ranges = max_coords - min_coords
    max_range = np.max(ranges)

    # print(max_range,max_range1)
    # exit(0)
    # Compute the scaling factor
    scaling_factor = 1.6 / max_range  # We want the largest side to be 2 units long (from -1 to 1)

    # Apply uniform scaling to the transformed points
    transformed_points *= scaling_factor

    # Step 11: Update the transformation matrix to include scaling
    # Build the scaling matrix in homogeneous coordinates
    scaling_matrix = np.eye(4)
    scaling_matrix[:3, :3] *= scaling_factor



    #print(transformation_matrix.shape)
    ###apply the transformation matrix to the mesh vertices
    mesh_verts = np.concatenate([mesh_verts,np.ones((mesh_verts.shape[0],1))],axis=1)
    mesh_verts = np.matmul(scaling_matrix,mesh_verts.T).T[:,:3]

    # Update the total transformation matrix
    transformation_matrix = scaling_matrix @ transformation_matrix
    points_in_world = np.concatenate([points_in_world,np.ones((points_in_world.shape[0],1))],axis=1)
    points_in_world = np.matmul(transformation_matrix,points_in_world.T).T[:,:3]
    boundary_verts = np.concatenate([boundary_verts,np.ones((boundary_verts.shape[0],1))],axis=1)
    boundary_verts = np.matmul(transformation_matrix,boundary_verts.T).T[:,:3]

    return transformed_points, mesh_verts, boundary_verts, points_in_world



def poisson_reconstruction_pymeshlab(mesh):
    """
    Performs Poisson surface reconstruction without adaptivity on a trimesh object,
    using the existing normals from the input mesh.

    Parameters:
        input_mesh (trimesh.Trimesh): The input mesh with vertex normals.
        octree_depth (int): The depth of the octree used in reconstruction (controls resolution).

    Returns:
        trimesh.Trimesh: The reconstructed mesh.
    """

    # Extract vertices and normals from the input mesh
    points, face_indices = mesh.sample(num_samples, return_index=True)

    # Extract normals for the sampled points
    normals = mesh.face_normals[face_indices]

    # Create a PyMeshLab MeshSet and add the input mesh with normals
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=points, v_normals_matrix=normals)
    ms.add_mesh(mesh, 'input_mesh')
    # Perform Poisson reconstruction with adaptivity disabled
    ms.apply_filter('generate_surface_reconstruction_screened_poisson',
                    depth=depth,
                    fulldepth=2)
    
    # Retrieve the reconstructed mesh
    reconstructed_mesh = ms.current_mesh()

    # Extract vertices and faces from the reconstructed mesh
    reconstructed_vertices = reconstructed_mesh.vertex_matrix()
    reconstructed_faces = reconstructed_mesh.face_matrix()

    # Create a new trimesh object from the reconstructed data
    output_mesh = trimesh.Trimesh(vertices=reconstructed_vertices,
                                  faces=reconstructed_faces,
                                  process=False)
    
    if largest_component:
        components = output_mesh.split(only_watertight=False)
        output_mesh = max(components, key=lambda c: len(c.faces))
    return output_mesh


def poisson_reconstruction_no_adapt(mesh, octree_depth=8,largest_component = False):
    """
    Performs Poisson surface reconstruction without adaptivity on a trimesh object,
    using the existing normals from the input mesh.

    Parameters:
        input_mesh (trimesh.Trimesh): The input mesh with vertex normals.
        octree_depth (int): The depth of the octree used in reconstruction (controls resolution).

    Returns:
        trimesh.Trimesh: The reconstructed mesh.
    """

    # Extract vertices and normals from the input mesh
    points, face_indices = mesh.sample(num_samples, return_index=True)

    # Extract normals for the sampled points
    normals = mesh.face_normals[face_indices]

    # Create a PyMeshLab MeshSet and add the input mesh with normals
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=points, v_normals_matrix=normals)
    ms.add_mesh(mesh, 'input_mesh')

    # Perform Poisson reconstruction with adaptivity disabled
    ms.apply_filter('generate_surface_reconstruction_screened_poisson',
                    depth=octree_depth,
                    fulldepth=octree_depth)

    # Retrieve the reconstructed mesh
    reconstructed_mesh = ms.current_mesh()

    # Extract vertices and faces from the reconstructed mesh
    reconstructed_vertices = reconstructed_mesh.vertex_matrix()
    reconstructed_faces = reconstructed_mesh.face_matrix()

    # Create a new trimesh object from the reconstructed data
    output_mesh = trimesh.Trimesh(vertices=reconstructed_vertices,
                                  faces=reconstructed_faces,
                                  process=False)
    
    if largest_component:
        components = output_mesh.split(only_watertight=False)
        output_mesh = max(components, key=lambda c: len(c.faces))
    return output_mesh


def combine_all_meshes_into_trimesh():
    """
    Combines all mesh objects in the Blender scene into a single trimesh object.
    
    Returns:
        trimesh.Trimesh: A single trimesh object containing all meshes in the scene.
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # Get the evaluated dependency graph for modifiers and deformations
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    for obj in bpy.data.objects:
        # Only process mesh objects
        if obj.type == 'MESH':
            # Get the evaluated version of the object (with modifiers applied)
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            
            # Get the object's world matrix for applying transformations
            world_matrix = obj.matrix_world
            
            # Extract vertices and apply transformations
            vertices = np.array([world_matrix @ vertex.co for vertex in mesh.vertices])
            faces = np.array([face.vertices[:] for face in mesh.polygons])
            
            # Offset the faces by the current vertex count
            faces += vertex_offset
            
            # Append to the global list
            all_vertices.append(vertices)
            all_faces.append(faces)
            
            # Update vertex offset
            vertex_offset += len(vertices)
            
            # Free the mesh data after extraction
            obj_eval.to_mesh_clear()
    
    # Combine all vertices and faces into a single array
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)
    
    # Create the combined trimesh object
    combined_trimesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    
    return combined_trimesh



def extract_largest_cluster(point_cloud, eps=0.03, min_samples=30, output_ply='colored_point_cloud_unique.ply'):
    """
    Identify clusters in a point cloud using DBSCAN, assign unique colors to each cluster,
    and export the colored point cloud to a PLY file.

    Parameters:
    - point_cloud (numpy.ndarray): A NumPy array of shape (n_points, 3) representing the point cloud.
    - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
                   Default is 0.05.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
                         Default is 20.
    - output_ply (str): The filename for the output PLY file. Default is 'colored_point_cloud_unique.ply'.

    Returns:
    - largest_cluster_points (numpy.ndarray): A NumPy array containing the points in the largest cluster.
    - colored_pcd (open3d.geometry.PointCloud): The Open3D point cloud object with colors.
    """
    # Validate input
    if not isinstance(point_cloud, np.ndarray):
        raise TypeError("point_cloud must be a NumPy array.")
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError("point_cloud must be of shape (n_points, 3).")

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(point_cloud)

    # Find unique labels (clusters), ignoring noise if present (-1 label)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label if present

    if not unique_labels:
        print("No clusters found.")
        return np.array([]), None

    print(f"Number of clusters found: {len(unique_labels)}")

    # Generate a unique color for each cluster
    # For a large number of clusters, we can generate random colors
    num_clusters = len(unique_labels)
    colors = plt.cm.get_cmap('hsv', num_clusters + 1)  # HSV colormap for distinct colors

    # Assign colors to each cluster
    label_to_color = {label: colors(idx)[:3] for idx, label in enumerate(unique_labels)}

    # Assign colors to points
    point_colors = np.zeros((point_cloud.shape[0], 3))  # Initialize to black
    for label in unique_labels:
        point_colors[labels == label] = label_to_color[label]
    point_colors[labels == -1] = [0, 0, 0]  # Optional: Color noise as black



    # Identify the largest cluster
    largest_cluster_label = None
    max_points = 0
    for label in unique_labels:
        num_points = np.sum(labels == label)
        if num_points > max_points:
            max_points = num_points
            largest_cluster_label = label

    # Extract points belonging to the largest cluster
    largest_cluster_points = point_cloud[labels == largest_cluster_label]
    print(f"Largest cluster label: {largest_cluster_label} with {max_points} points.")

    return largest_cluster_points#, pcd



def sample_filter_reconstruct(
    mesh: trimesh.Trimesh,
    input_points: np.ndarray,
    distance_threshold: float = 0.03,
    num_samples: int = 200000,
    poisson_depth: int = 7,
) -> o3d.geometry.TriangleMesh:

    """
    Samples points from the mesh, filters them based on distance to input point cloud,
    computes normals, and performs Poisson reconstruction.

    Parameters:
    - mesh (trimesh.Trimesh): The input mesh.
    - input_points (np.ndarray): The input point cloud as an (N, 3) NumPy array.
    - distance_threshold (float): Maximum allowed distance from sampled points to input point cloud.
    - num_samples (int): Number of points to sample from the mesh surface.
    - poisson_depth (int): Depth parameter for Poisson reconstruction.

    Returns:
    - reconstructed_mesh (open3d.geometry.TriangleMesh): The mesh obtained from Poisson reconstruction.
    """

    # 1. Sample points on the mesh surface
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    print(f"Sampled {len(sampled_points)} points from the mesh surface.")


    input_points = extract_largest_cluster(input_points)
    # Build KDTree for efficient nearest neighbor search
    kdtree = KDTree(input_points)
    print("Built KDTree for input point cloud.")

    # Query the nearest distance for each sampled point
    distances, _ = kdtree.query(sampled_points, k=1)
    print("Computed distances from sampled points to input point cloud.")

    # Create a mask for points within the distance threshold
    within_threshold_mask = distances <= distance_threshold
    filtered_points = sampled_points[within_threshold_mask]
    filtered_face_indices = face_indices[within_threshold_mask]

    ####refilter to 50000 points, first extract submesh, then sample 
    filtered_face_indices = np.unique(filtered_face_indices)
    submesh = mesh.submesh([filtered_face_indices] , append=True)
    filtered_points, filtered_face_indices = trimesh.sample.sample_surface(submesh, sub_samples)



    print(f"Filtered down to {len(filtered_points)} points within distance threshold {distance_threshold}.")



    # 3. Calculate normals using face normals
    # Retrieve face normals for the filtered points
    face_normals = submesh.face_normals[filtered_face_indices]
    # Ensure normals are unit vectors
    normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    # 4. Create an Open3D PointCloud object and Perform Poisson reconstruction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    print("Created Open3D PointCloud with normals.")

    a = time.time()
    print("Performing Poisson reconstruction...",poisson_depth)
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    print("Poisson reconstruction completed. Total time:", time.time() - a)


    vertices = np.asarray(poisson_mesh.vertices)
    faces = np.asarray(poisson_mesh.triangles)

    # Create a trimesh object from the vertices and faces
    reconstructed_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    #reconstructed_trimesh.export("reconstructed.obj")
    components = reconstructed_trimesh.split(only_watertight=False)
    largest_mesh = max(components, key=lambda c: len(c.faces))

    ####
    face_centers =largest_mesh.triangles_center  # Shape: (num_faces, 3)
    
    # Step 2: Build a KD-Tree from the input points for efficient distance querying
    tree = cKDTree(input_points)
    
    # Step 3: For each face center, check if it's within the distance_threshold from any input point
    # The 'query_ball_point' method returns a list of lists, where each sublist contains the indices
    # of input points within the threshold distance from the corresponding face center.
    # We only need to know if the sublist is non-empty.
    nearby_faces_mask = tree.query_ball_point(face_centers, r=distance_threshold, return_length=True) > 0
    
    # Step 4: Extract the indices of faces that are close to any input point
    nearby_face_indices = np.nonzero(nearby_faces_mask)[0]

    # Step 5: Extract the submesh consisting of the nearby faces
    # The 'submesh' method expects a list of face index lists. Since we're extracting a single group
    # of faces, we wrap 'nearby_face_indices' in another list.
    submesh = largest_mesh.submesh([nearby_face_indices], append=True, repair=True)
    return submesh



def extract_outside_portion_with_buffer(mesh, boundary_vertices, buffer_distance=0.1):
    """
    Extract the portion of a planar mesh that lies outside a defined boundary,
    retaining a buffer zone around the boundary.

    Parameters:
    - mesh (trimesh.Trimesh): The input planar mesh lying on z=0.
    - boundary_vertices (numpy.ndarray): Array of shape (N, 3) defining boundary vertices on z=0.
    - buffer_distance (float): The width of the buffer zone around the boundary polygon.

    Returns:
    - outside_mesh (trimesh.Trimesh): The mesh containing only the outside portion with buffer zone retained.
    """
    # Validate inputs
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The 'mesh' parameter must be a trimesh.Trimesh object.")
    
    if not isinstance(boundary_vertices, np.ndarray):
        raise TypeError("The 'boundary_vertices' must be a NumPy array.")
    
    if boundary_vertices.shape[1] != 3:
        raise ValueError("The 'boundary_vertices' array must have shape (N, 3).")
    
    if len(boundary_vertices) < 3:
        raise ValueError("At least three boundary vertices are required to define a polygon.")
    
    # Ensure all boundary vertices lie on z=0 within a tolerance
    boundary_z = boundary_vertices[:, 2]
    if not np.allclose(boundary_z, 0, atol=1e-6):
        raise ValueError("All boundary vertices must lie on the z=0 plane.")
    
    # Extract 2D coordinates (x, y) of the boundary vertices
    boundary_coords_2d = boundary_vertices[:, :2]
    
    # Create a Shapely polygon from the boundary coordinates
    boundary_polygon = Polygon(boundary_coords_2d)
    
    if not boundary_polygon.is_valid:
        boundary_polygon = boundary_polygon.buffer(0)  # Attempt to fix invalid polygon
        if not boundary_polygon.is_valid:
            raise ValueError("The boundary vertices do not form a valid polygon.")
    
    # Prepare the polygon for faster spatial operations
    prepared_polygon = prep(boundary_polygon)
    
    # Create a buffer zone around the boundary polygon
    # Buffer outward by buffer_distance to retain a ring around the boundary
    buffer_polygon = boundary_polygon.buffer(-buffer_distance)
    
    # Define the final polygon to determine which faces to remove
    # We want to remove faces strictly inside the boundary polygon
    # Retain faces in the buffer zone and outside the boundary
    # No need to subtract buffer_polygon from boundary_polygon as buffer is outward
    
    # Extract all face centroids
    face_centroids = mesh.triangles_center  # Shape: (n_faces, 3)
    
    # Project centroids to 2D (x, y)
    centroids_2d = face_centroids[:, :2]
    
    # Function to determine if a point is inside the boundary polygon
    def is_inside_boundary(xy):
        return buffer_polygon.contains(Point(xy))
    
    # Vectorized point-in-polygon test using list comprehension for efficiency
    inside_flags = np.array([is_inside_boundary(xy) for xy in centroids_2d])
    
    # Identify faces to remove: those inside the boundary polygon
    faces_to_remove = inside_flags
    
    # Identify faces to keep: those outside the boundary polygon (including buffer zone)
    faces_outside = mesh.faces[~faces_to_remove]
    
    if len(faces_outside) == 0:
        print("No faces found outside the boundary.")
        return trimesh.Trimesh()  # Return an empty mesh
    
    # Create the outside mesh with retained buffer zone
    outside_mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                                   faces=faces_outside,
                                   process=False)
    
    # Remove unreferenced vertices to clean up the mesh
    outside_mesh.remove_unreferenced_vertices()
    
    return outside_mesh


def preprocess_mask(mesh,points_in_world):

    poisson_mesh = sample_filter_reconstruct(mesh, points_in_world,distance_threshold)
    a = time.time()
    boundary_loop, boundary_target_verts,plane_verts,plane_faces = boundary_vertices_projection(poisson_mesh)

    print("done boundary ",time.time()-a)
    verts = np.array(poisson_mesh.vertices)
    faces = np.array(poisson_mesh.faces)
    plane_verts,verts,boundary_target_verts,points_in_world = align_plane_to_xy(plane_verts,verts,boundary_target_verts,points_in_world)
    plane_mesh = trimesh.Trimesh(vertices=plane_verts, faces=plane_faces)

    a = time.time()
    plane_mesh = extract_outside_portion_with_buffer(plane_mesh, boundary_target_verts)
    print("done extract outside",time.time()-a)
    boundary_loop = boundary_loop.reshape(1,-1).tolist()
    boundary_target_verts = boundary_target_verts.reshape(1,-1,3).tolist()
    poisson_blending3(verts, faces, boundary_loop, boundary_target_verts, force_correspondence=False)
    print("done blending",time.time()-a)
    modified_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return modified_mesh,plane_mesh,points_in_world, poisson_mesh











def normalize_trimesh(mesh):
    bounding_box = mesh.bounding_box.bounds  # [min_bound, max_bound]
    # Calculate the current center and size of the bounding box
    center = (bounding_box[0] + bounding_box[1]) / 2
    size = bounding_box[1] - bounding_box[0]
    
    # Scale factor to normalize the mesh to a [-1, 1] bounding box
    scale = 1.0 / np.max(size)  # largest dimension scales to 2 units
    
    # Translate to origin and scale
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    
    return mesh










def rotate_trimesh(mesh, angle_x=0.0, angle_y=0.0, angle_z=0.0):
    """
    Rotate a trimesh object around the x, y, and z axes by specified angles.

    Parameters:
    mesh (trimesh.Trimesh): The mesh to rotate.
    angle_x (float): Rotation angle around the x-axis in degrees.
    angle_y (float): Rotation angle around the y-axis in degrees.
    angle_z (float): Rotation angle around the z-axis in degrees.

    Returns:
    trimesh.Trimesh: The rotated mesh.
    """
    # Convert angles from degrees to radians
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)
    angle_z_rad = np.radians(angle_z)

    # Create rotation matrices for each axis
    rotation_matrix_x = trimesh.transformations.rotation_matrix(
        angle_x_rad, [1, 0, 0]
    )
    rotation_matrix_y = trimesh.transformations.rotation_matrix(
        angle_y_rad, [0, 1, 0]
    )
    rotation_matrix_z = trimesh.transformations.rotation_matrix(
        angle_z_rad, [0, 0, 1]
    )

    # Combine the rotations: apply x, then y, then z
    rotation_matrix = trimesh.transformations.concatenate_matrices(
        rotation_matrix_z, rotation_matrix_y, rotation_matrix_x
    )

    # Apply the combined rotation to the mesh
    mesh.apply_transform(rotation_matrix)

    return mesh

def load_voxel(filename):
    label_filename = os.path.join(voxel_label_dir, filename + "_labels.npz")
    voxel_model_file = os.path.join(voxel_dir, filename + ".binvox")
    npz_file = np.load(label_filename)
    indices = npz_file['indices']
    offsets = npz_file['offsets']
    selected_label = []
    for i in range(len(offsets) - 1):
        start = offsets[i]
        end = offsets[i+1]
        selected_label.append(indices[start:end].tolist())
    #     print(i,indices[start:end].tolist())

    # exit(0)
    f =  open(voxel_model_file, 'rb')
    batch_voxels = binvox_rw.read_as_3d_array(f).data.astype(bool)
    f.close()


    px, pz, py = np.nonzero(batch_voxels)
    voxel_idx = np.stack([px, py , pz], axis=1)
    voxel_pc = ((voxel_idx.astype(np.float32) + 0.5) / 1024 - 0.5 )* 2
    voxel_pc[:,1] = voxel_pc[:,1] * -1
    del batch_voxels


    return voxel_idx, voxel_pc, selected_label   ###X2 here?




from collections import defaultdict, deque
import heapq

def dijkstra_shortest_path(adj_list, start_vertex, end_vertex, num_vertices):
    """
    Implements Dijkstra's algorithm to find the shortest path between two vertices.
    
    Parameters:
        adj_list (dict): Adjacency list representing the graph.
        start_vertex (int): The starting vertex index.
        end_vertex (int): The ending vertex index.
        num_vertices (int): Total number of vertices in the graph.
        
    Returns:
        path (list): List of vertex indices representing the shortest path from start to end.
    """
    import heapq
    distances = [float('inf')] * num_vertices
    previous = [None] * num_vertices
    distances[start_vertex] = 0
    queue = [(0, start_vertex)]

    while queue:
        dist_u, u = heapq.heappop(queue)
        if u == end_vertex:
            break
        for v in adj_list[u]:
            alt = distances[u] + 1  # Assuming all edges have weight 1
            if alt < distances[v]:
                distances[v] = alt
                previous[v] = u
                heapq.heappush(queue, (alt, v))

    # Reconstruct path
    path = []
    u = end_vertex
    if previous[u] is not None or u == start_vertex:
        while u is not None:
            path.insert(0, u)
            u = previous[u]

    return path


def count_connected_components(dims, voxels):
    """
    Counts the number of connected components in a 3D voxel grid.

    Parameters:
    - dims: Tuple of ints (dimx, dimy, dimz)
    - voxels: List of lists, each sublist is [x, y, z] of active voxel

    Returns:
    - num_components: Integer, number of connected components
    - connected_components: List of connected components, each is a list of [x, y, z]
    """
    # Initialize the set of unvisited voxels
    unvisited = set(tuple(voxel) for voxel in voxels)
    
    connected_components = []
    num_components = 0

    while unvisited:
        # Select a random seed voxel from unvisited
        seed = random.choice(list(unvisited))
        
        # Perform flood fill
        connected = flood_fill_cython(dims, list(unvisited), list(seed))
        
        if connected:
            num_components += 1
            connected_components.append(connected)
            # Remove connected voxels from unvisited
            unvisited -= set(tuple(voxel) for voxel in connected)
        else:
            # If flood fill returns empty, remove the seed to prevent infinite loop
            unvisited.remove(seed)

    return num_components, connected_components

def remove_neighboring_voxels(X, Y, delta=5):
    """
    Remove from X all voxels that are within (x±delta, y±delta, z±delta) of any Y voxel.

    Parameters:
    - X: array-like, shape (N, 3), list of X voxels as [x, y, z].
    - Y: array-like, shape (M, 3), list of Y voxels as [x, y, z].
    - delta: int, defines the neighborhood range in each dimension (default=5).

    Returns:
    - X_filtered: NumPy array of X voxels with neighboring voxels removed.
    """
    # Convert X and Y to NumPy arrays if they aren't already
    X = np.asarray(X, dtype=np.int32).copy()
    Y = np.asarray(Y, dtype=np.int32).copy()

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be a 2D array with shape (N, 3).")
    if Y.ndim != 2 or Y.shape[1] != 3:
        raise ValueError("Y must be a 2D array with shape (M, 3).")

    print(f"Original number of voxels in X: {X.shape[0]}")
    print(f"Number of Y voxels: {Y.shape[0]}")

    # Step 1: Generate all relative offsets within the delta range
    print("Generating relative offsets...")
    offsets = np.array([[dx, dy, dz] 
                        for dx in range(-delta, delta + 1) 
                        for dy in range(-delta, delta + 1) 
                        for dz in range(-delta, delta + 1)], dtype=np.int32)
    print(f"Total relative offsets generated: {offsets.shape[0]}")

    # Step 2: Expand Y voxels by adding relative offsets
    print("Expanding Y voxels with relative offsets...")
    # Using broadcasting to add offsets to each Y voxel
    # Y[:, np.newaxis, :] has shape (M, 1, 3)
    # offsets[np.newaxis, :, :] has shape (1, K, 3)
    # Resulting neighbor_Y has shape (M, K, 3)
    neighbor_Y = Y[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    neighbor_Y = neighbor_Y.reshape(-1, 3)  # Shape (M*K, 3)

    print(f"Total neighbor voxels generated: {neighbor_Y.shape[0]}")

    # Step 3: Convert neighbor_Y to a set of tuples for efficient lookup
    print("Converting neighbor voxels to set...")
    neighbor_Y_set = set(map(tuple, neighbor_Y))
    print(f"Total unique neighbor voxels: {len(neighbor_Y_set)}")

    # Step 4: Convert X to a set of tuples
    print("Converting X voxels to set...")
    X_set = set(map(tuple, X))
    print(f"Total X voxels before removal: {len(X_set)}")

    # Step 5: Perform set difference to remove neighboring voxels
    print("Removing neighboring voxels from X...")
    X_filtered_set = X_set - neighbor_Y_set
    print(f"Total X voxels after removal: {len(X_filtered_set)}")

    # Step 6: Convert the set back to a NumPy array
    print("Converting filtered X back to NumPy array...")
    X_filtered = np.array(list(X_filtered_set), dtype=np.int32)

    return X_filtered

def convert_to_pc(voxel):
    voxel_pc = (voxel.astype(np.float32) + 0.5) / 1024 - 0.5
    voxel_pc  = voxel_pc  * 2
    voxel_pc[:,1] = -voxel_pc[:,1]
    #voxel_pc  = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])[:3, :3].dot(voxel_pc.T).T

    return voxel_pc


def remove_consecutive_duplicates_diff(arr):
    # Ensure input is a NumPy array
    arr = np.asarray(arr)
    
    if arr.size == 0:
        return arr  # Return empty array as is

    # Compute the difference between consecutive elements
    diff = np.diff(arr)
    
    # Identify where the difference is not zero
    mask = diff != 0
    
    # Select the elements where mask is True, plus the first element
    return np.concatenate(([arr[0]], arr[1:][mask]))

def remove_inner_loops(index_list):
    # Dictionary to store the first occurrence index of each element
    first_occurrence = {}
    # Set to keep track of elements to retain
    retain = set()
    
    for i, elem in enumerate(index_list):
        if elem not in first_occurrence:
            first_occurrence[elem] = i
        else:
            # When a duplicate is found, mark the elements to be retained
            retain.update(index_list[first_occurrence[elem]:i+1])
    
    # Always retain the first and last elements to ensure the outer loop
    retain.add(index_list[0])
    retain.add(index_list[-1])
    
    # Construct the processed list by retaining elements in their original order
    processed_list = []
    seen = set()
    for elem in index_list:
        if elem in retain and (elem not in seen or elem == index_list[-1]):
            processed_list.append(elem)
            seen.add(elem)
    
    return np.array(processed_list)


def inside_out(mesh,points):
    verts = np.array(mesh.vertices, order='F')
    faces = np.array(mesh.faces, order='F')
    points = np.array(points, order='F')
    winding_numbers = igl.winding_number(verts, faces, points)
    threshold = 0.5
    inside = winding_numbers >= threshold
    outside = winding_numbers < threshold
    points_outside = points[outside]
    return points_outside, outside

def filter_voxel(voxels_pc,voxels,mesh):
    num_sample_points = 500000
    sample_points,face_index = trimesh.sample.sample_surface(mesh, num_sample_points)

    sample_points, _ = inside_out(mesh,sample_points)

    kdtree = KDTree(sample_points)
    distances, index = kdtree.query(voxels_pc, k=1)

    #trimesh.Trimesh(vertices=sample_points, faces=None).export("outside_points_ref.obj")

    idx = distances <0.002
    voxels_pc_outside = voxels_pc[idx]
    voxel_outside = voxels[idx]


    #trimesh.Trimesh(vertices = voxels_pc_outside, faces = None).export("outside_points_pc.obj")
    return voxel_outside,voxels_pc_outside



def sample_filter_reconstruct2(
    mesh: trimesh.Trimesh,
    input_points: np.ndarray,
    pre_distance_threshold: float = 0.01,
    after_distance_threshold: float = 0.01,
    num_samples: int = 200000,
    poisson_depth: int = 7,
) -> o3d.geometry.TriangleMesh:

    """
    Samples points from the mesh, filters them based on distance to input point cloud,
    computes normals, and performs Poisson reconstruction.

    Parameters:
    - mesh (trimesh.Trimesh): The input mesh.
    - input_points (np.ndarray): The input point cloud as an (N, 3) NumPy array.
    - distance_threshold (float): Maximum allowed distance from sampled points to input point cloud.
    - num_samples (int): Number of points to sample from the mesh surface.
    - poisson_depth (int): Depth parameter for Poisson reconstruction.

    Returns:
    - reconstructed_mesh (open3d.geometry.TriangleMesh): The mesh obtained from Poisson reconstruction.
    """

    # trimesh.Trimesh(vertices=input_points).export("input_points.obj")
    # exit(0)
    # 1. Sample points on the mesh surface
    # sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    # print(f"Sampled {len(sampled_points)} points from the mesh surface.")


    # input_points = extract_largest_cluster(input_points)
    # # Build KDTree for efficient nearest neighbor search
    # kdtree = KDTree(input_points)
    # print("Built KDTree for input point cloud.")

    # # Query the nearest distance for each sampled point
    # distances, _ = kdtree.query(sampled_points, k=1)
    # print("Computed distances from sampled points to input point cloud.")

    # # Create a mask for points within the distance threshold
    # within_threshold_mask = distances <= distance_threshold
    # filtered_points = sampled_points[within_threshold_mask]
    # filtered_face_indices = face_indices[within_threshold_mask]

    # #mesh.export("this_mesh.obj")
    # trimesh.Trimesh(vertices=filtered_points, faces=None).export("filtered_points_before.obj")
    # filtered_points, filter_indices = inside_out(mesh,filtered_points)
    # trimesh.Trimesh(vertices=filtered_points, faces=None).export("filtered_points_after.obj")
    # filtered_face_indices = filtered_face_indices[filter_indices]
    # ####refilter to 50000 points, first extract submesh, then sample 
    # filtered_face_indices = np.unique(filtered_face_indices)
    # submesh = mesh.submesh([filtered_face_indices] , append=True)
    # print("submesh",submesh,filtered_face_indices.shape)
    _,_,filtered_face_indices = trimesh.proximity.closest_point(mesh,input_points)
    submesh = mesh.submesh([filtered_face_indices] , append=True)
    submesh.export("submesh.obj")
    filtered_points, filtered_face_indices = trimesh.sample.sample_surface(submesh, sub_samples)
    filtered_points, filter_indices = inside_out(mesh,filtered_points)
    filtered_face_indices = filtered_face_indices[filter_indices]
    tree = cKDTree(input_points)
    nearby_faces_mask = tree.query_ball_point(filtered_points, r=pre_distance_threshold, return_length=True) > 0
    filtered_points = filtered_points[nearby_faces_mask]
    filtered_face_indices = filtered_face_indices[nearby_faces_mask]



    print(f"Filtered down to {len(filtered_points)} points within distance threshold {distance_threshold}.")



    # 3. Calculate normals using face normals
    # Retrieve face normals for the filtered points
    face_normals = submesh.face_normals[filtered_face_indices]
    # Ensure normals are unit vectors
    normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, np.newaxis]

    trimesh.Trimesh(vertices=filtered_points, faces=None).export("poisson_points.obj")
    # 4. Create an Open3D PointCloud object and Perform Poisson reconstruction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    print("Created Open3D PointCloud with normals.")

    a = time.time()
    print("Performing Poisson reconstruction...",poisson_depth)
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    print("Poisson reconstruction completed. Total time:", time.time() - a)


    vertices = np.asarray(poisson_mesh.vertices)
    faces = np.asarray(poisson_mesh.triangles)

    # Create a trimesh object from the vertices and faces
    reconstructed_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    reconstructed_trimesh.export("reconstructed.obj")
    components = reconstructed_trimesh.split(only_watertight=False)
    largest_mesh = max(components, key=lambda c: len(c.faces))

    ####
    face_centers =largest_mesh.triangles_center  # Shape: (num_faces, 3)
    
    # Step 2: Build a KD-Tree from the input points for efficient distance querying
    tree = cKDTree(filtered_points)
    
    # Step 3: For each face center, check if it's within the distance_threshold from any input point
    # The 'query_ball_point' method returns a list of lists, where each sublist contains the indices
    # of input points within the threshold distance from the corresponding face center.
    # We only need to know if the sublist is non-empty.
    nearby_faces_mask = tree.query_ball_point(face_centers, r=after_distance_threshold, return_length=True) > 0
    
    # Step 4: Extract the indices of faces that are close to any input point
    nearby_face_indices = np.nonzero(nearby_faces_mask)[0]

    # Step 5: Extract the submesh consisting of the nearby faces
    # The 'submesh' method expects a list of face index lists. Since we're extracting a single group
    # of faces, we wrap 'nearby_face_indices' in another list.
    submesh = largest_mesh.submesh([nearby_face_indices], append=True, repair=True)

    components = submesh.split(only_watertight=False)
    submesh = max(components, key=lambda c: len(c.faces))
    submesh.fill_holes()
    return submesh


def align_vertices_faces_with_z(vertices,faces):
    # Compute face normals using NumPy
    # For each face, get the three vertex positions
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute the two edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Compute the cross product (face normals)
    face_normals = np.cross(edge1, edge2)

    # Normalize the face normals
    norms = np.linalg.norm(face_normals, axis=1)
    
    # Handle zero-length normals to avoid division by zero
    zero_norms = norms == 0
    if np.any(zero_norms):
        print(f"Warning: {np.sum(zero_norms)} faces have zero area and will be ignored in normal averaging.")
        # To prevent division by zero, set zero normals to a default value (e.g., [0,0,1])
        face_normals[zero_norms] = np.array([0, 0, 1])
        norms[zero_norms] = 1.0  # Avoid division by zero

    normalized_face_normals = face_normals / norms[:, np.newaxis]

    # Calculate the average normal
    average_normal = normalized_face_normals.mean(axis=0)
    norm_avg = np.linalg.norm(average_normal)
    if norm_avg == 0:
        raise ValueError("Average normal has zero length; cannot determine alignment.")
    average_normal /= norm_avg  # Normalize to unit vector

    # Define the positive Z-axis
    z_positive = np.array([0, 0, 1])

    # Compute the dot product with the positive Z-axis
    dot_product = np.dot(average_normal, z_positive)

    # Debug statements (optional)
    print(f"Average Normal: {average_normal}")
    print(f"Dot Product with Z+: {dot_product}")

    # Initialize output as input
    flipped_vertices = vertices.copy()
    flipped_faces = faces.copy()

    # Threshold to determine alignment (optional)
    # You can adjust this threshold as needed
    threshold = 1e-6

    # If the average normal is more aligned with negative Z-axis, flip the mesh
    if dot_product < threshold:
        print("Average normal is aligned with Z+. Flipping the mesh by inverting Z coordinates.",dot_product)
        # Flip the z-coordinates of all vertices
        flipped_vertices[:, 2] *= -1

        # Reverse the winding order of each face to maintain correct normals
        flipped_faces = flipped_faces[:, ::-1]

    elif dot_product > threshold:
        print("Average normal is aligned with Z-. No flipping needed.",dot_product)

    return flipped_vertices, flipped_faces

def generate_plane_mesh(loop_vertices):
    """
    Generates a plane mesh connecting the given loop to the square boundary X,Y = ±1 in the Z=0 plane.

    Parameters:
    ----------
    loop_vertices : np.ndarray
        A NumPy array of shape [N, 3] representing the loop vertices in the Z=0 plane.

    Returns:
    -------
    mesh : trimesh.Trimesh
        The generated mesh connecting the loop to the boundary.
    """


    # Close the loop if it's not closed
    if not np.array_equal(loop_vertices[0], loop_vertices[-1]):
        loop_vertices = np.vstack([loop_vertices, loop_vertices[0]])
    
    # Define the outer square boundary at X,Y = ±1
    boundary = np.array([
        [-1, -1, 0],
        [ 1, -1, 0],
        [ 1,  1, 0],
        [-1,  1, 0],
        [-1, -1, 0]  # Close the boundary loop
    ])
    
    # Create Shapely polygons for outer boundary and inner loop
    outer_polygon = Polygon(boundary[:, :2])
    inner_polygon = Polygon(loop_vertices[:, :2])
    
    #print(loop_vertices.shape)
    #trimesh.Trimesh(vertices=loop_vertices).export("loop_vertices.obj")
    # Validate polygons
    # if not outer_polygon.is_valid:
    #     raise ValueError("The outer boundary polygon is invalid.")
    # if not inner_polygon.is_valid:
    #     raise ValueError("The inner loop polygon is invalid.")
    # if not outer_polygon.contains(inner_polygon):
    #     raise ValueError("The inner loop must be entirely within the outer boundary.")
    
    # Create a polygon with a hole (the loop)
    polygon_with_hole = Polygon(outer_polygon.exterior.coords, [inner_polygon.exterior.coords])
    
    try:
        vertices,faces = trimesh.creation.triangulate_polygon(polygon_with_hole)
    except Exception as e:
        raise RuntimeError(f"Triangulation failed: {e}")
    

    # Ensure the mesh is in the Z=0 plane
    vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1))])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def pymeshlab_smoothing(verts,faces):
    ms = pymeshlab.MeshSet()
    #print(ms.apply_coord_laplacian_smoothing)
    ms.add_mesh(pymeshlab.Mesh(verts, faces), "input_mesh")
    ms.apply_coord_laplacian_smoothing(stepsmoothnum=smooth_iterations,cotangentweight=True,boundary = True)
    verts = np.array(ms.current_mesh().vertex_matrix())
    faces = np.array(ms.current_mesh().face_matrix())
    return verts,faces

def filter_voxel_B(A, B):
    # Convert A to a set of tuples for efficient lookup
    set_A = set(map(tuple, A))
    
    # Filter B by retaining only voxels that are also in A, maintaining the order
    filtered_B = np.array([voxel for voxel in B if tuple(voxel) in set_A])
    
    return filtered_B


from collections import deque

def get_neighbors(voxel):
    """
    Get 6-connected neighbors of a voxel.

    Parameters:
    - voxel: tuple (x, y, z)

    Returns:
    - List of neighboring voxels
    """
    x, y, z = voxel
    return [
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1)
    ]

def bfs_shortest_path(A_set, start, goal):
    """
    Perform BFS to find the shortest path from start to goal within A_set.

    Parameters:
    - A_set: set of tuples representing voxel A
    - start: tuple (x, y, z)
    - goal: tuple (x, y, z)

    Returns:
    - path: list of tuples representing the path from start to goal
            Returns None if no path exists
    """
    if start == goal:
        return [start]
    
    queue = deque([start])
    visited = {start}
    predecessor = {start: None}
    
    while queue:
        current = queue.popleft()
        
        for neighbor in get_neighbors(current):
            if neighbor in A_set and neighbor not in visited:
                visited.add(neighbor)
                predecessor[neighbor] = current
                queue.append(neighbor)
                
                if neighbor == goal:
                    # Reconstruct path
                    path = [goal]
                    while predecessor[path[-1]] is not None:
                        path.append(predecessor[path[-1]])
                    path.reverse()
                    return path
    return None  # No path found

def form_new_loop_BFS(A, filtered_B):
    """
    Form a new loop by connecting filtered B voxels using BFS on A.

    Parameters:
    - A: numpy array of shape [N, 3]
    - filtered_B: numpy array of shape [K, 3]

    Returns:
    - new_loop: list of tuples representing the new loop
                Returns None if loop formation fails
    """
    A_set = set(map(tuple, A))
    B_list = [tuple(v) for v in filtered_B]
    
    if len(B_list) < 2:
        print("Not enough voxels in B to form a loop.")
        return None
    
    new_loop = []
    
    for i in range(len(B_list)):
        start = B_list[i]
        end = B_list[(i + 1) % len(B_list)]  # Wrap around for loop closure
        
        path = bfs_shortest_path(A_set, start, end)
        # path_np = np.array(path)
        # cut_voxels = convert_to_pc(path_np)
        # trimesh.Trimesh(vertices=cut_voxels).export(f"./cut_voxel.obj")
        # print(path)
        # exit(0)
        if path is None:
            print(f"No path found between {start} and {end}.")
            return None
        
        if i == 0:
            new_loop.extend(path)
        else:
            # Avoid duplicating the start voxel of the current path
            new_loop.extend(path[1:])
    
    new_loop  = np.array(new_loop)
    return new_loop


def mesh_smooth(verts, faces,boundary_verts):
    ms1 = pymeshlab.MeshSet()
    ms1.add_mesh(pymeshlab.Mesh(verts, faces), "original_mesh")
    ###try to find the selected_vertices
    remeshed = trimesh.Trimesh(vertices=verts, faces=faces)
    centroids = np.array(remeshed.triangles_center)
    tree = cKDTree(centroids)
    indices = tree.query_ball_point(boundary_verts, smoothing_threshold)

    # selected_faces = np.unique([idx for sublist in indices for idx in sublist])
    # print("Total selected face for smoothing", len(selected_faces))
    # condition2 = " || ".join([f"fi == {i}" for i in selected_faces])
    # ms1.apply_filter('compute_selection_by_condition_per_face', condselect=condition2)
    # ms1.apply_coord_two_steps_smoothing(stepsmoothnum = 10, normalthr = 90.000000)
    counter = 0
    accumulate_indices = []
    for i in indices:
        accumulate_indices.extend(i)
        accumulate_unique = np.unique(np.array(accumulate_indices))
        if accumulate_unique.shape[0] > 500:
            counter += 1
            print(counter, accumulate_unique.shape[0])
            accumulate_indices = []
            condition2 = " || ".join([f"fi == {i}" for i in accumulate_unique])
            ms1.apply_filter('compute_selection_by_condition_per_face', condselect=condition2)
            ms1.apply_coord_two_steps_smoothing(stepsmoothnum = 5, normalthr = 180.000, stepfitnum = 5)


    processed_mesh = ms1.current_mesh()
    updated_vertices = np.array(processed_mesh.vertex_matrix())
    updated_faces = np.array(processed_mesh.face_matrix())
    return updated_vertices,updated_faces


def augmentation2(instance_mesh, boundary_loop, scale_limits=(1.1, 1.8), position_limit=0.8, verbose=False):


    # Convert vertices to a NumPy array for manipulation
    verts = np.array(instance_mesh.vertices)

    # -----------------------------
    # 1. Random Rotation in X-Y Plane
    # -----------------------------

    # Generate a random rotation angle between 0 and 2*pi radians
    rotation_angle = np.random.uniform(0, 2 * np.pi)

    # Create a rotation matrix for rotation about the Z-axis (i.e., in the x-y plane)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
        [0,                      0,                     1]
    ])

    # Apply rotation to all vertices
    verts = verts.dot(rotation_matrix.T)

    if verbose:
        print(f"Rotation Angle (radians): {rotation_angle}")
        print(f"Rotation Matrix:\n{rotation_matrix}")

    # -----------------------------
    # 2. Scaling
    # -----------------------------

    # Compute the overall size of the mesh
    max_coords = np.max(verts, axis=0)
    min_coords = np.min(verts, axis=0)
    ranges = max_coords - min_coords
    max_range = np.max(ranges)

    if max_range == 0:
        raise ValueError("The mesh has zero size; cannot scale.")

    # Randomly scale the mesh within the specified limits
    scaling_factor_small = scale_limits[0] / max_range
    scaling_factor_large = scale_limits[1] / max_range
    scaling_factor = np.random.uniform(scaling_factor_small, scaling_factor_large)
    verts *= scaling_factor

    if verbose:
        print(f"Scaling Factor: {scaling_factor}")

    # -----------------------------
    # 3. Centering the Mesh
    # -----------------------------

    # Extract boundary vertices after rotation and scaling
    boundary_verts = verts[boundary_loop]

    # Compute center of boundary vertices in the x-y plane
    max_boundary = np.max(boundary_verts, axis=0)[:2]
    min_boundary = np.min(boundary_verts, axis=0)[:2]
    center = (max_boundary + min_boundary) / 2

    # Translate vertices to center the boundary in the x-y plane
    verts[:, :2] -= center

    if verbose:
        print(f"Center of Boundary (x, y): {center}")

    # -----------------------------
    # 4. Translational Augmentation
    # -----------------------------

    # Re-extract boundary vertices after centering
    boundary_verts = verts[boundary_loop]

    # Compute new max and min coordinates in x and y after centering
    max_coords = np.max(boundary_verts, axis=0)[:2]
    min_coords = np.min(boundary_verts, axis=0)[:2]

    # Calculate translation limits to keep the mesh within the positional boundary
    x_neg_lim = -position_limit - min_coords[0]
    x_pos_lim = position_limit - max_coords[0]
    y_neg_lim = -position_limit - min_coords[1]
    y_pos_lim = position_limit - max_coords[1]

    # Generate random translation values within the calculated limits
    x_move = np.random.uniform(x_neg_lim, x_pos_lim)
    y_move = np.random.uniform(y_neg_lim, y_pos_lim)
    translation = np.array([x_move, y_move])

    # Apply translation to the x and y coordinates
    verts[:, :2] += translation

    if verbose:
        print(f"Translation in x: {x_move}")
        print(f"Translation in y: {y_move}")

    # -----------------------------
    # 5. Create the Augmented Mesh
    # -----------------------------

    # Construct the augmented mesh with the transformed vertices and original faces
    augmented_mesh = trimesh.Trimesh(vertices=verts, faces=instance_mesh.faces, process=False)

    return augmented_mesh


def merge_meshes(instance_mesh, plane_mesh, boundary_loop):
    boundary_loop = boundary_loop.copy()[:-1]
    instance_verts = np.array(instance_mesh.vertices)
    boundary_verts = instance_verts[boundary_loop].tolist().copy()
    print(len(boundary_verts), len(boundary_loop))
    boundary_dict = {tuple(j)[:2]:boundary_loop[i] for i,j in enumerate(boundary_verts)}

    plane_verts = np.array(plane_mesh.vertices)
    plane_faces = np.array(plane_mesh.faces)
    instance_faces = np.array(instance_mesh.faces)

    trimesh.Trimesh(vertices=instance_verts, faces=instance_faces).export('insp.obj')
    map_dict = {}
    counter = 0
    not_visited_map = dict()
    not_visited_verts = []
    for i, vert in enumerate(plane_verts):
        # if i < 10:
        #     continue
        # print(tuple(vert.tolist()))
        # exit(0)
        key= tuple(vert.tolist())[:2]
        if key in boundary_dict:
            map_dict[i] = boundary_dict[key]
        else:
            not_visited_verts.append(vert)
            not_visited_map[i] = len(instance_verts) + counter
            counter += 1
        #raise RuntimeError("Vertex not found in boundary")
    print(len(not_visited_verts), "found")
    rest_verts = np.array(not_visited_verts)
    total_verts = np.vstack([instance_verts, rest_verts])
    updated_plane_faces = []
    for face in plane_faces:
        updated_face = []
        for idx in face:
            if idx in map_dict:
                updated_face.append(map_dict[idx])
            else:
                updated_face.append(not_visited_map[idx])
        updated_plane_faces.append(updated_face)
    updated_plane_faces = np.array(updated_plane_faces)
    total_faces = np.vstack([instance_faces, updated_plane_faces])
    merge_mesh = trimesh.Trimesh(vertices=total_verts, faces=total_faces)
    return merge_mesh,np.arange(len(updated_plane_faces)) + len(instance_faces)
    
    
def mesh_remesh(verts,faces, face_indices):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts, faces), "original_mesh")
    # condition = " || ".join([f"fi == {i}" for i in face_indices])
    # print(face_indices.shape)
    # ms.apply_filter('compute_selection_by_condition_per_face', condselect=condition)
    # selected_face_count = ms.current_mesh().selected_face_number()
    # print(f"Selected {selected_face_count} faces for remeshing.")
    ms.meshing_isotropic_explicit_remeshing(targetlen = pymeshlab.PercentageValue(0.75))
    processed_mesh = ms.current_mesh()
    updated_vertices = np.array(processed_mesh.vertex_matrix())
    updated_faces = np.array(processed_mesh.face_matrix())
    return updated_vertices,updated_faces

def cloest_point_attributes(updated_vertices,updated_faces,verts,faces,selected_face_index):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    points, face_indices = mesh.sample(num_samples, return_index=True)
    selected_faces = set(selected_face_index.tolist())
    points_label = np.array([True if i in selected_faces else False for i in face_indices])
    
    tree = KDTree(points)
    distances, indices = tree.query(updated_vertices)
    labels = points_label[indices]
    return labels

def save_ply(filename, verts, faces, vert_attributes):

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    dtype.append((f'vertex_attr', 'i4'))
    vertex_data = np.empty(len(verts), dtype=dtype)
    vertex_data['x'] = verts[:, 0]
    vertex_data['y'] = verts[:, 1]
    vertex_data['z'] = verts[:, 2]
    vertex_data[f'vertex_attr'] = vert_attributes


    # Prepare face data
    face_dtype = [('vertex_indices', 'i4', (3,))]
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces

    # Create PlyElement objects
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')

    # Write to PLY file
    PlyData([vertex_element, face_element], text=True).write(filename)
    print(f"PLY file saved as '{filename}'.")

def process_voxel(mesh,voxels,single_voxel_label):
    ####voxel extraction
    filtered_voxel_surface_loop = filter_voxel_B(voxels, single_voxel_label)
    print("original voxel", single_voxel_label.shape[0],"now voxel",filtered_voxel_surface_loop.shape[0])
    new_loop = form_new_loop_BFS(voxels, filtered_voxel_surface_loop)
    print("new_loop voxel", len(new_loop),"old_loop voxel",len(single_voxel_label))
    # cut_voxels = convert_to_pc(voxels)
    #trimesh.Trimesh(vertices=cut_voxels).export(f"./orig_voxels.obj")
    voxels = remove_neighboring_voxels(voxels,new_loop, delta=3)
    a = time.time()
    # cut_voxels = convert_to_pc(voxels)
    # trimesh.Trimesh(vertices=cut_voxels).export(f"./cut_voxels.obj")
    # cut_voxels = convert_to_pc(single_voxel_label)
    # trimesh.Trimesh(vertices=cut_voxels).export(f"./original_loop.obj")
    # cut_voxels = convert_to_pc(filtered_voxel_surface_loop)
    # trimesh.Trimesh(vertices=cut_voxels).export(f"./surface_loop.obj")
    # cut_voxels = convert_to_pc(new_loop)
    # trimesh.Trimesh(vertices=cut_voxels).export(f"./after_loop.obj")
    # exit(0)
    num_components, connected_components = count_connected_components(dims, voxels)

    print(time.time()-a, "in count_connected_components")
    print(f"Found {num_components} connected components in the voxel grid.")
    # exit(0)
    # sort connected components by size, larger to smaller
    connected_components.sort(key=len, reverse=True)
    for i in range(len(connected_components)):
        print(f"Connected component {i}: {len(connected_components[i])} voxels")
    ###the second largest connected component

    ####3sample filter reconstruct
    a = time.time()
    points_in_world = convert_to_pc(np.array(connected_components[1]))
    #trimesh.Trimesh(vertices=points_in_world).export(f"./points_in_world.obj")
    #mesh.export(f"./basemesh.obj")
    poisson_mesh = sample_filter_reconstruct2(mesh, points_in_world, pre_distance_threshold=0.01,after_distance_threshold=0.01)

    poisson_mesh.export(f"./poisson_mesh.obj")
    exit(0)
    print("sample_filter_reconstruct",time.time()-a)



    ####boundary extraction and alignment
    a = time.time()
    boundary_loop, boundary_target_verts,plane_verts,plane_faces = boundary_vertices_projection(poisson_mesh)
    verts = np.array(poisson_mesh.vertices)
    faces = np.array(poisson_mesh.faces)
    plane_verts,verts,boundary_target_verts,points_in_world = align_plane_to_xy(plane_verts,verts,boundary_target_verts,points_in_world)
    verts,faces = align_vertices_faces_with_z(verts,faces)
    print("done boundary ",time.time()-a)

    #####poisson blending#######
    #trimesh.Trimesh(vertices=verts, faces=faces).export(f"./align_mesh.obj")
    a = time.time()
    boundary_loop = boundary_loop.reshape(1,-1).tolist()
    boundary_target_verts = boundary_target_verts.reshape(1,-1,3).tolist()
    poisson_blending3(verts, faces, boundary_loop, boundary_target_verts, force_correspondence=False)
    #trimesh.Trimesh(vertices=verts, faces=faces).export(f"./blend_mesh.obj")
    print("blending")
    verts,faces = pymeshlab_smoothing(verts,faces)
    instance_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    print("done blending",time.time()-a)

    ####extract plane mesh
    # a = time.time()
    # plane_mesh = generate_plane_mesh(verts[boundary_loop][0])
    # print("done extract outside",time.time()-a)
    # return modified_mesh,plane_mesh,points_in_world, poisson_mesh
    a = time.time()
    augment_meshes = []
    boundary_loop = np.array(boundary_loop)[0]
    for _ in range(repeat):
        print("boundary loop",boundary_loop.shape)
        instance_mesh = augmentation2(instance_mesh,boundary_loop)
        verts = np.array(instance_mesh.vertices)
        faces = np.array(instance_mesh.faces)

        plane_mesh = generate_plane_mesh(verts[boundary_loop])
        merge_mesh,selected_face_index = merge_meshes(instance_mesh, plane_mesh, boundary_loop)

        verts = np.array(merge_mesh.vertices)
        faces = np.array(merge_mesh.faces)
        boundary_verts = verts[boundary_loop].copy()
        updated_vertices,updated_faces = mesh_remesh(verts,faces,selected_face_index)
        updated_attributes = cloest_point_attributes(updated_vertices,updated_faces,verts,faces,selected_face_index)
        verts,faces = mesh_smooth(updated_vertices,updated_faces,boundary_verts)

        augment_meshes.append((verts.copy(),faces.copy(),updated_attributes.copy()))

    print("done plane attach",time.time()-a)
    return augment_meshes




if __name__ == "__main__":

    # 1. Load your config
    config = OmegaConf.load("config/config.yaml")

    # 2. Extract paths
    voxel_dir = config.paths.voxel_dir
    mesh_dir = config.paths.mesh_dir
    voxel_label_dir = config.paths.voxel_label_dir
    result_dir = config.paths.result_dir

    # 3. Extract hyper-parameters
    dims = config.params.dims
    distance_threshold = config.params.distance_threshold
    num_samples = config.params.num_samples
    sub_samples = config.params.sub_samples
    num_sample_points = config.params.num_sample_points
    smooth_iterations = config.params.smooth_iterations
    depth = config.params.depth
    distance = config.params.distance
    max_label = config.params.max_label
    smoothing_threshold = config.params.smoothing_threshold
    repeat = config.params.repeat
    largest_component = config.params.largest_component

    # 4. Make sure your output dir exists
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 5. Create name_list
    name_list = [ i[: i.find("_label")] for i in os.listdir(voxel_label_dir) if i.endswith("_labels.npz") ]
    name_list = [ name for name in name_list if os.path.exists(os.path.join(voxel_dir, name + ".binvox")) ]
    name_list = [ name for name in name_list if os.path.exists(os.path.join(mesh_dir, name[8:] + ".glb")) ]

    print("Total:", len(name_list)) 
    name_list = [name for name in name_list if not glob.glob(f"{result_dir}/{name}_*_arg0.ply")]
    print("Remaining:", len(name_list))

    np.random.seed(42)
    random.seed(42)
    start_time = time.time()
    counter = 0

    for index, name in enumerate(name_list):

        # ----- load voxel
        try:
            voxel_idx, voxel_pc, selected_label = load_voxel(name)
        except Exception as e:
            print("loading voxel error", e)
            continue

        # ----- load mesh
        try:
            mesh = trimesh.load(os.path.join(mesh_dir, name[8:] + ".glb"))
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            mesh = normalize_trimesh(mesh)
            mesh = rotate_trimesh(mesh, 90, 0, 0)
        except Exception as e:
            print("Error on loading and surface sample", e)
            continue

        vertices = np.array(mesh.vertices)

        # ----- voxel-based processing
        label_voxels = []
        for k in range(max_label):
            if selected_label[k] == []:
                label_voxels.append([])
                continue
            label_voxels.append(voxel_idx[np.array(selected_label[k]).astype(np.int32)])

        voxel_pc_outside = voxel_pc
        voxel_idx_outside = voxel_idx

        for k in range(max_label):
            if len(label_voxels[k]) == 0:
                continue

            print(f"Processing label {k}")

            single_voxel_label = label_voxels[k]

            # process_voxel
            try:
                augment_meshes = process_voxel(mesh, voxel_idx_outside, single_voxel_label)
            except Exception as e:
                print("Error on processing voxel", e)
                continue

            # save
            try:
                for idx, tu in enumerate(augment_meshes):
                    verts, faces, attributes = tu
                    save_ply(
                        os.path.join(result_dir, f"{name}_{k}_arg{idx}.ply"),
                        verts,
                        faces,
                        attributes,
                    )
            except Exception as e:
                print("Error saving voxel mesh", e)
                continue

            counter += 1
            elapsed = (time.time() - start_time)
            avg_time = elapsed / counter
            print(f"Processed {counter} in {elapsed:.2f} seconds | Average per item: {avg_time:.2f}")


###python patch_gen.py