import os
import sys

# add the repo to sys.path
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import numpy as np
import argparse
import gc
from omegaconf import OmegaConf
import trimesh

import torch
import torch.nn.functional as F

import torchvision.transforms.functional as VF
from PIL import Image
import time

from datetime import datetime
import trimesh
import torch.nn as nn
from chamferdist import ChamferDistance

chamferDist = ChamferDistance()
#import pytorch3d
import OpenEXR
import Imath
import OpenEXR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt
import shutil


def sample_points_in_ring(inner_radius, outer_radius, num_points):
    # Randomly generate angles uniformly between 0 and 2*pi
    angles = torch.rand(num_points) * 2 * torch.pi
    
    # Randomly generate radii uniformly between the inner and outer radius
    radii = torch.sqrt(torch.rand(num_points) * (outer_radius**2 - inner_radius**2) + inner_radius**2)
    
    # Convert polar coordinates to Cartesian coordinates
    x = radii * torch.cos(angles)
    y = radii * torch.sin(angles)
    
    # Stack x and y coordinates into a shape [num_points, 2]
    points = torch.stack([x, y], dim=1)
    
    return points.cuda(),radii.cuda()


def generate_grid_2d(size,radius = 0.8):
    radius = cfg.circle_radius
    # Create a grid of coordinates in the range [0, 1]
    x = torch.linspace(-1, 1, steps=size)
    y = torch.linspace(-1, 1, steps=size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Stack the grids to create (size x size) 2D vertices
    grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).cuda()
    
    #Initialize a list to store triangles
    triangles = []
    
    # Loop over each square in the grid
    for i in range(size - 1):
        for j in range(size - 1):
            # Calculate the indices of the square's corners
            top_left = i * size + j
            top_right = top_left + 1
            bottom_left = (i + 1) * size + j
            bottom_right = bottom_left + 1
            
            # Define two triangles for each square
            triangles.append([top_left, bottom_left, top_right])
            triangles.append([top_right, bottom_left, bottom_right])
    
    triangles = torch.tensor(triangles, dtype=torch.long)


    # Identify boundary vertices
    circle_mask = torch.nonzero(((x**2 + y**2) <= radius**2),as_tuple=True)

    angles = torch.linspace(0, 2 * torch.pi, cfg.num_boundary_points)
    x_boundary = radius * torch.cos(angles)
    y_boundary = radius * torch.sin(angles)
    
    # Stack boundary points into a 2D tensor
    points_on_boundary = torch.stack([x_boundary, y_boundary], dim=1).cuda()

    ###points on concentric circle
    concentric_points,concentric_radius = sample_points_in_ring(radius - cfg.radius_delta_boundary, radius, cfg.num_concentric_points)

    ##radius
    points_on_boundary = torch.cat([points_on_boundary,concentric_points],dim=0)
    points_radius = torch.cat([torch.ones(cfg.num_boundary_points).cuda()*radius,concentric_radius],dim=0)
    radius_weight = (points_radius - (radius - cfg.radius_delta_boundary))/cfg.radius_delta_boundary
    return grid, triangles, circle_mask, points_on_boundary,radius_weight


def chamfer_distance_torch3d(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer distance between two 3D point clouds using torch3d.

    Args:
        point_cloud1: Tensor of shape (N, 3) representing the first point cloud.
        point_cloud2: Tensor of shape (M, 3) representing the second point cloud.

    Returns:
        chamfer_dist: The Chamfer distance between the two point clouds.
    """

    # Chamfer distance
    loss = chamferDist(point_cloud1.unsqueeze(0), point_cloud2.unsqueeze(0), bidirectional=True)
    return loss

def chamfer_distance_torch(point_cloud1, point_cloud2):
    """
    Calculate the Chamfer Distance between two 3D point clouds using pure PyTorch.

    Args:
        point_cloud1 (torch.Tensor): Tensor of shape (N, 3) representing the first point cloud.
        point_cloud2 (torch.Tensor): Tensor of shape (M, 3) representing the second point cloud.

    Returns:
        torch.Tensor: The Chamfer Distance between the two point clouds.
    """
    # Ensure the point clouds are of shape (N, 3) and (M, 3)
    assert point_cloud1.dim() == 2 and point_cloud1.size(1) == 3, \
        "point_cloud1 must be of shape (N, 3)"
    assert point_cloud2.dim() == 2 and point_cloud2.size(1) == 3, \
        "point_cloud2 must be of shape (M, 3)"

    # Compute pairwise squared distances between point_cloud1 and point_cloud2
    # Using (a - b)^2 = a^2 + b^2 - 2ab
    # Expand the squared norms
    pc1_sq = torch.sum(point_cloud1 ** 2, dim=1).unsqueeze(1)  # Shape: (N, 1)
    pc2_sq = torch.sum(point_cloud2 ** 2, dim=1).unsqueeze(0)  # Shape: (1, M)
    
    # Compute the inner product
    inner_product = torch.matmul(point_cloud1, point_cloud2.t())  # Shape: (N, M)
    
    # Compute pairwise squared Euclidean distances
    distances = pc1_sq + pc2_sq - 2 * inner_product  # Shape: (N, M)
    
    # Ensure no negative distances due to numerical errors
    distances = torch.clamp(distances, min=0.0)
    
    # For each point in point_cloud1, find the nearest point in point_cloud2
    min_dist_pc1, _ = torch.min(distances, dim=1)  # Shape: (N,)
    
    # For each point in point_cloud2, find the nearest point in point_cloud1
    min_dist_pc2, _ = torch.min(distances, dim=0)  # Shape: (M,)
    
    # Compute the Chamfer Distance as the sum of the mean of minimum distances
    chamfer_dist = torch.mean(min_dist_pc1) + torch.mean(min_dist_pc2)
    
    return chamfer_dist



class LeakyReLUMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, residual_layer_index=4, negative_slope=0.01):
        super(LeakyReLUMLP, self).__init__()
        self.n_layers = n_layers
        self.residual_layer_index = residual_layer_index

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        # Hidden layers
        for i in range(1, n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        # Convert to ModuleList to access layers individually
        self.layers = nn.ModuleList(layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward pass through the first half of the network
        for i in range(self.residual_layer_index):
            x = self.layers[2 * i](x)
            x = self.layers[2 * i + 1](x)
        
        # Residual connection (skip connection)
        residual = x
        
        # Forward pass through the second half of the network
        for i in range(self.residual_layer_index, self.n_layers):
            x = self.layers[2 * i](x)
            x = self.layers[2 * i + 1](x)
        
        # Add the residual connection
        x = x + residual
        
        # Output layer
        output = self.output_layer(x)
        return output

class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super(SirenLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        
        # Initialization
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-torch.sqrt(torch.tensor(6 / self.linear.in_features)) / self.omega_0, 
                                            torch.sqrt(torch.tensor(6 / self.linear.in_features)) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SirenNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, omega_0=30.0, omega_0_out=30.0):
        super(SirenNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input layer
        self.input_layer = SirenLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0)

        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList([
            SirenLayer(hidden_dim + input_dim, hidden_dim, omega_0=omega_0) for _ in range(n_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.omega_0_out = omega_0_out

    def forward(self, x):
        # Forward pass with skip connections
        h = self.input_layer(x)
        for layer in self.hidden_layers:
            h = layer(torch.cat([h, x], dim=-1))  # Skip connection
        output = torch.sin(self.omega_0_out * self.output_layer(h))
        return output

def get_weight(curent_time,times,weights):
    for i in range(len(times)):
        if curent_time < times[0]:
            return weights[0]
        if curent_time < times[i] and curent_time >= times[i-1]:
            return weights[i-1] + (weights[i] - weights[i-1]) * (curent_time - times[i-1]) / (times[i] - times[i-1])
    return weights[-1]


def interpolate_between_rings(image, inner_radius, outer_radius):
    # Get the image size and create a grid of (x, y) coordinates
    image = torch.from_numpy(image).float()
    image_size = image.shape[0]
    y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
    y = y + 0.5
    x = x + 0.5
    
    # Calculate the distance from the center for each pixel
    center = (image_size) / 2
    dist_from_center = torch.sqrt((x - center)**2 + (y - center)**2)

    # Calculate the angles for the rings
    angles = torch.atan2(y - center, x - center) + torch.pi  # range [0, 2*pi)


    # Create an empty image for the output
    interpolated_image = torch.zeros_like(image)
    
    # Interpolation function
    def interpolate(inner_value, outer_value, ratio):
        return inner_value + ratio * (outer_value - inner_value)
    

    # Interpolate between the rings for each pixel
    for i in range(image_size):
        for j in range(image_size):
            distance = dist_from_center[i, j]
            angle = angles[i, j]
            
            if inner_radius <= distance and distance <= outer_radius+1:
                ratio = (distance - inner_radius) / (outer_radius - inner_radius)

                inner_pixel = image[int(center - inner_radius * torch.sin(angle)), int(center - inner_radius * torch.cos(angle))]
                outer_pixel = torch.zeros_like(inner_pixel)#

                # Perform the interpolation
                interpolated_image[i, j] = interpolate(inner_pixel, outer_pixel, ratio)
            elif distance < inner_radius:
                interpolated_image[i, j] = image[i, j]
            elif distance > outer_radius:
                interpolated_image[i, j] = image[i, j]
    
    return interpolated_image.numpy()


def mask_circle_in_square_image(image):
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Ensure the image is square
    if height != width:
        raise ValueError("Image must be square")
    
    # Calculate the radius of the largest inscribed circle
    radius = height // 2
    center = (radius, radius)
    
    # Create a mask with the same dimensions as the image
    Y, X = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = distance_from_center <= radius
    
    # Apply the mask to the image: set pixels outside the circle to [0, 0, 0]
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    
    return masked_image


def rasterize_points(points, colors, grid_size=512):
    """
    Rasterizes points with RGB values onto a grid using nearest neighbor method, 
    and fills empty pixels with the nearest non-empty pixel's value.
    
    Parameters:
    - points: Array of shape (num_points, 2) with normalized coordinates in [0, 1].
    - colors: Array of shape (num_points, 3) with RGB values.
    - grid_size: Size of the output square grid (default is 512).
    
    Returns:
    - image: 3D numpy array of shape (grid_size, grid_size, 3) representing the rasterized image.
    """
    
    # Initialize the image and mask
    half_grid_size = grid_size // 2
    image = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    
    # Convert normalized coordinates to grid coordinates
    def normalize_to_grid(coords, grid_size):
        return (coords * (grid_size - 1)).astype(int)
    
    grid_coords = normalize_to_grid(points, grid_size)
    
    # Rasterize points using nearest neighbor method
    for i in range(len(points)):
        x, y = grid_coords[i]
        if 0 <= x < grid_size and 0 <= y < grid_size:
            image[y, x] = colors[i]
            mask[y, x] = True
    
    indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    
    # Use the indices to fill in the empty pixels
    filled_image = image[indices[0], indices[1]]
    filled_image = mask_circle_in_square_image(filled_image)
    filled_image = interpolate_between_rings(filled_image, int(half_grid_size * (1-cfg.radius_delta_blend)),half_grid_size)
    
    return filled_image

def save_exr(array,path):
    R = array[:, :, 0].flatten().tobytes()
    G = array[:, :, 1].flatten().tobytes()
    B = array[:, :, 2].flatten().tobytes()

    # Create OpenEXR header
    exr_header = OpenEXR.Header(array.shape[1], array.shape[0])

    # Set the file to be a scanline image (default type)
    exr_header['compression'] = Imath.Compression(Imath.Compression.NO_COMPRESSION)
    exr_header['type'] = b'scanlineimage'

    # Set xDensity attribute to 72.0
    exr_header['xDensity'] = 72.0
    # Create the OutputFile with the scanline option
    exr_file = OpenEXR.OutputFile(path, exr_header)

    # Write the pixels (R, G, B channels)
    exr_file.writePixels({'R': R, 'G': G, 'B': B})

    # Close the file
    exr_file.close()

#bake VDM
def bake_vdm(verts,uv):
    diff = (verts[:,:2] - (uv -0.5) * 2)  # for uv in range [0,1]
    z = verts[:, 2]  # Get the z-coordinates
    xyz = np.concatenate([diff, z[:, None]], axis=1)  # Combine diff and z
    return xyz

def mesh2VDM(verts, uv, save_path):
    xyz = bake_vdm(verts, uv)
    images = rasterize_points(uv, xyz)
    ##flipping the image vertically

    images = np.flipud(images)
    save_exr(images, save_path)





def keep_largest_connected_component(mesh):
    # Find all connected components in the mesh
    connected_components = mesh.split(only_watertight=False)
    
    # Find the largest connected component by comparing the number of faces
    largest_component = max(connected_components, key=lambda comp: comp.faces.shape[0])
    
    return largest_component

def remove_vertices_above_threshold(mesh, vertices_to_remove):
    # Identify the vertices to remove
    #vertices_to_remove = np.where(mesh.vertices[:, 0] > x_threshold)[0]
    
    # Identify the faces to remove (faces that have any of their vertices in the vertices_to_remove)
    mask_faces = np.isin(mesh.faces, vertices_to_remove).any(axis=1)
    
    # Invert the mask to keep the faces that don't need to be removed
    faces_to_keep = ~mask_faces
    
    # Create a new mesh with the remaining faces
    new_mesh = mesh.submesh([faces_to_keep], only_watertight=False)[0]
    
    return new_mesh

def rotate_mesh_y_90(mesh):
    mesh_vertices = np.array(mesh.vertices)
    # Define the 90-degree rotation matrix around the Y-axis
    rotation_matrix = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    
    # Apply the rotation to each vertex
    rotated_vertices = np.matmul(mesh_vertices, rotation_matrix.T)
    mesh = trimesh.Trimesh(vertices=rotated_vertices, faces=np.array(mesh.faces))
    return mesh

def calculate_threshold(mesh):
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    upper_verts = verts[np.nonzero(verts[:, 2] > 0.02)]
    upper_verts_bbox = np.array([np.min(upper_verts, axis=0), np.max(upper_verts, axis=0)])
    cut_threshold = np.max(np.abs(upper_verts_bbox[:,:2]))

    cut_threshold = cut_threshold + (cfg.plane_bound - cut_threshold) * cfg.margin_ratio
    print("cut_threahold",cut_threshold)
    return cut_threshold

def scale_mesh(mesh):
    verts = np.array(mesh.vertices)
    verts_bbox = np.array([np.min(verts, axis=0), np.max(verts, axis=0)])[:,:2]

    lower_verts_idx = np.nonzero(verts[:, 2] < 0.01)
    lower_verts_z_before = verts[lower_verts_idx][:, 2].mean()

    # Scale the mesh to fit within the bounding box
    center = np.concatenate([(verts_bbox[0] + verts_bbox[1]) / 2,np.array([0])],axis=0)
    verts = verts - center
    verts_bbox = np.array([np.min(verts, axis=0), np.max(verts, axis=0)])[:,:2]

    scale = np.mean(1/np.abs(verts_bbox))
    verts = verts * scale

    lower_verts_z_after = verts[lower_verts_idx][:, 2].mean()
    print(lower_verts_z_before, lower_verts_z_after)
    verts = verts + np.array([0, 0, lower_verts_z_before - lower_verts_z_after])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(mesh.faces))
    return mesh


def scale_mesh2(mesh,specail_z_verts):
    verts = np.array(mesh.vertices)
    verts_bbox = np.array([np.min(verts, axis=0), np.max(verts, axis=0)])[:,:2]


    # Scale the mesh to fit within the bounding box
    center = np.concatenate([(verts_bbox[0] + verts_bbox[1]) / 2,np.array([0])],axis=0)
    verts = verts - center
    specail_z_verts = specail_z_verts - center
    verts_bbox = np.array([np.min(verts, axis=0), np.max(verts, axis=0)])[:,:2]

    scale = np.mean(1/np.abs(verts_bbox))
    verts = verts * scale
    specail_z_verts = specail_z_verts * scale

    z_value_new = specail_z_verts[:, 2].mean()

    verts = verts + np.array([0, 0,  - z_value_new])# -0.05
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(mesh.faces))
    mesh.export(f"{cfg.save_dir}/after.obj")
    return mesh



def compute_z_threshold(mesh,cut_threshold,flip_flag = False):
    special_verts = []
    verts = np.array(mesh.vertices)
    vertex_normals = np.array(mesh.vertex_normals)
    special_verts.append(np.where(verts[:, 0] > cut_threshold * cfg.z_cut_ratio)[0])
    special_verts.append(np.where(verts[:, 0] < -cut_threshold * cfg.z_cut_ratio)[0])
    special_verts.append(np.where(verts[:, 1] > cut_threshold * cfg.z_cut_ratio)[0])
    special_verts.append(np.where(verts[:, 1] < -cut_threshold * cfg.z_cut_ratio)[0])
    special_index = np.concatenate(special_verts)
    special_verts = verts[special_index]
    special_vertex_normals = vertex_normals[special_index]
    special_vertex_normals = special_vertex_normals/np.linalg.norm(special_vertex_normals, axis=1,keepdims=True)
    dot_product = np.dot(special_vertex_normals, np.array([[0, 0, 1]]).T)

    vertex_normals = np.array(mesh.vertex_normals)
    unit_vertex_normals = vertex_normals/np.linalg.norm(vertex_normals, axis=1,keepdims=True)
    dot_product_all = np.dot(unit_vertex_normals, np.array([[0, 0, 1]]).T)
    z_pos = verts[dot_product_all[:, 0] > 0][:, 2].mean()
    z_neg = verts[dot_product_all[:, 0] < 0][:, 2].mean()
    print("z_pos",z_pos,"z_neg",z_neg)
    print("flip_flag",z_neg > z_pos)

    if z_neg < z_pos:
        special_verts = special_verts[dot_product[:, 0] > 0]
    else:
        special_verts = special_verts[dot_product[:, 0] < 0]
    z_value = special_verts[:, 2].mean()
    return z_value,special_verts

def calculate_negative_normal(mesh):
    vertices = np.array(mesh.vertices)
    vertex_normals = np.array(mesh.vertex_normals)
    unit_vertex_normals = vertex_normals/np.linalg.norm(vertex_normals, axis=1,keepdims=True)
    dot_product = np.dot(unit_vertex_normals, np.array([[0, 0, 1]]).T)

    z_pos = vertices[dot_product[:, 0] > 0][:, 2].mean()
    z_neg = vertices[dot_product[:, 0] < 0][:, 2].mean()
    print("z_pos",z_pos,"z_neg",z_neg)
    print("flip_flag",z_neg > z_pos)

    if z_neg > z_pos:
        negative_normal_index = (dot_product > 0)
    else:
        negative_normal_index = (dot_product < 0)

    return negative_normal_index[:,0]


def cut_mesh(mesh):
    mesh = keep_largest_connected_component(mesh)
    mesh = rotate_mesh_y_90(mesh)

    cut_threshold = calculate_threshold(mesh)


    verts = np.array(mesh.vertices)
    print("verts bbox", np.array([np.min(verts, axis=0), np.max(verts, axis=0)]))
    mesh = remove_vertices_above_threshold(mesh, np.where(verts[:, 0] > cut_threshold)[0])
    verts = np.array(mesh.vertices)
    mesh = remove_vertices_above_threshold(mesh, np.where(verts[:, 0] < -cut_threshold)[0])
    verts = np.array(mesh.vertices)
    mesh = remove_vertices_above_threshold(mesh, np.where(verts[:, 1] > cut_threshold)[0])
    verts = np.array(mesh.vertices)
    mesh = remove_vertices_above_threshold(mesh, np.where(verts[:, 1] < -cut_threshold)[0])
    verts = np.array(mesh.vertices)
    z_threshold, special_z_verts = compute_z_threshold(mesh,cut_threshold,flip_flag=False)
    print("z_threshold",z_threshold)


    verts = np.array(mesh.vertices)
    negative_normal_index = calculate_negative_normal(mesh)


    mesh = remove_vertices_above_threshold(mesh, np.where(np.logical_and((verts[:, 2] < z_threshold- 0.02) , negative_normal_index))[0])
    mesh = keep_largest_connected_component(mesh)
    mesh = scale_mesh2(mesh,special_z_verts)
    return mesh


def generate_uv_map(mesh, uv_size=512):
    u = np.linspace(0, 1, uv_size)
    v = np.linspace(0, 1, uv_size)
    xv, yv = np.meshgrid(u, v)

    u = xv.flatten()
    v = yv.flatten()
    uv = np.column_stack([u, v])
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
    return mesh

def main(cfg):
    ###ini
    time_stamp = sorted(os.listdir(f"./instant-nsr-pl/exp/{exp_name}/{base_name}"))[-1]
    GT_mesh = trimesh.load_mesh(f"./instant-nsr-pl/exp/{exp_name}/{base_name}/{time_stamp}/save/it{cfg.extract_epoch}-mc192.obj")
    GT_mesh = cut_mesh(GT_mesh)



    GT_vertices = torch.from_numpy(np.array(GT_mesh.vertices)).float().cuda()



    input_grid,triangles,pred_circle_mask,boundary_points,boundary_weight = generate_grid_2d(cfg.grid_size)

    boundary_input_grid = boundary_points
    boundary_gt_grid = torch.cat([boundary_input_grid, torch.zeros(boundary_input_grid.shape[0],1).cuda()], dim=-1)

    if cfg.mlp == "leakyrelu":
        mlp = LeakyReLUMLP(2, 3, cfg.hidden_dim, cfg.n_layers).cuda()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=cfg.lr)
    else: 
        raise NotImplementedError("MLP type not implemented")

    for i in range(cfg.num_iters+1):
        optimizer.zero_grad()
        rand_input_grid = input_grid 

        input_vertices = torch.cat([rand_input_grid, boundary_input_grid], dim=0)
        output_vertices = mlp(input_vertices)
        output_random_vertices = output_vertices[:cfg.batch_size]

        output_boundary_vertices = output_vertices[cfg.batch_size:]
        if i <= 50:
            zero_output_random_vertices = torch.cat([rand_input_grid, torch.zeros(rand_input_grid.shape[0],1).cuda()], dim=-1)
            recon_loss = torch.mean((output_random_vertices - zero_output_random_vertices)**2)
        else:
            GT_vertices_epoch = GT_vertices
            recon_loss = chamfer_distance_torch3d(output_random_vertices, GT_vertices_epoch)

    

        bounrdary_loss = torch.mean((output_boundary_vertices - boundary_gt_grid)**2 * boundary_weight[:,None])


        boundary_loss = bounrdary_loss * get_weight(i,cfg.boundary_weight_interval,cfg.boundary_weight)
        loss = recon_loss + boundary_loss #+ edge_constrain_loss
        loss.backward()
        optimizer.step()

        
        if i % 50 == 0:
            print(f"iter: {i}, loss: {loss.item()}, recon_loss: {recon_loss.item()}, boundary_loss: {boundary_loss.item()},") # edge_constrain_loss: {edge_constrain_loss.item()}")
            print(f"boundary weight: {get_weight(i,cfg.boundary_weight_interval,cfg.boundary_weight)}")

        if i  in cfg.frequency and i != 0:
            with torch.no_grad():
                validate_input_grid,validate_triangles,_,_,_ = generate_grid_2d(cfg.validate_vdm_dim)
                validate_vertices = mlp(validate_input_grid)
                zero_validate_input_grid = torch.cat([validate_input_grid, torch.zeros(validate_input_grid.shape[0],1).cuda()], dim=-1)
                validate_mesh_input,validate_mesh_faces,_,_,_ = generate_grid_2d(cfg.validate_mesh_dim)
                validate_mesh_vertices = mlp(validate_mesh_input)
                mesh = trimesh.Trimesh(vertices= validate_mesh_vertices.cpu().detach().numpy(),faces = validate_mesh_faces)
                mesh = generate_uv_map(mesh,cfg.validate_vdm_dim)
            mesh.export(f"{cfg.save_dir}/{base_name}_{i}.obj")
            uv= zero_validate_input_grid[:,:2].cpu().detach().numpy() /2 + 0.5
            verts = validate_vertices.cpu().detach().numpy()
            mesh2VDM(verts, uv,f"{cfg.save_dir}/{base_name}_{i}.exr")
            print("successfully saved mesh and VDM")
            # exit(0)

        writer.add_scalar('Loss/loss', loss.item(), i)
        writer.add_scalar('Loss/recon_loss', recon_loss.item(), i)
        writer.add_scalar('Loss/boundary_loss', boundary_loss.item(), i)
    print("done",time.time()-start_time)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize an image using PyTorch.")
    parser.add_argument('--base_name',required=True, default="",type=str, help="Base name.")
    parser.add_argument('--exp_name', required=True, default="",type=str, help="Exp name.")
    arg = parser.parse_args()
    
    base_name = arg.base_name
    exp_name = arg.exp_name

    
    start_time = time.time()
    # --- 1. Load Config from YAML ---
    config_file_path = "./config/VDM_setting.yaml"
    cfg = OmegaConf.load(config_file_path)
    cfg.batch_size = (cfg.grid_size) ** 2

    # --- 2. Handle any dynamic setup you need (e.g., directories) ---
    # Combine base_name + current time to replicate your existing logic
    output_base_dir = os.path.join(cfg.output_base_dir, exp_name)

    os.makedirs(output_base_dir, exist_ok=True)

    # e.g., directory name: baseName_YYMMDD-HHMMSS
    dirname = base_name + "_" + datetime.now().strftime("%y%m%d-%H%M%S")

    # Where to save
    cfg.save_dir = os.path.join(output_base_dir, dirname)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_dir, "tf"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_dir, "code"), exist_ok=True)

    # (Optional) Copy this script into the code folder for record-keeping
    shutil.copy(__file__, os.path.join(cfg.save_dir, "code"))
    shutil.copy(config_file_path, os.path.join(cfg.save_dir, "code"))

    # --- 3. Dynamically compute or adjust certain config values if needed ---
    if isinstance(cfg.boundary_weight_interval, list):
        cfg.boundary_weight_interval = [
            int(cfg.num_iters * frac) for frac in cfg.boundary_weight_interval
        ]

    # Example: if you want to have a derived batch_size
    # cfg.batch_size = cfg.grid_size ** 2

    # --- 4. Save the final config for reproducibility ---
    final_cfg_path = os.path.join(cfg.save_dir, "VDM_setting.yaml")
    OmegaConf.save(config=cfg, f=final_cfg_path)

    # --- 5. (Optional) TensorBoard writer, etc. ---
    writer = SummaryWriter(log_dir=os.path.join(cfg.save_dir, "tf"))

    # --- 6. Call your main function with the OmegaConf dictionary ---
    main(cfg)
