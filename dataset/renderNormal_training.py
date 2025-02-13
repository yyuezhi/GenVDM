import blenderproc as bproc
import bpy
import pip
import numpy as np
import math
import os
import mathutils
import PIL.Image as Image
from mathutils import Vector, Matrix
import time
import trimesh
import mathutils
import bmesh
import time
from scipy.spatial import KDTree
from trimesh.transformations import rotation_matrix
import sys
import random
pip.main(["install","plyfile"])
from plyfile import PlyData, PlyElement
pip.main(["install","omegaconf"])
from omegaconf import OmegaConf
print("start")
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def clear_imported_objects():

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.objects["Object"].select_set(True)
    bpy.ops.object.delete()


def clear_all_objects():
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
        bpy.ops.object.delete()
    for camera in bpy.data.cameras:
        # Remove the camera data block
        bpy.data.cameras.remove(camera)

def setLight(location, energy = 100, size = 5):
    light_data = bpy.data.lights.new(name="AreaLight", type='AREA')
    light_object = bpy.data.objects.new(name="AreaLight", object_data=light_data)
    light_data.energy = 50  # Adjust the light energy/intensity
    light_data.size = 0.2  # Adjust the size of the area light
    light_object.location = location
    bpy.context.collection.objects.link(light_object)

def setLight_ambient(color = (0,0,0,1)):
    bpy.data.scenes[0].world.use_nodes = True
    bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = color
    bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Strength'].default_value = 2.0

def setCircleLight(light_energy = 500,radius = 1,height = 1):
    num_light = 3
    for i in range(num_light):
        angle = i * (2 * math.pi / num_light)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        # Create a new light datablock
        light_data = bpy.data.lights.new(name=f"AreaLight_{i}", type='AREA')
        light_data.energy = light_energy
        # if i == 0:
        #     light_data.energy = 2000
        light_data.size = 3  # Adjust the size of the area light if needed
        
        # Create a new object with the light datablock
        light_object = bpy.data.objects.new(name=f"AreaLight_{i}", object_data=light_data)
        
        # Set the location of the light
        light_object.location = (x, y, height)
        
        # Point the light towards the origin
        direction = mathutils.Vector((x, y, height))
        light_object.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
        
        # Link the light object to the current collection
        bpy.context.collection.objects.link(light_object)

    light_data = bpy.data.lights.new(name=f"AreaLight_{i+1}", type='AREA')
    light_data.energy = 300
    # if i == 0:
    #     light_data.energy = 2000
    light_data.size = 1  # Adjust the size of the area light if needed
        
    light_object = bpy.data.objects.new(name=f"AreaLight_{i+1}", object_data=light_data)
    
    # Set the location of the light
    light_object.location = (4,0,0.3)
    
    # Point the light towards the origin
    direction = mathutils.Vector((0,0,0.3))
    light_object.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    bpy.context.collection.objects.link(light_object)


def view_plane(camd, winx, winy, xasp, yasp):    
    #/* fields rendering */
    ycor = yasp / xasp
    use_fields = False
    if (use_fields):
      ycor *= 2

    def BKE_camera_sensor_size(p_sensor_fit, sensor_x, sensor_y):
        #/* sensor size used to fit to. for auto, sensor_x is both x and y. */
        if (p_sensor_fit == 'VERTICAL'):
            return sensor_y;

        return sensor_x;

    if (camd.type == 'ORTHO'):
      #/* orthographic camera */
      #/* scale == 1.0 means exact 1 to 1 mapping */
      pixsize = camd.ortho_scale
    else:
      #/* perspective camera */
      sensor_size = BKE_camera_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
      pixsize = (sensor_size * camd.clip_start) / camd.lens

    #/* determine sensor fit */
    def BKE_camera_sensor_fit(p_sensor_fit, sizex, sizey):
        if (p_sensor_fit == 'AUTO'):
            if (sizex >= sizey):
                return 'HORIZONTAL'
            else:
                return 'VERTICAL'

        return p_sensor_fit

    sensor_fit = BKE_camera_sensor_fit(camd.sensor_fit, xasp * winx, yasp * winy)

    if (sensor_fit == 'HORIZONTAL'):
      viewfac = winx
    else:
      viewfac = ycor * winy

    print("data",pixsize, viewfac)
    pixsize /= viewfac

    #/* extra zoom factor */
    pixsize *= 1 #params->zoom

    #/* compute view plane:
    # * fully centered, zbuffer fills in jittered between -.5 and +.5 */
    xmin = -0.5 * winx
    ymin = -0.5 * ycor * winy
    xmax =  0.5 * winx
    ymax =  0.5 * ycor * winy

    #/* lens shift and offset */
    dx = camd.shift_x * viewfac # + winx * params->offsetx
    dy = camd.shift_y * viewfac # + winy * params->offsety

    xmin += dx
    ymin += dy
    xmax += dx
    ymax += dy

    #/* fields offset */
    #if (params->field_second):
    #    if (params->field_odd):
    #        ymin -= 0.5 * ycor
    #        ymax -= 0.5 * ycor
    #    else:
    #        ymin += 0.5 * ycor
    #        ymax += 0.5 * ycor

    #/* the window matrix is used for clipping, and not changed during OSA steps */
    #/* using an offset of +0.5 here would give clip errors on edges */
    xmin *= pixsize
    xmax *= pixsize
    ymin *= pixsize
    ymax *= pixsize

    return xmin, xmax, ymin, ymax


def projection_matrix(camd):
    r = bpy.context.scene.render
    left, right, bottom, top = view_plane(camd, r.resolution_x, r.resolution_y, 1, 1)
    print("LRBT",left, right, bottom, top)
    farClip, nearClip = camd.clip_end, camd.clip_start

    Xdelta = right - left
    Ydelta = top - bottom
    Zdelta = farClip - nearClip
    print("NF",nearClip, farClip)
    print("XYZ Delta",Xdelta, Ydelta, Zdelta)
    mat = [[0]*4 for i in range(4)]

    ###persepective projection
    # mat[0][0] = nearClip * 2 / Xdelta
    # mat[1][1] = nearClip * 2 / Ydelta
    # mat[2][0] = (right + left) / Xdelta #/* note: negate Z  */
    # mat[2][1] = (top + bottom) / Ydelta
    # mat[2][2] = -(farClip + nearClip) / Zdelta
    # mat[2][3] = -1
    # mat[3][2] = (-2 * nearClip * farClip) / Zdelta

    ###ortho projection
    mat[0][0] = 2 / Xdelta
    mat[1][1] = 2 / Ydelta
    mat[2][2] = -2 / Zdelta
    mat[3][0] = -(right + left) / Xdelta
    mat[3][1] = -(top + bottom) / Ydelta
    mat[3][2] = -(farClip + nearClip) / Zdelta
    mat[3][3] = 1
    return sum([c for c in mat], [])


###the up the camera and the object track to 
def setup_camera(azimuth_range, elevation_range,distance):
    # Add a target empty object for the camera to look at
    #bpy.ops.object.mode_set(mode='OBJECT')
    if not bpy.data.objects.get('CameraTarget'):
        bpy.ops.object.empty_add(type='PLAIN_AXES')
        bpy.context.active_object.name = 'CameraTarget'
    target = bpy.data.objects['CameraTarget']
    target.location = (0, 0, 0)#(-0.41, -0.13, 0.117)  #big ball location -0.5585, 0, 0.5080

    bpy.ops.object.mode_set(mode='OBJECT')
    if not bpy.data.objects.get('Camera'):
        bpy.ops.object.camera_add()
    camera = bpy.data.objects['Camera']
    if ORTHO_FLAG:
        bpy.data.cameras[0].type = "ORTHO"
        bpy.data.cameras[0].ortho_scale= ORTHO_SCALE
    bpy.context.scene.camera = camera

    
    for i,azimuth in enumerate(azimuth_range):
        elevation = elevation_range[i]
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW')
        camera = bpy.context.selected_objects[0]
        # Convert azimuth and elevation to radians
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)
        # Compute camera position
        x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
        y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        z = distance * math.sin(elevation_rad)
        # Set camera position
        camera.location = (x, y, z)
        camera.data.angle = math.radians(fov)
        camera.name = f'Camera.{i:03d}'
        constraint = camera.constraints.new(type='TRACK_TO')
        camera.parent = target
        constraint.target = target
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
        
    bpy.context.view_layer.update()   
    bpy.ops.object.select_all(action='DESELECT')

    bproc.camera.set_resolution(resolution, resolution)
    for i,azimuth in enumerate(azimuth_range):
        cam = bpy.data.objects[f'Camera.{i:03d}']
        location, rotation = cam.matrix_world.decompose()[0:2]
        cam_pose = bproc.math.build_transformation_mat(location, rotation.to_matrix())
        bproc.camera.add_camera_pose(cam_pose,i)
        bpy.context.view_layer.update()  
        
        
        RT = get_3x4_RT_matrix_from_blender(cam)
        #RT_path = os.path.join(output_folder, f"{i}_RT.txt")
        #np.savetxt(RT_path, RT)
        #a = projection_matrix(bpy.context.scene.camera.data)
        #print(a)
        #print(RT)


    bpy.ops.object.select_all(action='DESELECT')


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    # print("=========================")
    # for obj in bpy.data.objects:
    #     print(obj.name)
    # for obj in bpy.data.cameras:
    #     print(obj.name)
    # for obj in bpy.data.materials:
    #     print(obj.name)
    # print("num of objects",len(bpy.data.objects))
    # print("num of camera",len(bpy.data.cameras))
    # print("num of materials",len(bpy.data.materials))
    for obj in bpy.data.objects:
        if obj.type not in {"LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    #delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)
    for cam in bpy.data.cameras:
        bpy.data.cameras.remove(cam, do_unlink=True)
    # print("+++++++++++++++++++++++++")
    # print("num of objects",len(bpy.data.objects))
    # print("num of camera",len(bpy.data.cameras))
    # print("num of materials",len(bpy.data.materials))
    # for obj in bpy.data.objects:
    #     print(obj.name)
    # for obj in bpy.data.cameras:
    #     print(obj.name)
    # for obj in bpy.data.materials:
    #     print(obj.name)

def create_material(name, base_color, ambient_color, metallic, roughness, specular):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create new nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    ambient_occlusion_node = nodes.new(type='ShaderNodeAmbientOcclusion')
    mix_shader_node = nodes.new(type='ShaderNodeMixShader')
    diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')

    # Set up the Principled BSDF node
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Metallic'].default_value = metallic
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['Specular'].default_value = specular

    # Set up the Ambient Occlusion node
    ambient_occlusion_node.inputs['Color'].default_value = ambient_color
    ambient_occlusion_node.inputs['Distance'].default_value = 3.0

    # Connect nodes
    links.new(principled_node.outputs['BSDF'], mix_shader_node.inputs[2])
    links.new(ambient_occlusion_node.outputs['Color'], diffuse_node.inputs['Color'])
    links.new(diffuse_node.outputs['BSDF'], mix_shader_node.inputs[1])
    links.new(mix_shader_node.outputs['Shader'], output_node.inputs['Surface'])

    return material


def set_material():
    bpy.context.scene.objects["Object"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects["Object"]
    # Create a new material
    material = bpy.data.materials.new(name="Diffuse_Reddish_Brown_With_Ambient")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create new nodes
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    ambient_occlusion_node = nodes.new(type='ShaderNodeAmbientOcclusion')
    mix_shader_node = nodes.new(type='ShaderNodeMixShader')
    diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')

    # Set up the Principled BSDF node for a less metallic and more diffuse appearance
    principled_node.inputs['Base Color'].default_value = (0.35, 0.15, 0.1, 1)  # Reddish-brown color
    principled_node.inputs['Metallic'].default_value = 0.2  # Less metallic
    principled_node.inputs['Roughness'].default_value = 0.6  # More diffuse
    principled_node.inputs['Specular'].default_value = 0.1  # Specular reflection

    # Set up the Ambient Occlusion node
    ambient_occlusion_node.inputs['Color'].default_value = (0.35, 0.15, 0.1, 1)  # Ambient color
    ambient_occlusion_node.inputs['Distance'].default_value = 3.0  # Adjust the strength

    # Connect nodes
    links.new(principled_node.outputs['BSDF'], mix_shader_node.inputs[2])
    links.new(ambient_occlusion_node.outputs['Color'], diffuse_node.inputs['Color'])
    links.new(diffuse_node.outputs['BSDF'], mix_shader_node.inputs[1])
    links.new(mix_shader_node.outputs['Shader'], output_node.inputs['Surface'])

    # Assign the material to the active object
    if bpy.context.object:
        obj = bpy.context.object
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    else:
        print("No active object selected.")

def set_material_background_grey():
    bpy.context.scene.objects["Object"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects["Object"]
    obj = bpy.context.scene.objects["Object"]
    obj.data.materials.clear()
    for group_name, mat_name in vertex_groups_materials.items():
        material = bpy.data.materials.get(mat_name)
        obj.data.materials.append(material)


    for group_name, mat_name in vertex_groups_materials.items():
        # Get the material index
        mat_index = obj.data.materials.find(mat_name)
        
        # Set to Edit mode to select vertices
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')

        # Select vertices in the vertex group
        bpy.ops.object.vertex_group_set_active(group=group_name)
        bpy.ops.object.vertex_group_select()
        
        # Assign material to the selected vertices
        bpy.context.object.active_material_index = mat_index
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')



        
def render_image(azimuth, elevation,distance,brush_name):
    setup_camera(azimuth, elevation,distance)
    #set_material()
    set_material_background_grey()

    bproc.renderer.enable_normals_output()
    # Render the scene
    data = bproc.renderer.render()
    
    num_views = len(azimuth)


    for j in range(num_views):
        index = j
                
        # Nomralizes depth maps
        depth_map = data['depth'][index]
        depth_max = np.max(depth_map)
        valid_mask = depth_map!=depth_max
        invalid_mask = depth_map==depth_max
        depth_map[invalid_mask] = 0
        
        depth_map = np.uint16((depth_map / 10) * 65535)

        normal_map = data['normals'][index]*255

        valid_mask = valid_mask.astype(np.int8)*255

        color_map = data['colors'][index]
        color_map = np.concatenate([color_map, valid_mask[:, :, None]], axis=-1)

        normal_map = np.concatenate([normal_map, valid_mask[:, :, None]], axis=-1)

        Image.fromarray(color_map.astype(np.uint8)).save(
        f'{color_save_path}/{brush_name}_{j}.png', "png", quality=100)
        
        Image.fromarray(normal_map.astype(np.uint8)).save(
        f'{normal_save_path}/{brush_name}_{j}.png', "png", quality=100)

def set_vertex_group():
    bpy.context.scene.objects["Object"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects["Object"]
    obj = bpy.context.active_object
    group_z0 = obj.vertex_groups.new(name="Background")
    group_rest = obj.vertex_groups.new(name="Foreground")

    mesh = bpy.data.objects["Object"].data

    # Ensure the mesh has UV layers
    if mesh.uv_layers:
        # Get the active UV layer
        uv_layer = mesh.uv_layers.active.data

        # Dictionary to store vertices and their UV coordinates
        vertex_uv_map = {}

        # Loop through the UV coordinates
        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                vertex_index = mesh.loops[loop_index].vertex_index
                uv_coord = uv_layer[loop_index].uv
                vertex_uv_map[vertex_index] = uv_coord


    # Iterate over all vertices
    ray_cast = {}
    for i,vertex in enumerate(obj.data.vertices):
        if vertex.co.y <= 0.01 and vertex.co.y >= -0.01:
            group_z0.add([vertex.index], 1.0, 'ADD')
            vertex.select = False
        else:
            group_rest.add([vertex.index], 1.0, 'ADD')
            vertex.select = True
            u,v = vertex_uv_map[i]
            if u not in ray_cast:
                ray_cast[u] = [v,v]
            else:
                if v < ray_cast[u][0]:
                    ray_cast[u][0] = v
                if v > ray_cast[u][1]:
                    ray_cast[u][1] = v
    #print(ray_cast)
    for i,vertex in enumerate(obj.data.vertices):
        u,v = vertex_uv_map[i]
        if not vertex.select and u in ray_cast:            
            if v >= ray_cast[u][0] and v <= ray_cast[u][1]:
                group_rest.add([vertex.index], 1.0, 'ADD')


def create_square_plane():
    # Define the vertices of the 2x2 square plane in the X, Y plane at Z = 0
    vertices = [
        (-1.0, -1.0, 0.0),  # Bottom-left corner
        (1.0, -1.0, 0.0),   # Bottom-right corner
        (1.0, 1.0, 0.0),    # Top-right corner
        (-1.0, 1.0, 0.0)    # Top-left corner
    ]
    
    # Define the face using the vertices
    faces = [(0, 1, 2, 3)]
    
    # Create a new mesh
    mesh = bpy.data.meshes.new("Plane")
    
    # Create the object with the mesh
    obj = bpy.data.objects.new("Plane", mesh)
    
    # Link the object to the scene
    bpy.context.collection.objects.link(obj)
    
    # Set the mesh data (vertices and faces)
    mesh.from_pydata(vertices, [], faces)
    
    # Update the mesh
    mesh.update()
    
    print("2x2 Square Plane created at Z=0")


def normalize_z_axis(mesh_object):
    # Get the bounding box of the object
    mesh = mesh_object.data
    min_x, min_y, min_z = mesh_object.bound_box[0]
    max_x, max_y, max_z = mesh_object.bound_box[6]

    # Calculate the Z range of the bounding box
    if not (max_z >= z_limit or max_x >= 1.0 or max_y >= 1.0 or min_x <= -1.0 or min_y <= -1.0):
        print("no need to normalize")
        return

    ratio = 1/ max([math.fabs(max_z/z_limit), math.fabs(max_y) , math.fabs(min_y), math.fabs(max_x) , math.fabs(min_x)])
    print("normalizeing with ratio",ratio)
    # Iterate over all vertices to normalize the Z values
    for vertex in mesh.vertices:
        # Normalize Z to range between 0 and 2
        vertex.co.x = vertex.co.x  * ratio
        vertex.co.y = vertex.co.y  * ratio
        vertex.co.z = vertex.co.z  * ratio

    # Output the new bounding box after normalization
    bpy.context.view_layer.update()  # Update scene after modification
    #create_square_plane()
    





def flip_mesh_if_needed():
    # Get the mesh data
    mesh = bpy.data.objects["Object"].data

    # Counters for vertices above and below the Z=0 plane
    positive_z_count = 0
    negative_z_count = 0

    # Iterate over all vertices and count how many are above or below Z=0
    for vertex in mesh.vertices:
        if vertex.co.z > 0:
            positive_z_count += 1
        else:
            negative_z_count += 1

    # Print the counts
    print(f"Vertices above Z=0: {positive_z_count}")
    print(f"Vertices below or equal Z=0: {negative_z_count}")

    # If the majority of vertices are in the negative Z direction, flip the mesh
    if negative_z_count > positive_z_count:
        print("Flipping the mesh along the Z axis...")
        # Apply a 180-degree rotation around the Z axis to flip the mesh
        for vertex in mesh.vertices:
            vertex.co.z = -vertex.co.z

        # Update the scene to reflect the changes to the vertex positions
        bpy.context.view_layer.update()

def load_trimesh_to_blender(trimesh_object, mesh_name="Object"):
    """
    Load a trimesh.Trimesh object into Blender as a mesh.

    :param trimesh_object: The trimesh.Trimesh object to import.
    :param mesh_name: The name of the new mesh object in Blender.
    """
    # Convert trimesh vertices and faces to Blender format
    vertices = np.array(trimesh_object.vertices)
    faces = np.array(trimesh_object.faces)

    # Create a new mesh in Blender
    mesh_data = bpy.data.meshes.new(mesh_name)
    mesh_obj = bpy.data.objects.new(mesh_name, mesh_data)
    bpy.context.collection.objects.link(mesh_obj)

    # Create a new bmesh object and fill it with data
    bm = bmesh.new()
    for v in vertices:
        bm.verts.new(v)
    bm.verts.ensure_lookup_table()
    
    for f in faces:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except ValueError:
            continue
    
    # Update the mesh with bmesh data
    bm.to_mesh(mesh_data)
    bm.free()

    # Recalculate the normals
    mesh_data.update()
    mesh_data.validate()

    # Set the mesh object as the active object
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    #mesh_obj.rotation_euler = (math.radians(90), 0, -math.radians(90))
    mesh_obj.rotation_euler = (math.radians(90),0, 0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


def delete_faces(obj, target_coords, tolerance=1e-5):
    if obj.type != 'MESH':
        print("Selected object is not a mesh.")
        return

    mesh = obj.data

    # Switch to Object Mode to ensure proper updates
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Identify target vertices
    target_vertices = []
    for v in bm.verts:
        for coord in target_coords:
            if (abs(v.co.x - coord[0]) < tolerance and
                abs(v.co.y - coord[1]) < tolerance and
                abs(v.co.z - coord[2]) < tolerance):
                target_vertices.append(v)
                break  # Move to next vertex after a match

    print(f"Expected 4 target vertices, found {len(target_vertices)}.")


    # Get the indices of the target vertices for easier comparison
    target_indices = set(v.index for v in target_vertices)

    # Collect faces to delete
    faces_to_delete = []
    for face in bm.faces:
        if len(face.verts) != 3:
            continue  # Only interested in triangles

        # Check if all 3 vertices of the face are among the target vertices
        face_vertex_indices = set(v.index for v in face.verts)

        if face_vertex_indices.issubset(target_indices) and len(face_vertex_indices) == 3:
            faces_to_delete.append(face)

    print(f"Found {len(faces_to_delete)} triangular face(s) to delete.")

    # Delete the collected faces
    for face in faces_to_delete:
        bm.faces.remove(face)

    # Write the changes back to the mesh
    bm.to_mesh(mesh)
    bm.free()

    # Update the mesh in the viewport
    mesh.update()

    # Optional: Recalculate normals
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

target_vertices_coords = [
    (1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, 1.0),
    (-1.0, 0.0, -1.0)
]

def load_and_split(path):
    mesh = trimesh.load(path)
    a = mesh.split(only_watertight=False)
    if a[0].vertices.shape[0] > a[1].vertices.shape[0]:
        idx = 1
    else:
        idx = 0
    #a[idx] = a[idx].apply_translation([0,0,0.005])
    instance_mesh = a[(idx+1)%2]
    mesh = trimesh.util.concatenate(a)
    load_trimesh_to_blender(mesh)
    delete_faces(bpy.context.scene.objects["Object"], target_vertices_coords, tolerance=1e-4)
    # bpy.ops.wm.save_as_mainfile(filepath=f"/home/yuezhiy/sensei-fs-link/Objaverse/avd.blend")
    # exit(0)
    rot = rotation_matrix(np.deg2rad(90), [1, 0, 0])
    mesh.apply_transform(rot)
    instance_mesh.apply_transform(rot)


    bpy.context.scene.objects["Object"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects["Object"]
    obj = bpy.context.active_object

    # Create two vertex groups: z0 and rest
    group_z0 = obj.vertex_groups.new(name="Background")
    group_rest = obj.vertex_groups.new(name="Foreground")

    # Access the mesh data of the "Object" object
    blender_vertices = np.array([vert.co for vert in obj.data.vertices])
    # print(np.max(blender_vertices,axis=0),np.min(blender_vertices,axis=0))
    # exit(0)
    tree = KDTree(np.array(instance_mesh.vertices))
    distances, _ = tree.query(blender_vertices)
    # print(distances)
    # print(distances.shape)
    counter = 0
    a = 0
    for i,d in enumerate(distances):
        vertex = obj.data.vertices[i]
        if d < 0.01:
            a += 1
            group_rest.add([vertex.index], 1.0, 'ADD')
            vertex.select = False
        else:
            counter += 1
            group_z0.add([vertex.index], 1.0, 'ADD')
            vertex.select = False
    # print(counter,a)
    # exit(0)
    # # Iterate over all vertices
    # for vertex in obj.data.vertices:
    #     if -0.01 <= vertex.co.z <= 0.01:
    #         # Add vertex to group "z0" if its y-coordinate is between -0.01 and 0.01
    #         group_z0.add([vertex.index], 1.0, 'ADD')
    #         vertex.select = False
    #     else:
    #         # Add vertex to group "rest" if it doesn't meet the above condition
    #         group_rest.add([vertex.index], 1.0, 'ADD')
    #         vertex.select = False

def load_ply(filename):
    """
    Load a mesh from a PLY file, including vertex attributes.

    Parameters:
    - filename (str): The name of the PLY file to load.

    Returns:
    - verts (np.ndarray): Vertex coordinates of shape (N, 3).
    - faces (np.ndarray): Face indices of shape (M, 3).
    - vert_attributes (np.ndarray): Vertex attributes of shape (N,) or (N, K).
    """
    ply = PlyData.read(filename)

    # Extract vertex coordinates
    verts = np.vstack([ply['vertex']['x'], ply['vertex']['y'], ply['vertex']['z']]).T

    # Extract faces
    faces = np.vstack(ply['face']['vertex_indices'])

    # Extract attributes (exclude x, y, z)
    attr_names = [name for name in ply['vertex'].data.dtype.names if name not in ('x', 'y', 'z')]
    if not attr_names:
        vert_attributes = None
    elif len(attr_names) == 1:
        vert_attributes = ply['vertex'][attr_names[0]]
    else:
        vert_attributes = np.vstack([ply['vertex'][name] for name in attr_names]).T

    print(f"PLY file '{filename}' loaded successfully.")
    return verts, faces, vert_attributes

def load_and_split2(filename):
    verts, faces, vert_attributes = load_ply(filename)
    mesh = trimesh.Trimesh(vertices=verts,faces=faces)
    load_trimesh_to_blender(mesh)

    
    bpy.context.scene.objects["Object"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects["Object"]
    obj = bpy.context.active_object
    # Create two vertex groups: z0 and rest
    group_z0 = obj.vertex_groups.new(name="Background")
    group_rest = obj.vertex_groups.new(name="Foreground")
    for i,vertex in enumerate(obj.data.vertices):
        label = vert_attributes[i]

        if label == 1:
            # Add vertex to group "z0" if its y-coordinate is between -0.01 and 0.01
            group_z0.add([vertex.index], 1.0, 'ADD')
            vertex.select = False
        else:
            # Add vertex to group "rest" if it doesn't meet the above condition
            group_rest.add([vertex.index], 1.0, 'ADD')
            vertex.select = False


if __name__ == "__main__":
    cfg = OmegaConf.load("./config/render.yaml")

    # --- 2. Extract or compute local variables from cfg ---
    resolution = cfg.resolution
    exr_path = cfg.exr_path
    base_path = cfg.base_path

    # Join paths for normal/color
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    normal_save_path = os.path.join(base_path, cfg.normal_subdir)
    color_save_path = os.path.join(base_path, cfg.color_subdir)
    if not os.path.exists(normal_save_path):
        os.makedirs(normal_save_path, exist_ok=True)
    if not os.path.exists(color_save_path):
        os.makedirs(color_save_path, exist_ok=True)
    ORTHO_FLAG = cfg.ORTHO_FLAG
    ORTHO_SCALE = cfg.ORTHO_SCALE

    # Initialize blenderproc (or whichever library)
    bproc.init()
    bproc.renderer.enable_depth_output(activate_antialiasing=cfg.activate_antialiasing)
    bproc.renderer.set_max_amount_of_samples(cfg.max_amount_of_samples)

    # Set random seed
    random.seed(cfg.random_seed)

    vertex_groups_materials = {
        "Background": "Diffuse_Grey_With_Ambient",
        "Foreground": "Diffuse_Reddish_Brown_With_Ambient",
    }
    # Access main block for camera settings
    elevation = cfg.main.elevation
    distance = cfg.main.distance
    azimuth_range = cfg.main.azimuth_range
    fov = cfg.main.fov
    z_limit = cfg.main.z_limit

    # --- 3. Handle any command-line arguments (if you still want sys.argv) ---
    # e.g. python render.py <current_thread> <total_thread>
    # or swap them if the order is reversed
    if len(sys.argv) >= 3:
        current_thread = int(sys.argv[1])
        total_thread = int(sys.argv[2])
    else:
        print("Usage: python render.py <current_thread> <total_thread>")
        sys.exit(1)

    # --- 4. Example: read in brush_collections from exr_path ---
    brush_collections = [ i[:-4] for i in os.listdir(exr_path) if i.endswith(".ply")]
    print("before", len(brush_collections))

    # Filter out items that already exist
    brush_collections = [ i for i in brush_collections if not os.path.exists(f"{normal_save_path}/{i}_6.png") ]
    print("after", len(brush_collections))

    # --- 5. Split among threads ---
    start_idx = int(len(brush_collections) / total_thread * current_thread)
    end_idx = int(len(brush_collections) / total_thread * (current_thread + 1))
    brush_collections = brush_collections[start_idx:end_idx]
    print("total num collection", len(brush_collections))

    # --- 6. Clear / reset scene ---
    clear_all_objects()
    reset_scene()
    setCircleLight(1000,15,0)
    setLight_ambient(color=(0.1,0.1,0.1,1)) 
    for light in bpy.data.lights:
        light.cycles.cast_shadow = False

    
    for k,brush_name in enumerate(brush_collections):
        print("name",brush_name)
        try:

            load_and_split2(os.path.join(exr_path,f"{brush_name}.ply"))

        except Exception as e:
            print("Error",e)
            continue


        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                obj.name = "Object"
                obj.data.name = "Object"
        imported_obj = bpy.data.objects["Object"]
        bpy.data.objects["Object"].select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        imported_obj.rotation_euler = (-math.radians(90),0,0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


        # Export the object as .obj file
        imported_obj = bpy.data.objects["Object"]


        brown_material = create_material(
            name="Diffuse_Reddish_Brown_With_Ambient",
            base_color=(0.35, 0.15, 0.1, 1),  # Reddish-brown color
            ambient_color=(0.35, 0.15, 0.1, 1),  # Ambient color
            metallic=0.2,
            roughness=0.6,
            specular=0.1
        )

        grey_material = create_material(
            name="Diffuse_Grey_With_Ambient",
            base_color=(0.5, 0.5, 0.5, 1),  # Grey color
            ambient_color=(0.5, 0.5, 0.5, 1),  # Ambient color
            metallic=0.1,
            roughness=0.7,
            specular=0.2
        )


        imported_obj = bpy.data.objects["Object"]
        max_x, max_y, max_z = imported_obj.bound_box[6]
        min_x, min_y, min_z = imported_obj.bound_box[0]
        normalize_z_axis(imported_obj)
        max_x, max_y, max_z = imported_obj.bound_box[6]
        
        ####rotation
        bpy.data.objects["Object"].select_set(True)
        rotation_radians = math.pi / 2
        bpy.data.objects["Object"].rotation_euler[1] += rotation_radians
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


        #############normal rendering
        render_image(azimuth_range,elevation,distance,brush_name)
        reset_scene()
        exit(0)




###COMMAND
# blenderproc run ./renderNormal_training.py 0 1