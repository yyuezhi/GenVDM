import numpy as np
import cv2
import os

import trimesh
import os
import numpy as np
import argparse


def normalize_obj(name):
    mesh = trimesh.load(os.path.join(source_dir,f"{name}.glb"))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.faces)
    
    #remove isolated points
    vertices_mapping = np.full([len(vertices)], -1, np.int32)
    for i in range(len(triangles)):
        for j in range(3):
            vertices_mapping[triangles[i,j]] = 1
    counter = 0
    for i in range(len(vertices)):
        if vertices_mapping[i]>0:
            vertices_mapping[i] = counter
            counter += 1
    vertices = vertices[vertices_mapping>=0]
    triangles = vertices_mapping[triangles]
    
    
    #normalize diagonal=1
    x_max = np.max(vertices[:,0])
    y_max = np.max(vertices[:,1])
    z_max = np.max(vertices[:,2])
    x_min = np.min(vertices[:,0])
    y_min = np.min(vertices[:,1])
    z_min = np.min(vertices[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = max(max(x_scale,y_scale),z_scale)
    
    vertices[:,0] = (vertices[:,0]-x_mid)/scale
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.export(os.path.join(target_dir,f"{name}.obj"))

    

source_dir = "./mesh/"
target_dir = "./voxel/"

obj_names = [i[:-4] for i in os.listdir(source_dir) if ".glb" in i]



for i in range(len(obj_names)):
    normalize_obj(obj_names[i])
    this_name = target_dir + f"{obj_names[i]}.obj"
    print(i,this_name)

    maxx = 1.0
    maxy = 1.0
    maxz = 1.0
    minx = -1.0
    miny = -1.0
    minz = -1.0

    command = "./binvox -bb "+str(minx)+" "+str(miny)+" "+str(minz)+" "+str(maxx)+" "+str(maxy)+" "+str(maxz)+" "+" -d 1024 -e "+this_name

    os.system(command)
