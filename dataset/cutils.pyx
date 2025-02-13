#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

def z_buffering_points(int[::1] xs, int[::1] ys, float[::1] zs, float[:,::1] depth, int[:,::1] ids):
    cdef int imgsize, plen
    imgsize = depth.shape[0]
    plen = xs.shape[0]
    cdef int i,x,y,x_,y_
    cdef float z

    for i in range(plen):
        x  = xs[i]
        y =  ys[i]
        z = zs[i]
        if x>=0 and x<imgsize and y>=0 and y<imgsize and z<depth[x,y]:
            depth[x,y] = z
            ids[x,y] = i
