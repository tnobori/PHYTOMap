#!/usr/bin/env python
# coding: utf-8

import io
import numpy as np
from bigstream import transform
from matplotlib import pyplot as plt
import zarr
from bigstream import affine
import os
from skimage import io
from numcodecs import Blosc


def global_affine_reg(
    impath_fix, impath_mov,
    spacing, downsampling,
    slc,
    min_radius,
    max_radius,
    match_threshold
):
    
    fix_img = io.imread(impath_fix)
    mov_img = io.imread(impath_mov)

    compressor = Blosc(
                    cname='zstd',
                    clevel=9,
                    shuffle=Blosc.BITSHUFFLE,
                )

    fix_zarr = zarr.array(fix_img, chunks=(40,300, 150), compressor=compressor )
    mov_zarr = zarr.array(mov_img, chunks=(40,300, 150), compressor=compressor )

    # Adjust spacing in case you are using reduced resolution images. [xyz]*[downsampling factors]
    fix_spacing = np.array(spacing) * downsampling
    mov_spacing = np.array(spacing) * downsampling
  
    # read data into memory as numpy arrays
    # Why transpose? Images were stored as zyx, but we prefer xyz (metadata is already xyz)
    fix_data = fix_zarr[...].transpose(2, 1, 0)
    mov_data = mov_zarr[...].transpose(2, 1, 0)

    # affine transformation
    global_affine = affine.ransac_affine(
        fix_data, mov_data,
        fix_spacing, mov_spacing,
        min_radius=min_radius, max_radius=max_radius, match_threshold=match_threshold,
    )

    # sanity check: print the result
    print(global_affine)

    # apply the global affine to the moving image
    mov_aligned = transform.apply_global_affine(
        fix_data, mov_data,
        fix_spacing, mov_spacing,
        global_affine,
    )

    #image checking
    from matplotlib import pyplot as plt

    # plot some image slices to check on things
    f_slc = fix_data[..., slc]
    a_slc = mov_aligned[..., slc]
    m_slc = mov_data[..., slc]

    # normalize for display
    f_slc = f_slc.astype(np.float32) / f_slc.max()
    a_slc = a_slc.astype(np.float32) / a_slc.max()
    m_slc = m_slc.astype(np.float32) / m_slc.max()

    # make RGB versions
    f_rgb = np.zeros(f_slc.shape + (3,))
    f_rgb[..., 0] = f_slc * 2
    a_rgb = np.zeros(a_slc.shape + (3,))
    a_rgb[..., 0] = f_slc * 2
    a_rgb[..., 1] = a_slc * 2
    m_rgb = np.zeros(m_slc.shape + (3,))
    m_rgb[..., 1] = m_slc * 2

    # create figure and subplots
    fig = plt.figure(figsize=(48,96))
    fig.add_subplot(3, 1, 1)
    plt.imshow(f_rgb)
    fig.add_subplot(3, 1, 2)
    plt.imshow(f_rgb)
    plt.imshow(a_rgb)
    fig.add_subplot(3, 1, 3)
    plt.imshow(m_rgb)
    plt.show()

    return global_affine, mov_aligned



def ch_submit(channel,
                image_dir,
              image_prefix ,
              out_dir,
              im_round, 
              impath_fix_highres,
              slc, 
              spacing,
              global_affine
              
):

    fix_highres = io.imread(impath_fix_highres)
    mov_highres = io.imread(image_dir  + image_prefix + "CH"+ channel +".tif")

    compressor = Blosc(
                cname='zstd',
                clevel=4,
                shuffle=Blosc.BITSHUFFLE,
            )

    fix_highres_zarr = zarr.array(fix_highres, chunks=(40,300, 150), compressor=compressor )
    mov_highres_zarr = zarr.array(mov_highres, chunks=(40,300, 150), compressor=compressor )


    # Do not need to change this
    fix_highres_spacing = np.array(spacing) * [1,1,1]
    mov_highres_spacing = np.array(spacing) * [1,1,1]

    # read data into memory as numpy arrays
    # Why transpose? Images were stored as zyx, but we prefer xyz (metadata is already xyz)
    fix_highres_data = fix_highres_zarr[...].transpose(2, 1, 0)
    mov_highres_data = mov_highres_zarr[...].transpose(2, 1, 0)

    # apply the global affine to the moving image
    mov_highres_aligned = transform.apply_global_affine(
        fix_highres_data, mov_highres_data,
        fix_highres_spacing, mov_highres_spacing,
        global_affine,
    )

    #image checking
    # plot some image slices to check results
    f_slc = fix_highres_data[..., slc]
    a_slc = mov_highres_aligned[..., slc]
    m_slc = mov_highres_data[..., slc]

    # normalize for display
    f_slc = f_slc.astype(np.float32) / f_slc.max()
    a_slc = a_slc.astype(np.float32) / a_slc.max()
    m_slc = m_slc.astype(np.float32) / m_slc.max()

    # make RGB versions
    f_rgb = np.zeros(f_slc.shape + (3,))
    f_rgb[..., 0] = f_slc * 2
    a_rgb = np.zeros(a_slc.shape + (3,))
    a_rgb[..., 0] = f_slc * 2
    a_rgb[..., 1] = a_slc * 2
    m_rgb = np.zeros(m_slc.shape + (3,))
    m_rgb[..., 1] = m_slc * 2

    # create figure and subplots
    fig = plt.figure(figsize=(48,96))
    fig.add_subplot(3, 1, 1)
    plt.imshow(f_rgb)
    fig.add_subplot(3, 1, 2)
    plt.imshow(f_rgb)
    plt.imshow(a_rgb)
    fig.add_subplot(3, 1, 3)
    plt.imshow(m_rgb)
    plt.show()

            
    mov_highres_aligned = mov_highres_aligned.transpose(2, 1, 0)

    array = mov_highres_aligned

    out_file = out_dir + "/registered_R"+ im_round + "_Ch" + channel +".tif"
    io.imsave(out_file, array) 




def round_submit(
    imdir,
    im_round,
    image_prefix,
    channels,
    slc,
    impath_fix_highres,
    spacing,
    global_affine,
    out_main,
    img_dir
    
    ):
    
    image_dir = imdir + img_dir + "/"
    
    out_main_dir = "./" + out_main
    if not os.path.exists(out_main_dir):
        os.makedirs(out_main_dir)

    out_dir = out_main_dir + "/R" + im_round
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for i in channels:
        ch_submit(channel=i, 
                  image_dir=image_dir, 
                  image_prefix=image_prefix,
                  out_dir=out_dir,
                  im_round=im_round,
                  slc=slc,
                  impath_fix_highres=impath_fix_highres,
                  spacing=spacing,
                  global_affine=global_affine
                  )

