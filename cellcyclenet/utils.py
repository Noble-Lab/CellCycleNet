import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import re
import os
import ast
from math import floor, ceil
from tifffile import imread, imwrite
from glob import glob
from skimage.transform import downscale_local_mean, resize_local_mean
from skimage.measure import regionprops
from time import time
from datetime import datetime
from multiprocessing.pool import Pool

####################################################################################################

def _load_and_calc_params(args):
    '''Loads tile + mask --> calculates median pixel and median nuclear dims (avoids loading dataset twice).'''
    # load image + mask #
    image_fn = args[0]
    mask_fn = args[1]
    image = imread(image_fn)
    mask = imread(mask_fn)

    # calculate median nonzero pixel value #
    nonzero_pixels = image[mask != 0]
    median_pixel = np.median(nonzero_pixels).astype(int)

    # calculate median nuclear dims #
    prop = regionprops(mask)
    dims = np.array([(prop[k].bbox[3]-prop[k].bbox[0], # Z
                      prop[k].bbox[4]-prop[k].bbox[1], # Y
                      prop[k].bbox[5]-prop[k].bbox[2]) # X
                      for k in range(len(prop))])
    median_dims = np.median(dims, axis=0).astype(int)

    return (median_pixel, median_dims)


def _calc_norm_and_scale_factor(image_fns, mask_fns, num_cores):
    '''
    docstring goes here
    '''
    # package image and mask fns together for multiprocessing compatiability #
    image_and_mask_fns = [(i_fn, m_fn) for i_fn, m_fn in zip(image_fns, mask_fns)]

    # compute median non-zero pixel value for each tile #
    if num_cores == None:
        median_pixels_and_dims = [_load_and_calc_params(fns) for fns in image_and_mask_fns]
    else:
        with Pool(num_cores) as pool:
            median_pixels_and_dims = pool.map(_load_and_calc_params, image_and_mask_fns)

    # separate out median pixel values + median dims #
    median_pixels = [elmt[0] for elmt in median_pixels_and_dims]
    median_dims = [elmt[1] for elmt in median_pixels_and_dims]

    # norm factor is median of medians #
    norm_factor = np.median(median_pixels)

    # scale factor is median of nuclear dims #
    pretrained_dims = np.array([16, 37, 37]) # FIXME replace with actual values 
    user_dims = np.median(median_dims, axis=0)
    scale_factor = user_dims / pretrained_dims
    
    return norm_factor, scale_factor

####################################################################################################

def _gen_SNI(args):
    # load image / mask pair #
    image = imread(args[0])
    mask = imread(args[1])

    output_dir = args[2]
    norm_factor = args[3]
    scale_factor = args[4]

    # for each object in the mask... #
    obj_nums = np.unique(mask)[1:]
    for obj_num in obj_nums:

        # isolate the current object #
        obj_mask = (mask == obj_num)
        obj = np.where(obj_mask, image, 0)

        # crop out empty rows/columns/planes #
        full_rows = np.any(obj, axis=(1,2))
        full_columns = np.any(obj, axis=(0,2))
        full_stacks = np.any(obj, axis=(0,1))
        obj_crop = obj[full_rows][:, full_columns][:, :, full_stacks]

        # normalize image based on norm factor #
        obj_norm = obj_crop / norm_factor

        # rescale unpadded image based on scale factor #
        out_dims = np.array([int(in_dim / scale) for in_dim, scale in zip(obj_crop.shape, scale_factor)])
        obj_rescale = resize_local_mean(obj_norm, out_dims)

        # pad object to get consistent dimensions #
        x = 75; y = 75; z = 45 # size of images used for training CCN #
        obj_z, obj_y, obj_x = obj_rescale.shape

        # image larger than expected dimensions #
        if (obj_x > x) or (obj_y > y) or (obj_z > z):
            os.makedirs(f'{output_dir}/large_objects/', exist_ok=True)
            imwrite(f'{output_dir}/large_objects/{os.path.splitext(os.path.basename(args[0]))[0]}_obj_{obj_num}.tif', obj_rescale)

        # else pad + write image #
        else:
            x_add = (int(floor((x - obj_x) / 2)), int(ceil((x - obj_x) / 2)))
            y_add = (int(floor((y - obj_y) / 2)), int(ceil((y - obj_y) / 2)))
            z_add = (int(floor((z - obj_z) / 2)), int(ceil((z - obj_z) / 2)))
            obj_pad = np.pad(obj_rescale, [z_add, y_add, x_add])
            imwrite(f'{output_dir}/{os.path.splitext(os.path.basename(args[0]))[0]}_obj_{obj_num}.tif', obj_pad)


def _gen_SNI_label(args):
    # load image / mask pair #
    image = imread(args[0])
    mask = imread(args[1])
    label_arr = imread(args[2])

    output_dir = args[3]
    norm_factor = args[4]
    scale_factor = args[5]

    # for each object in the mask... #
    obj_nums = np.unique(mask)[1:]
    for obj_num in obj_nums:

        # isolate the current object #
        obj_mask = (mask == obj_num)
        obj = np.where(obj_mask, image, 0)

        # get label from label array #
        label = label_arr[obj_mask][0]
        label_str = 'G1' if label == 1 else 'S-G2'

        # crop out empty rows/columns/planes #
        full_rows = np.any(obj, axis=(1,2))
        full_columns = np.any(obj, axis=(0,2))
        full_stacks = np.any(obj, axis=(0,1))
        obj_crop = obj[full_rows][:, full_columns][:, :, full_stacks]

        # normalize image based on norm factor #
        obj_norm = obj_crop / norm_factor

        # rescale unpadded image based on scale factor #
        out_dims = np.array([int(in_dim / scale) for in_dim, scale in zip(obj_crop.shape, scale_factor)])
        obj_rescale = resize_local_mean(obj_norm, out_dims)

        # pad object to get consistent dimensions #
        x = 75; y = 75; z = 45 # size of images used for training CCN #
        obj_z, obj_y, obj_x = obj_rescale.shape

        # image larger than expected dimensions #
        if (obj_x > x) or (obj_y > y) or (obj_z > z):
            os.makedirs(f'{output_dir}/large_objects/', exist_ok=True)
            imwrite(f'{output_dir}/large_objects/{os.path.splitext(os.path.basename(args[0]))[0]}_obj_{obj_num}_class_{label_str}.tif', obj_rescale)

        # else pad + write image #
        else:
            x_add = (int(floor((x - obj_x) / 2)), int(ceil((x - obj_x) / 2)))
            y_add = (int(floor((y - obj_y) / 2)), int(ceil((y - obj_y) / 2)))
            z_add = (int(floor((z - obj_z) / 2)), int(ceil((z - obj_z) / 2)))
            obj_pad = np.pad(obj_rescale, [z_add, y_add, x_add])
            imwrite(f'{output_dir}/{os.path.splitext(os.path.basename(args[0]))[0]}_obj_{obj_num}_class_{label_str}.tif', obj_pad)


def _gen_DF(SNI_fns, with_labels):
    
    extract_numbers = lambda filename : list(map(int, re.findall(r'\d+', filename)))
    nums = np.asarray([extract_numbers(fn.split('/')[-1]) for fn in SNI_fns])
    tile_nums = nums[:,0]
    obj_nums = nums[:,1]

    df_dict = {'tile_num': tile_nums,
               'obj_num': obj_nums,
               'filename': SNI_fns}
    
    if with_labels:
        labels = np.asarray(['G1' if fn.split('_')[-1].split('.')[0] == 'G1' else 'S/G2' for fn in SNI_fns])
        df_dict['label'] = labels

    df_fn = 'CCN_single_nuclei_labeled.csv' if with_labels else 'CCN_single_nuclei.csv'
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(df_fn, index=False)
    return df


def _validate_paths(dir):

    # check that provided dirs exist #
    if not os.path.exists(dir):
        raise FileNotFoundError(f'The path {dir} does not exist.')
    if not os.path.isdir(dir):
        raise NotADirectoryError(f'The path {dir} is not a directory.')
    
    # check that provided dirs are full of .tif files only #
    for fn in os.listdir(dir):
        if os.path.isfile(fn):
            ext = os.path.splitext(fn)[1].lower()
            if not ext in ['.tif', '.tiff']:
                raise ValueError(f'Expected a .tif or .tiff file, but got a {ext} file ({os.path.join(dir, fn)}).')

####################################################################################################

def generate_images(image_dir, mask_dir, output_dir=None, return_df=False, num_cores=None):
    
    # convert path to abs path #
    image_dir = os.path.abspath(image_dir)
    mask_dir = os.path.abspath(mask_dir)

    # if user does not give output dir --> make one for them #
    if output_dir == None:
        output_dir = os.path.join(os.path.dirname(image_dir), 'single_nucleus_images/')
        os.mkdir(output_dir)

    # if user gives output dir... #
    else:
        # if given dir does not exist --> make it for them #
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.abspath(output_dir)

    # error handling on input paths #
    for path in [image_dir, mask_dir, output_dir]:
        _validate_paths(path)

    # collate image and mask fns #
    image_fns = sorted([os.path.join(image_dir, fn) for fn in os.listdir(image_dir)])
    mask_fns = sorted([os.path.join(mask_dir, fn) for fn in os.listdir(mask_dir)])

    # calculate normalization and scaling factors #
    norm_factor, scale_factor = _calc_norm_and_scale_factor(image_fns, mask_fns, num_cores)

    # generate SNIs #
    if num_cores == None:
        for image_fn, mask_fn in zip(image_fns, mask_fns):
            _gen_SNI([image_fn, mask_fn, output_dir, norm_factor, scale_factor])
    else:
        with Pool(num_cores) as pool:
            pool.map(_gen_SNI, [(image_fn, mask_fn, output_dir, norm_factor, scale_factor) for image_fn, mask_fn in zip(image_fns, mask_fns)])

    # generate DF #
    SNI_fns = sorted([os.path.join(output_dir, fn) for fn in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, fn))])
    df = _gen_DF(SNI_fns, with_labels=False)
    if return_df: return df

####################################################################################################

def generate_images_labeled(image_dir, mask_dir, label_dir, output_dir=None, return_df=False, num_cores=None):

    # convert path to abs path #
    image_dir = os.path.abspath(image_dir)
    mask_dir = os.path.abspath(mask_dir)
    label_dir = os.path.abspath(label_dir)

    # if user does not give output dir --> make one for them #
    if output_dir == None:
        output_dir = os.path.join(os.path.dirname(image_dir), 'single_nucleus_images/')
        os.mkdir(output_dir)

    # if user gives output dir... #
    else:
        # if given dir does not exist --> make it for them #
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = os.path.abspath(output_dir)

    # error handling on input paths #
    for path in [image_dir, mask_dir, label_dir, output_dir]:
        _validate_paths(path)

    # collate image, mask, and label array fns #
    image_fns = sorted([os.path.join(image_dir, fn) for fn in os.listdir(image_dir)])
    mask_fns = sorted([os.path.join(mask_dir, fn) for fn in os.listdir(mask_dir)])
    label_fns = sorted([os.path.join(label_dir, fn) for fn in os.listdir(label_dir)])

    # calculate normalization and scaling factors #
    norm_factor, scale_factor = _calc_norm_and_scale_factor(image_fns, mask_fns, num_cores)

    # generate SNIs #
    if num_cores == None:
        for image_fn, mask_fn, label_fn in zip(image_fns, mask_fns, label_fns):
            _gen_SNI_label([image_fn, mask_fn, label_fn, output_dir, norm_factor, scale_factor])
    else:
        with Pool(num_cores) as pool:
            pool.map(_gen_SNI_label, [(image_fn, mask_fn, label_fn, output_dir, norm_factor, scale_factor) for image_fn, mask_fn, label_fn in zip(image_fns, mask_fns, label_fns)])

    # generate DF #
    SNI_fns = sorted([os.path.join(output_dir, fn) for fn in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, fn))])
    df = _gen_DF(SNI_fns, with_labels=True)
    if return_df: return df