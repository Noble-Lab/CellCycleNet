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
from skimage.transform import downscale_local_mean
from time import time
from datetime import datetime
from multiprocessing.pool import Pool

####################################################################################################

def _get_median_pixel(fn):
    '''Return median value of non-zero pixels from a given single nucleus image.'''
    image = imread(fn)
    nonzero_pixels = image[image != 0]
    median_pixel = np.median(nonzero_pixels)
    return median_pixel   

def calc_norm_factor(df, num_cores=None):
    '''
    Calculate normalization factor (median value of median non-zero pixel values for each single nucleus image).
        Args:
            - image_dir [str] : directory of single nucleus images with glob-style wildcard 
            - num_cores [int] : number of cores to use for multiprocessing; default None will not use multiprocessing
        Out:
            - norm_factor [float] : normalization factor to input to CellCycleNet.create_datasets()
    '''

    ### FIXME ###
    # small dataset for testing #
    image_fns = df['filename'].values

    if num_cores == None:
        all_medians = [_get_median_pixel(fn) for fn in image_fns]
    else:
        with Pool(num_cores) as pool:
            all_medians = pool.map(_get_median_pixel, image_fns)

    norm_factor = np.median(all_medians)
    return norm_factor

####################################################################################################

def calc_scale_factor():
    pass

####################################################################################################

def _gen_SNI(args):
    # load image / mask pair #
    image = imread(args[0])
    mask = imread(args[1])

    output_dir = args[2]
    shape = args[3]

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

        # pad object to get consistent dimensions #
        x = shape[2]; y = shape[1]; z = shape[0]
        obj_z, obj_y, obj_x = obj_crop.shape
        x_add = (int(floor((x - obj_x) / 2)), int(ceil((x - obj_x) / 2)))
        y_add = (int(floor((y - obj_y) / 2)), int(ceil((y - obj_y) / 2)))
        z_add = (int(floor((z - obj_z) / 2)), int(ceil((z - obj_z) / 2)))
        obj_pad = np.pad(obj_crop, [z_add, y_add, x_add])

        # write image #
        imwrite(f'{output_dir}{args[0].split("/")[-1].split(".")[0]}_obj_{obj_num}.tif', obj_pad)


def _gen_SNI_label(args):
    # load image / mask pair #
    image = imread(args[0])
    mask = imread(args[1])
    label_arr = imread(args[2])

    output_dir = args[3]
    shape = args[4]

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

        # pad object to get consistent dimensions #
        x = shape[2]; y = shape[1]; z = shape[0]
        obj_z, obj_y, obj_x = obj_crop.shape
        x_add = (int(floor((x - obj_x) / 2)), int(ceil((x - obj_x) / 2)))
        y_add = (int(floor((y - obj_y) / 2)), int(ceil((y - obj_y) / 2)))
        z_add = (int(floor((z - obj_z) / 2)), int(ceil((z - obj_z) / 2)))
        obj_pad = np.pad(obj_crop, [z_add, y_add, x_add])

        # write image #
        imwrite(f'{output_dir}{args[0].split("/")[-1].split(".")[0]}_obj_{obj_num}_class_{label_str}.tif', obj_pad)


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


def generate_images(image_dir, mask_dir, output_dir, shape=(90,150,150), num_cores=None):
    
    image_fns = sorted(glob(f'{image_dir}*'))
    mask_fns = sorted(glob(f'{mask_dir}*'))

    if num_cores == None:
        for image_fn, mask_fn in zip(image_fns, mask_fns):
            _gen_SNI([image_fn, mask_fn, output_dir, shape])

    else:
        with Pool(num_cores) as pool:
            pool.map(_gen_SNI, [(image_fn, mask_fn, output_dir, shape) for image_fn, mask_fn in zip(image_fns, mask_fns)])

    SNI_fns = sorted(glob(f'{output_dir}*'))
    df = _gen_DF(SNI_fns, with_labels=False)
    return df


def generate_images_labeled(image_dir, mask_dir, label_dir, output_dir, shape=(90,150,150), num_cores=None):
    
    image_fns = sorted(glob(f'{image_dir}*'))
    mask_fns = sorted(glob(f'{mask_dir}*'))
    label_fns = sorted(glob(f'{label_dir}*'))

    if num_cores == None:
        for image_fn, mask_fn, label_fn in zip(image_fns, mask_fns, label_fns):
            _gen_SNI_label([image_fn, mask_fn, label_fn, output_dir, shape])

    else:
        with Pool(num_cores) as pool:
            pool.map(_gen_SNI, [(image_fn, mask_fn, label_fn, output_dir, shape) for image_fn, mask_fn in zip(image_fns, mask_fns, label_fns)])

    SNI_fns = sorted(glob(f'{output_dir}*'))
    df = _gen_DF(SNI_fns, with_labels=True)
    return df
