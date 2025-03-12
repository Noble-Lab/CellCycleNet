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
    '''
    Loads tile + mask --> calculates median non-zero pixel value + median nuclear diameter in that tile.
        Args:
            - args [list of str] : image and mask filenames
        Out:
            - median_pixel [float] : median non-zero pixel value
            - median_dims [np.ndarray] : median nuclear diameter along Z,Y,X axes
    '''
    # load image + mask #
    image_fn = args[0]
    mask_fn = args[1]
    is_3d = args[2]
    image = imread(image_fn)
    mask = imread(mask_fn)

    # calculate median nonzero pixel value #
    nonzero_pixels = image[mask != 0]
    median_pixel = np.median(nonzero_pixels).astype(int)

    # calculate median nuclear dims #
    prop = regionprops(mask)
    if is_3d:
        dims = np.array([(prop[k].bbox[3]-prop[k].bbox[0], # Z
                          prop[k].bbox[4]-prop[k].bbox[1], # Y
                          prop[k].bbox[5]-prop[k].bbox[2]) # X
                          for k in range(len(prop))])
    else:
        dims = np.array([(prop[k].bbox[2]-prop[k].bbox[0], # Y
                          prop[k].bbox[3]-prop[k].bbox[1]) # X
                          for k in range(len(prop))])
    median_dims = np.median(dims, axis=0).astype(int)

    return (median_pixel, median_dims)


def _calc_norm_and_scale_factor(image_fns, mask_fns, num_cores, is_3d):
    '''
    Uses DAPI images + corresponding masks to calculate median non-zero pixel value + median nuclear diameter across entire dataset.
        Args:
            - image_fns [list] : filenames for DAPI images (format as 'tile_n.tif')
            - mask_fns [list] : filenames for segmentation masks (format as 'mask_n.tif')
            - num_cores [int] : number of cores to use for parallel processing (default: None uses a single core)
            - is_3d [bool] : flag to determine if images are 3D or 2D
        Out:
            - norm_factor [float] : median non-zero pixel value across entire dataset
            - scale_factor [np.ndarray] : downsampling factors along each axis to apply to user's images such that they match dim. of pretraining images
    '''
    # package image and mask fns together for multiprocessing compatiability #
    image_and_mask_fns = [(i_fn, m_fn, is_3d) for i_fn, m_fn in zip(image_fns, mask_fns)]

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
    if is_3d:
        pretrained_dims = np.array([16, 37, 37])
    else:
        pretrained_dims = np.array([37, 37])
    user_dims = np.median(median_dims, axis=0)
    max_dims = np.max(median_dims, axis=0)
    scale_factor = user_dims / pretrained_dims
    
    return norm_factor, scale_factor, max_dims

####################################################################################################

def _gen_SNI(args):
    '''
    Generates unlabeled single-nucleus images for a given tile.
        Args:
            - args [list] : contains image filename, mask filename, path to output directory, normalization factor, and scaling factor
        Out:
            - None
    '''
    # load image / mask pair #
    image = imread(args[0])
    mask = imread(args[1])

    output_dir = args[2]
    norm_factor = args[3]
    scale_factor = args[4]
    max_dims = args[5]
    is_3d = args[6]

    # set padding to 1/8 of max dims #
    if is_3d: z, y, x = [1.05*dim for dim in max_dims]
    else:        y, x = [1.05*dim for dim in max_dims]

    # for each object in the mask... #
    obj_nums = np.unique(mask)[1:]
    for obj_num in obj_nums:

        # isolate the current object #
        obj_mask = (mask == obj_num)
        obj = np.where(obj_mask, image, 0)

        # crop out empty rows/columns/planes #
        if is_3d:
            full_rows = np.any(obj, axis=(1,2))
            full_columns = np.any(obj, axis=(0,2))
            full_stacks = np.any(obj, axis=(0,1))
            obj_crop = obj[full_rows][:, full_columns][:, :, full_stacks]
        else:
            full_rows = np.any(obj, axis=1)
            full_columns = np.any(obj, axis=0)
            obj_crop = obj[full_rows][:, full_columns]

        # normalize image based on norm factor #
        obj_norm = obj_crop / norm_factor

        # rescale unpadded image based on scale factor #
        out_dims = np.array([int(in_dim / scale) for in_dim, scale in zip(obj_crop.shape, scale_factor)])
        obj_rescale = resize_local_mean(obj_norm, out_dims)

        # pad image to have 1/8 of its diameter on each side #
        if is_3d:
            obj_z, obj_y, obj_x = obj_rescale.shape
            z_pad = (int((z - obj_z) / 2), int((z - obj_z + 1) / 2))
            y_pad = (int((y - obj_y) / 2), int((y - obj_y + 1) / 2))
            x_pad = (int((x - obj_x) / 2), int((x - obj_x + 1) / 2))
            obj_pad = np.pad(obj_rescale, [z_pad, y_pad, x_pad])
        else:
            obj_y, obj_x = obj_rescale.shape
            y_pad = (int((y - obj_y) / 2), int((y - obj_y + 1) / 2))
            x_pad = (int((x - obj_x) / 2), int((x - obj_x + 1) / 2))
            obj_pad = np.pad(obj_rescale, [y_pad, x_pad])

        # save #
        imwrite(f'{output_dir}/{os.path.splitext(os.path.basename(args[0]))[0]}_obj_{obj_num}.tif', obj_pad)


def _gen_SNI_label(args):
    '''
    Generates labeled single-nucleus images for a given tile.
        Args:
            - args [list] : contains image filename, mask filename, label array filename, path to output directory, normalization factor, and scaling factor
        Out:
            - None
    '''
    # load image / mask pair #
    image = imread(args[0])
    mask = imread(args[1])
    label_arr = imread(args[2])

    output_dir = args[3]
    norm_factor = args[4]
    scale_factor = args[5]
    max_dims = args[6]
    is_3d = args[7]

    # set padding to 1/20th of max dims #
    if is_3d: z, y, x = [1.05*dim for dim in max_dims]
    else:        y, x = [1.05*dim for dim in max_dims]

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
        if is_3d:
            full_rows = np.any(obj, axis=(1,2))
            full_columns = np.any(obj, axis=(0,2))
            full_stacks = np.any(obj, axis=(0,1))
            obj_crop = obj[full_rows][:, full_columns][:, :, full_stacks]
        else:
            full_rows = np.any(obj, axis=1)
            full_columns = np.any(obj, axis=0)
            obj_crop = obj[full_rows][:, full_columns]

        # normalize image based on norm factor #
        obj_norm = obj_crop / norm_factor

        # rescale unpadded image based on scale factor #
        out_dims = np.array([int(in_dim / scale) for in_dim, scale in zip(obj_crop.shape, scale_factor)])
        obj_rescale = resize_local_mean(obj_norm, out_dims)

        # pad image to have 1/8 of its diameter on each side #
        if is_3d:
            obj_z, obj_y, obj_x = obj_rescale.shape
            z_pad = (int((z - obj_z) / 2), int((z - obj_z + 1) / 2))
            y_pad = (int((y - obj_y) / 2), int((y - obj_y + 1) / 2))
            x_pad = (int((x - obj_x) / 2), int((x - obj_x + 1) / 2))
            obj_pad = np.pad(obj_rescale, [z_pad, y_pad, x_pad])
        else:
            obj_y, obj_x = obj_rescale.shape
            y_pad = (int((y - obj_y) / 2), int((y - obj_y + 1) / 2))
            x_pad = (int((x - obj_x) / 2), int((x - obj_x + 1) / 2))
            obj_pad = np.pad(obj_rescale, [y_pad, x_pad])

        # save #
        imwrite(f'{output_dir}/{os.path.splitext(os.path.basename(args[0]))[0]}_obj_{obj_num}_class_{label_str}.tif', obj_pad)


def _gen_DF(SNI_fns, with_labels):
    '''
    Generates a dataframe containing a tile number, object number, filename, (and GT label) for each generated SNI.
        Args:
            - SNI_fns [list] : filenames for each generated SNI
            - with_labels [bool] : flag to determine if a column for GT labels is added to the dataframe
        Out:
            - df [pd.DataFrame] : SNI dataframe
    '''
    # extract tile num and object num from SNI filename #
    extract_numbers = lambda filename : list(map(int, re.findall(r'\d+', filename)))
    nums = np.asarray([extract_numbers(fn.split('/')[-1]) for fn in SNI_fns])
    tile_nums = nums[:,0]
    obj_nums = nums[:,1]

    # create dict for dataframe #
    df_dict = {'tile_num': tile_nums,
               'obj_num': obj_nums,
               'filename': SNI_fns}
    
    # if SNIs are labeled, create entry for GT labels #
    if with_labels:
        labels = np.asarray(['G1' if fn.split('_')[-1].split('.')[0] == 'G1' else 'S/G2' for fn in SNI_fns])
        df_dict['label'] = labels

    # save dataframe #
    df_fn = 'CCN_single_nuclei_labeled.csv' if with_labels else 'CCN_single_nuclei.csv'
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(df_fn, index=False)
    return df


def _validate_paths(dir):
    '''
    Ensures that user inputted paths are exist and contain valid images, raises errors accordingly.
        Args:
            - dir [str] : path to user specified directory
        Out:
            - None
    '''
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

def generate_images(image_dir, mask_dir, output_dir=None, return_df=False, num_cores=None, is_3d=True):
    '''
    Generates unlabeled single-nucleus images from tiled data for input to model for prediction only.
        Args:
            - image_dir [str] : path to directory where DAPI images are located
            - mask_dir [str] : path to directory where segmentation masks are located
            - output_dir [str] : path to directory where SNIs will be saved (default: None creates a new directory named 'single_nucleus_images')
            - return_df [bool] : flag to choose if function returns SNI dataframe (default: False)
            - num_cores [int] : number of cores to use for parallel processing (default: None uses a single core)
        Out:
            - df [pd.DataFrame] : dataframe containing SNI tile + object numbers and filenames (if return_df = True)
    '''
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
    norm_factor, scale_factor, max_dims = _calc_norm_and_scale_factor(image_fns, mask_fns, num_cores, is_3d)

    # generate SNIs #
    if num_cores == None:
        for image_fn, mask_fn in zip(image_fns, mask_fns):
            _gen_SNI([image_fn, mask_fn, output_dir, norm_factor, scale_factor, max_dims, is_3d])
    else:
        with Pool(num_cores) as pool:
            pool.map(_gen_SNI, [(image_fn, mask_fn, output_dir, norm_factor, scale_factor, max_dims, is_3d) for image_fn, mask_fn in zip(image_fns, mask_fns)])

    # generate DF #
    SNI_fns = sorted([os.path.join(output_dir, fn) for fn in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, fn))])
    df = _gen_DF(SNI_fns, with_labels=False)
    if return_df: return df

####################################################################################################

def generate_images_labeled(image_dir, mask_dir, label_dir, output_dir=None, return_df=False, num_cores=None, is_3d=True):
    '''
    Generates labeled single-nucleus images from tiled data for input to model for prediction or fine-tuning.
        Args:
            - image_dir [str] : path to directory where DAPI images are located
            - mask_dir [str] : path to directory where segmentation masks are located
            - label_dir [str] : path to directory where label arrays are located
            - output_dir [str] : path to directory where SNIs will be saved (default: None creates a new directory named 'single_nucleus_images')
            - return_df [bool] : flag to choose if function returns SNI dataframe (default: False)
            - num_cores [int] : number of cores to use for parallel processing (default: None uses a single core)
        Out:
            - df [pd.DataFrame] : dataframe containing SNI tile + object numbers and filenames (if return_df = True)
    '''
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
    norm_factor, scale_factor, max_dims = _calc_norm_and_scale_factor(image_fns, mask_fns, num_cores, is_3d)

    # generate SNIs #
    if num_cores == None:
        for image_fn, mask_fn, label_fn in zip(image_fns, mask_fns, label_fns):
            _gen_SNI_label([image_fn, mask_fn, label_fn, output_dir, norm_factor, scale_factor, max_dims, is_3d])
    else:
        with Pool(num_cores) as pool:
            pool.map(_gen_SNI_label, [(image_fn, mask_fn, label_fn, output_dir, norm_factor, scale_factor, max_dims, is_3d) for image_fn, mask_fn, label_fn in zip(image_fns, mask_fns, label_fns)])

    # generate DF #
    SNI_fns = sorted([os.path.join(output_dir, fn) for fn in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, fn))])
    df = _gen_DF(SNI_fns, with_labels=True)
    if return_df: return df