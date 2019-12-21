#!/usr/bin/env python3

from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import data.preprocess as preprocess
import utils.utils as utils
import os
import glob
import psutil
import argparse
import openslide


def multiprocess_wrapper(args):
    """Function to generate arguments for multiprocess

    Parameters
    ----------
    args : tuple
        Arugments

    Returns
    -------
    Call functions for multiprocess
    """
    return extract_annotated_patches(*args)


def extract_annotated_patches(slide_path, save_dir, slide_label, annotation, patch_size, resize_size):
    """Function to generate patches based on annotated polygon

    Parameters
    ----------
    slide_path : str
        Aboluate path to a slide

    save_dir : str
        Aboluate path to a directory that stores patches

    slide_label : str
        The ground truth label for a whole slide image

    annotation : list
        A list of tuple that contains (label, ploygon)

    patch_size : int
        Extraction patch size

    resize_size : int
        Resizing size

    Returns
    -------
    None
    """
    slide = openslide.OpenSlide(slide_path)
    slide_width, slide_height = slide.level_dimensions[0]
    slide_id = utils.strip_extension(slide_path).split('/')[-1]

    for patch_loc_width in range(0, slide_width - patch_size, patch_size):
        for patch_loc_height in range(0, slide_height - patch_size, patch_size):
            corners = np.array([[patch_loc_width, patch_loc_height], [patch_loc_width+patch_size, patch_loc_height], [
                               patch_loc_width, patch_loc_height+patch_size], [patch_loc_width+patch_size, patch_loc_height+patch_size]])
            for label, polypath in annotation:
                if label == 'Tumor' and np.sum(polypath.contains_points(corners)) == 4:
                    patch = preprocess.extract_and_resize(
                        slide, patch_loc_width, patch_loc_height, patch_size, resize_size)
                    if preprocess.check_luminance(np.asarray(patch)):
                        patch.save(os.path.join(save_dir, '{}_{}.png'.format(
                            patch_loc_width, patch_loc_height)))


def generate_annotated_patches(slide_dir, save_dir, annotation_dir, n_process, patch_size, resize_size):
    """Function to map the extraction function to multiprocess

    Parameters
    ----------
    slide_path : str
        Aboluate path to a slide

    save_dir : str
        Aboluate path to a directory that stores patches

    slide_label : str
        The ground truth label for a whole slide image

    annotation : list
        A list of tuple that contains (label, ploygon)

    patch_size : int
        Extraction patch size

    resize_size : int
        Resizing size

    Returns
    -------
    None
    """
    slides = glob.glob(os.path.join(slide_dir, '**', '*.tiff'))
    annotations = utils.read_annotations(annotation_dir)
    slides = utils.exclude_slides_without_annotations(slides, annotations)

    with Pool(processes=n_process) as p:
        n_slides = len(slides)
        prefix = 'Extracting Tumor Patches: '
        for idx in tqdm(range(0, n_slides, n_process), desc=prefix):
            cur_slides = slides[idx:idx + n_process]
            multiprocess_args = produce_args(
                cur_slides, save_dir, annotations, patch_size=patch_size, resize_size=resize_size)
            p.map(multiprocess_wrapper, multiprocess_args)


def produce_args(slides, save_dir, annotations, patch_size, resize_size):
    """Function to generate arguments

    Parameters
    ----------
    slide_path : str
        Aboluate path to a slide

    save_dir : str
        Aboluate path to a directory that stores patches

    annotations : list
        A list of tuple that contains (label, ploygon)

    patch_size : int
        Extraction patch size

    resize_size : int
        Resizing size

    Returns
    -------
    None
    """
    args = []
    for slide in slides:
        slide_id, slide_label = utils.get_info_from_slide_path(slide)
        if not os.path.exists(os.path.join(save_dir, slide_label.name, slide_id)):
            os.makedirs(os.path.join(save_dir, slide_label.name, slide_id))
        arg = (slide, os.path.join(save_dir, slide_label.name, slide_id),
               slide_label.name, annotations[slide_id], patch_size, resize_size)
        args.append(arg)
    return args


def store_image_to_h5(data_dir, h5f_path):
    """Function to generate arguments

    Parameters
    ----------
    data_dir : str
        Absoluate path to a directory that stores patches

    h5f_path : str
        Absoluate path to store h5 file

    Returns
    -------
    None
    """
    with h5py.File(h5f_path) as h5f_image:
        patch_ids = glob.glob(os.path.join(data_dir, '**', '**', '*.png'))
        prefix = 'Storing Patches: '
        for idx, patch_id in enumerate(tqdm(patch_ids, desc=prefix)):
            if patch_id not in h5f_image:
                cur_image = Image.open(patch_id).convert('RGB')
                patch_id = utils.create_patch_id(patch_id)
                image_grp = h5f_image.require_group(patch_id)
                image_grp.create_dataset(
                    'image_data', data=np.asarray(cur_image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slides_dir', type=str, required=False,
                        default='/projects/ovcare/WSI/Dataset_Slides_500_cases')
    parser.add_argument('--patch_save_dir', type=str, required=False,
                        default='/projects/ovcare/classification/ywang/midl_dataset/768_monoscale_300')
    parser.add_argument('--annotation_dir', type=str, required=False,
                        default='/projects/ovcare/classification/ywang/midl_dataset/annotations')
    parser.add_argument('--h5_save_path', type=str, required=False,
                        default='/projects/ovcare/classification/ywang/midl_dataset/768_monoscale_300.hdf5')
    parser.add_argument('--patch_ids_save_path', type=str, required=False,
                        default='/projects/ovcare/classification/ywang/midl_dataset/768_monoscale_300_patch_ids.txt')
    parser.add_argument('--patch_size', type=int, required=True)
    parser.add_argument('--resize_size', type=int, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.patch_save_dir):
        os.makedirs(args.patch_save_dir)

    n_process = psutil.cpu_count()
    utils.make_subtype_dirs(args.patch_save_dir)
    generate_annotated_patches(args.slides_dir, args.patch_save_dir,
                               args.annotation_dir, n_process=n_process, patch_size=args.patch_size, resize_size=args.resize_size)
    store_image_to_h5(args.patch_save_dir, args.h5_save_path)
    utils.export_h5_ids(args.h5_save_path, args.patch_ids_save_path)
