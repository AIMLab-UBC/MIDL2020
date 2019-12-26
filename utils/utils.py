from utils.subtype_enum import SubtypeEnum
from pynvml import *
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import glob
import numpy as np
import matplotlib.path as pltPath
import re
import os
import random
import torch
import copy
import collections
import h5py
import json

PATIENT_REGEX = re.compile(r"^[A-Z]*-?(\d*).*\(?.*\)?.*$")


def make_subtype_dirs(dataset_dir):
    """Function to make directories for each subtype

    Parameters
    ----------
    dataset_dir : string
        Absoluate path to a dataset 

    Returns
    -------
    None
    """
    subtype_names = [s.name for s in SubtypeEnum]
    for subtype_name in subtype_names:
        subtype_dir = os.path.join(dataset_dir, subtype_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        if not os.path.exists(subtype_dir):
            os.makedirs(subtype_dir)


def latex_table_formatter(acc_per_subtype, overall_acc, overall_kappa):
    """Function to format the validation / testing results for LaTeX docs

    Parameters
    ----------
    acc_per_subtype : array_like (float)
        Accuracy for each subtype, index by SubtypeEnum

    overall_acc : float
        Overall accuracy

    overall_kappa : float
        Overall kappa

    Returns
    -------
    None
    Print the LaTex format: & {CC} & {LGSC} & {EC} & {MC} & {HGSC} & {Accuracy} & {Kappa}
    """
    acc_per_subtype = np.asarray(acc_per_subtype) * 100
    print('& {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.4f} \\'.format(
        acc_per_subtype[0], acc_per_subtype[1], acc_per_subtype[2], acc_per_subtype[3], acc_per_subtype[4], overall_acc * 100, overall_kappa) + '\\')


def exclude_slides_without_annotations(slides, annotations):
    """Function to exclude the whole slide images (*.tiff) that do not have annotations (*.txt)

    Parameters
    ----------
    slides : list (string)
        List of absolute paths to the whole slide images (*.tiff)

    annotations : dict
        Dictionary of slide ids that have annotation object

    Returns
    -------
    slide_have_annotations : list (string)
        List of absolute paths to the whole slide images (*.tiff) that have annotations (*.txt)
    """
    slides_have_annotations = []
    print_no_annotation_header = False
    for slide in slides:
        slide_id, slide_label = get_info_from_slide_path(slide)
        if slide_id in annotations:
            slides_have_annotations.append(slide)
        else:
            if not print_no_annotation_header:
                print('Following slides do not have annotations:')
                print_no_annotation_header = True
            print(slide_id, end=', ')
    if print_no_annotation_header:
        print('\n-----------------------------------------')
    return slides_have_annotations


def is_empty_file(file_path):
    """Function to check a file is empty or not

    Parameters
    ----------
    file_path : string
        Absoluate path to a file

    Returns
    -------
    is_empty : bool
        Empty or not
    """
    return os.stat(file_path).st_size == 0


def read_annotations(annotation_dir):
    """Function to read annotation files (*.txt)

    Parameters
    ----------
    annotation_dir : string
        Absoluate path to the directory contains annotation files

    Returns
    -------
    annotations : dict
        Dictionary maps slide ids to their annotation
        {'slide_id': [(type, [polygon vertices])]}
    """
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.txt'))
    annotations = {}
    for annotation_file in annotation_files:
        if is_empty_file(annotation_file):
            continue
        slide_id = strip_extension(annotation_file).split('/')[-1]
        annotations[slide_id] = get_annotation_polygons(annotation_file)
    return annotations


def get_info_from_slide_path(path, slide_id_idx=-1):
    """Function to obtain slide id and slide label from an absoluate path to slide

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    slide_id_idx (optional) : int
        Index of slide id after split path by `/`

    Returns
    -------
    slide_id : string
        Slide id
    slide_label : Enum
        Slide label
    """

    if 'clear_cell_carcinoma_100' in path:
        slide_label = SubtypeEnum['CC']
    elif 'endometrioid_carcinoma_100' in path:
        slide_label = SubtypeEnum['EC']
    elif 'high_grade_serous_carcinoma_300' in path:
        slide_label = SubtypeEnum['HGSC']
    elif 'mucinous_carcinoma_50' in path:
        slide_label = SubtypeEnum['MC']
    else:
        slide_label = SubtypeEnum['LGSC']

    slide_path = strip_extension(path)
    slide_id = slide_path.split('/')[slide_id_idx]

    return slide_id, slide_label


def strip_extension(path):
    """Function to strip file extension

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = Path(path)
    return str(p.with_suffix(''))


def create_patch_id(path):
    """Function to create patch id

    Parameters
    ----------
    path : string
        Absoluate path to a patch

    Returns
    -------
    patch_id : string
        Remove useless information before patch id for h5 file storage
    """
    label_idx = -3
    patch_id = strip_extension(path).split('/')[label_idx:]
    patch_id = '/'.join(patch_id)
    return patch_id


def create_multiscale_ids(patch_id):
    """Function to create multiscale patch ids

    Parameters
    ----------
    patch_id : string
        For multiscale patch, patch id has format: /subtype/slide_id/magnification/patch_location

    Returns
    -------
    patch_20x_id, patch_10x_id, patch_5x_id : string, string, string
        Three magnification patch ids for reading image data from h5
    """
    patch_info = patch_id.split('/')
    patch_info[-2] = '20'
    patch_20x_id = '/'.join(patch_info)
    patch_info[-2] = '10'
    patch_10x_id = '/'.join(patch_info)
    patch_info[-2] = '5'
    patch_5x_id = '/'.join(patch_info)
    return patch_20x_id, patch_10x_id, patch_5x_id


def count_subtype(input_src, n_subtypes=5):
    """Function to count the number of patches for each subtype

    Parameters
    ----------
    input_src : string or list
        When type of `input_src` is string, it means the input comes from *.txt file, usually it is a file contains patch ids
        When type of `input_src` is a list, it means the input is a list contains patch ids

    n_subtypes : int
        Number of subtypes (or classes)

    Returns
    -------
    count_per_subtype : numpy array
        Number of patches per subtypes

    """
    if isinstance(input_src, str):
        contents = read_data_ids(input_src)
    elif isinstance(input_src, list):
        contents = input_src
    elif isinstance(input_src, np.ndarray):
        contents = input_src
    else:
        raise NotImplementedError(
            'Data type of input_src needs to be str or list.' + type(input_src).__name__ + ' is currently not supported. Consider submitting a Pull Request to support this feature.')

    count_per_subtype = np.zeros(n_subtypes)

    for patch_id in contents:
        cur_label = get_label_by_patch_id(
            patch_id)
        count_per_subtype[cur_label] = count_per_subtype[cur_label] + 1.
    return count_per_subtype


def get_label_by_patch_id(patch_id):
    """Function to obtain label from patch id

    Parameters
    ----------
    patch_id : string
        For non-multiscale patch, patch id has format: /subtype/slide_id/patch_location
        For multiscale patch, patch id has format: /subtype/slide_id/magnification/patch_location

    Returns
    -------
    label: int
        Integer label from SubtypeEnum

    """
    label_idx = -3
    label = patch_id.split('/')[label_idx]
    label = SubtypeEnum[label.upper()]
    return label.value


def get_slide_by_patch_id(patch_id):
    """Function to obtain slide id from patch id

    Parameters
    ----------
    patch_id : string
        For non-multiscale patch, patch id has format: /subtype/slide_id/patch_location
        For multiscale patch, patch id has format: /subtype/slide_id/magnification/patch_location

    Returns
    -------
    slide_id : string
        Slide id extracted from `patch_id`

    """
    slide_idx = -2
    slide_id = patch_id.split('/')[slide_idx]
    return slide_id


def read_data_ids(data_id_path):
    """Function to read data ids (i.e., any *.txt contains row based information)

    Parameters
    ----------
    data_id_path : string
        Absoluate path to the *.txt contains data ids

    Returns
    -------
    data_ids : list
        List conntains data ids

    """
    with open(data_id_path) as f:
        data_ids = f.readlines()
        data_ids = [x.strip() for x in data_ids]
    return data_ids


def get_annotation_polygons(annotation_file_path):
    """Function to extract annotations (label type and polygen vertices) from annotation file

    Parameters
    ----------
    annotation_file_path : string
        Absoluate path to the *.txt contains annotation

    Returns
    -------
    annotations : list
        List of tuples (type, [polygon vertices])

    """
    annotations = []
    with open(annotation_file_path, 'r') as f:
        line = f.readline()
        while line != '' and 'Point' in line:
            label = line.split('[')[0].strip()
            xy = [float(s) for s in re.findall(r'-?\d+\.?\d*', line)]
            polygon = [[x, y] for x, y in zip(xy[::2], xy[1::2])]
            polypath = pltPath.Path(polygon)
            annotations.append((label, polypath))
            line = f.readline()
    return annotations


def set_gpus(n_gpus, verbose=False):
    """Function to set the exposed GPUs

    Parameters
    ----------
    n_gpus : int
        Number of GPUs

    Returns
    -------
    None

    """
    selected_gpu = []
    gpu_free_mem = {}

    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        mem_usage = nvmlDeviceGetMemoryInfo(handle)
        gpu_free_mem[i] = mem_usage.free
        if verbose:
            print("GPU: {} \t Free Memory: {}".format(i, mem_usage.free))

    res = sorted(gpu_free_mem.items(), key=lambda x: x[1], reverse=True)
    res = res[:n_gpus]
    selected_gpu = [r[0] for r in res]

    print("Using GPU {}".format(','.join([str(s) for s in selected_gpu])))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(
        [str(s) for s in selected_gpu])


def create_patch_ids_by_slide(patch_ids):
    """Function to create patch ids sorted by slide ids

    Parameters
    ----------
    patch_ids : list
        List of patch ids

    Returns
    -------
    slide_patches : list
        List of tuples (slide_id: [patch_ids])

    n_slides : int
        Number of slides

    """
    slide_patches = {}
    for patch_id in patch_ids:
        patch_id = create_patch_id(patch_id)
        slide_id = get_slide_by_patch_id(
            patch_ids)
        if slide_id not in slide_patches:
            slide_patches[slide_id] = []
        slide_patches[slide_id] += [patch_id]
    return list(slide_patches.items()), len(slide_patches)


def create_patch_ids_by_patient(patch_ids):
    """Function to create patch ids sorted by patient ids

    Parameters
    ----------
    patch_ids : list
        List of patch ids

    Returns
    -------
    patient_patches : list
        List of tuples (patient_id: [patch_ids])

    n_patients : int
        Number of slides

    """
    patient_patches = {}
    for patch_id in patch_ids:
        patch_id = create_patch_id(patch_id)
        slide_id = get_slide_by_patch_id(
            patch_ids)
        match = re.search(PATIENT_REGEX, slide_id)
        if match:
            patient_id = match.group(1)
            if patient_id not in patient_patches:
                patient_patches[patient_id] = {}
            patient_patches[patient_id] += [patch_id]
    return list(patient_patches.items()), len(patient_patches)


def create_subtype_slide_patch_dict(patch_ids):
    """Function to patch ids sorted by {subtype: {slide_id: [patch_id]}

    Parameters
    ----------
    patch_ids : list
        List of patch ids

    Returns
    -------
    subtype_slide_patch : dict
        {subtype: {slide_id: [patch_id]}

    """
    subtype_slide_patch = {}
    for patch_id in patch_ids:
        patch_id = create_patch_id(patch_id)
        patch_subtype = SubtypeEnum(get_label_by_patch_id(
            patch_id)).name
        if patch_subtype not in subtype_slide_patch:
            subtype_slide_patch[patch_subtype] = {}
        slide_id = get_slide_by_patch_id(patch_id)
        if slide_id not in subtype_slide_patch[patch_subtype]:
            subtype_slide_patch[patch_subtype][slide_id] = []
        subtype_slide_patch[patch_subtype][slide_id] += [patch_id]
    return subtype_slide_patch


def create_subtype_patient_slide_patch_dict(patch_ids):
    """Function to patch ids sorted by {subtype: {patient: {slide_id: [patch_id]}}

    Parameters
    ----------
    patch_ids : list
        List of patch ids

    Returns
    -------
    subtype_patient_slide_patch : dict
        {subtype: {patient: {slide_id: [patch_id]}}

    """
    subtype_patient_slide_patch = {}
    for patch_id in patch_ids:
        patch_id = create_patch_id(patch_id)
        patch_subtype = SubtypeEnum(get_label_by_patch_id(
            patch_id)).name
        if patch_subtype not in subtype_patient_slide_patch:
            subtype_patient_slide_patch[patch_subtype] = {}
        slide_id = get_slide_by_patch_id(patch_id)
        match = re.search(PATIENT_REGEX, slide_id)
        if match:
            patient_id = match.group(1)
            if patient_id not in subtype_patient_slide_patch[patch_subtype]:
                subtype_patient_slide_patch[patch_subtype][patient_id] = {}
            if slide_id not in subtype_patient_slide_patch[patch_subtype][patient_id]:
                subtype_patient_slide_patch[patch_subtype][patient_id][slide_id] = [
                ]
            subtype_patient_slide_patch[patch_subtype][patient_id][slide_id] += [patch_id]
        else:
            raise NotImplementedError(
                '{} is not detected by utils.PATIENT_REGEX'.format(slide_id))
    return subtype_patient_slide_patch


def export_h5_ids(h5_path, export_path):
    """Function to export the patch ids in a h5 file

    Parameters
    ----------
    h5_path : string
        Absoluate path to an h5 file

    export_path : string
        Absoluate path to a text file contains patch id by:
            subtype/slide_id/patch_id or subtype/slide_id/scale/patch_id

    Returns
    -------
    None

    """
    with h5py.File(h5_path, 'r') as h5f_image, open(export_path, 'w') as export_f:
        for subtype_key, subtype_group in h5f_image.items():
            for slide_key, slide_group in h5f_image[subtype_key].items():
                for patch_key, patch_group in h5f_image[subtype_key][slide_key].items():
                    patch_id = '/'.join([subtype_key, slide_key, patch_key])
                    export_f.write('{}\n'.format(patch_id))


def prob_gap(probs):
    """Function to compute the gap between the largest probability and the second probability

    Parameters
    ----------
    probs : numpy array (float)
        Numpy array contains the probabilities for each subtype

    Returns
    -------
    gap : float
        Gap between the largest probability and the second probability

    """
    if len(probs.shape) == 1:
        probs = probs.reshape(1, *probs.shape)
    sorted_probs = np.sort(probs, axis=1)
    largest_prob = sorted_probs[:, -1]
    sec_largest_prob = sorted_probs[:, -2]
    return largest_prob - sec_largest_prob


def compute_acc_and_kappa(labels, preds):
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    print('Acc: {} Kappa: {}'.format(acc, kappa))


def filtered_slide(slide_count_list):
    total = 0
    for slide_count in slide_count_list:
        total += len(slide_count)
    print(total)


def parse_distribution_file(file_path, n_subtypes=5, exclude_mode='gap', threshold=0.99):
    """Function to parse the distribution file (i.e., the probability for each subtype)

    Parameters
    ----------
    file_path : string
        Absoluate path to the distribution *.txt file

    n_subtypes (optional) : 5
        Number of subtype

    exclude_mode (optional) : gap
        Exclude criterion for the patch prediction

    threshold (optional): float
        Exclude criterion for the patch prediction

    Returns
    -------
    cls_cnt_mat : numpy array (int):
        Each row represents a slide, each column represents the number of patch for the subtype

    label_mat : numpy array (int):
        Slide-level label

    probs : numpy array (int):
        Each row represents a slide, each column represents the probability for the subtype

    gt_labels : numpy array (int):
        Patch-level label

    pred_labels : numpy array (int):
        Predicted Patch-level label

    """
    with open(file_path) as f:
        data_str = f.read()
        data_str = data_str.strip()
    patch_infos = data_str.split('---\n')
    # pre-allocate enough memory to store data
    probs = np.zeros((len(patch_infos), n_subtypes))
    pred_labels = np.zeros((len(patch_infos)))
    gt_labels = np.zeros((len(patch_infos)))
    slide_ids = []
    # store class count matrix
    cls_cnt_dict = {}
    label_dict = {}
    # index of each info
    patch_info_idx = 0
    distribution_idx = 1
    for idx, patch_info in enumerate(patch_infos):
        patch_info = patch_info.split('\n')
        # read distribution
        prob = np.fromstring(
            patch_info[distribution_idx][2:-2], count=n_subtypes, sep=' ')
        if exclude_mode == 'gap':
            include = prob_gap(prob) > threshold
        elif exclude_mode == 'var':
            include = np.var(prob) > threshold
        elif exclude_mode == 'none':
            include = True
        else:
            raise NotImplementedError(
                '{} for exclude mode is currently not supported. Consider submitting a Pull Request to support this feature.'.format(exclude_mode))
        if include:
            probs[idx] = prob
            # obtain ground truth label from data path
            gt_label = np.array(get_label_by_patch_id(
                patch_info[patch_info_idx][2:-3]))
            pred_label = np.argmax(prob)
            pred_labels[idx] = pred_label
            gt_labels[idx] = gt_label
            slide_id = get_slide_by_patch_id(patch_info[patch_info_idx][2:-3])
            if slide_id not in cls_cnt_dict:
                cls_cnt_dict[slide_id] = np.zeros(n_subtypes)
                label_dict[slide_id] = gt_label
            cls_cnt_dict[slide_id][pred_label] += 1
    # pre-allocate enough memory to store data
    cls_cnt_mat = np.zeros((len(cls_cnt_dict), n_subtypes))
    label_mat = np.zeros(len(cls_cnt_dict))
    # build the class count matrix
    for idx, (slide_id, cls_cnt_arr) in enumerate(cls_cnt_dict.items()):
        cls_cnt_mat[idx] = cls_cnt_arr
        label_mat[idx] = label_dict[slide_id]
        slide_ids += [slide_id]
    # exclude the rows contain all zero
    empty_row_idx = ~np.all(probs == 0, axis=1)
    return cls_cnt_mat, label_mat, probs[empty_row_idx], gt_labels.astype(np.int8)[empty_row_idx], pred_labels.astype(np.int8)[empty_row_idx], slide_ids


def check_duplicate_data_ids(data_ids):
    """Function to check duplicates of data ids

    Parameters
    ----------
    data_ids : list
        List of data ids

    Returns
    -------
    duplicate_ids : list
        List of data ids that are duplicated with number of duplicates

    """
    duplicate_ids = []
    for patch_id, count in collections.Counter(data_ids).items():
        if count > 1:
            duplicate_ids += [(patch_id, count)]
            print(patch_id, count)
    return duplicate_ids


def extract_patient_ids(slide_ids):
    patient_ids = set()
    for slide_id in slide_ids:
        match = re.search(PATIENT_REGEX, slide_id)
        if match:
            patient_id = match.group(1)
            if patient_id.isdigit():
                patient_ids.add(patient_id)
            else:
                raise NotImplementedError(
                    '{} is not a patient id since it contains non-digit chars.'.format(slide_id))
        else:
            raise NotImplementedError(
                '{} is not detected by utils.PATIENT_REGEX'.format(slide_id))
    return patient_ids


def classification_splits_summary(ids_dir, prefix):
    def _print_summary(counts, prefix):
        percentages = np.asarray(counts)
        percentages = counts / counts.sum() * 100
        print('{} & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {:.2f}\% & {} \\'.format(
            prefix, percentages[0], percentages[1], percentages[2], percentages[3], percentages[4], int(counts.sum())) + '\\')

    counts = count_subtype(os.path.join(
        ids_dir, prefix + '_train_ids.txt'))
    _print_summary(counts, 'Training Set')
    counts = count_subtype(os.path.join(
        ids_dir, prefix + '_val_ids.txt'))
    _print_summary(counts, 'Validation Set')
    counts = count_subtype(os.path.join(
        ids_dir, prefix + '_test_ids.txt'))
    _print_summary(counts, 'Testing Set')
