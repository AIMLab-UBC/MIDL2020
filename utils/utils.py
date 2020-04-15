from utils.subtype_enum import SubtypeEnum
from pynvml import *
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.path as pltPath
import glob
import re
import os
import torch
import h5py

PATIENT_REGEX = re.compile(r"^[A-Z]*-?(\d*).*\(?.*\)?.*$")


def print_patch_level_results(file_path):
    _, _, patch_prob, patch_labels, patch_pred, _ = parse_distribution_file(
        file_path, exclude_mode='none')
    compute_metric(patch_labels, patch_pred, patch_prob, verbose=True)


def parse_patch_level_info(split_a_file_path, split_b_file_path, split_c_file_path, split_d_file_path, split_e_file_path, split_f_file_path, exclude_mode='gap', exclude_threshold=0.8):
    """Function to parse the patch-level distribution files and exclude patches with `gap` and `var` option

    Parameters
    ----------
    split_a_file_path : string
        Absoluate path to the distribution file for split a

    split_b_file_path : string
        Absoluate path to the distribution file for split b

    split_c_file_path : string
        Absoluate path to the distribution file for split c

    split_d_file_path : string
        Absoluate path to the distribution file for split d

    split_e_file_path : string
        Absoluate path to the distribution file for split e

    split_f_file_path : string
        Absoluate path to the distribution file for split f

    exclude_mode : string
        Exclude patch mode. Option: `gap` and `var`
        `gap` is defined as the the difference between the largest prob and the second largest prob
        `var` is defined as the variance of the probility distribution

    exclude_threshold : float
        Excclude threshold value. If below this value, the patches will not be included


    Returns
    -------
    cls_cnt_mats : numpy array
        Each row represents a slide, and each row has five columns corresponding to number of classes

    label_mats : list
        A row contains the label for each slide

    patch_labels : list
        A row contains the label for each patch

    patch_preds : list
        A row conatins the predicted label for each patch

    patch_probs : numpy array
        Each row represents a probability distribution for each patch

    slide_ids : numpy array
        Corresponding to the slide id of each row of cls_cnt_mats
    """
    # parse the patch level results
    cls_cnt_mat_a, label_mat_a, patch_prob_a, patch_labels_a, patch_pred_a, slide_ids_a = parse_distribution_file(
        split_a_file_path, exclude_mode=exclude_mode, threshold=exclude_threshold)
    cls_cnt_mat_b, label_mat_b, patch_prob_b, patch_labels_b, patch_pred_b, slide_ids_b = parse_distribution_file(
        split_b_file_path, exclude_mode=exclude_mode, threshold=exclude_threshold)
    cls_cnt_mat_c, label_mat_c, patch_prob_c, patch_labels_c, patch_pred_c, slide_ids_c = parse_distribution_file(
        split_c_file_path, exclude_mode=exclude_mode, threshold=exclude_threshold)
    cls_cnt_mat_d, label_mat_d, patch_prob_d, patch_labels_d, patch_pred_d, slide_ids_d = parse_distribution_file(
        split_d_file_path, exclude_mode=exclude_mode, threshold=exclude_threshold)
    cls_cnt_mat_e, label_mat_e, patch_prob_e, patch_labels_e, patch_pred_e, slide_ids_e = parse_distribution_file(
        split_e_file_path, exclude_mode=exclude_mode, threshold=exclude_threshold)
    cls_cnt_mat_f, label_mat_f, patch_prob_f, patch_labels_f, patch_pred_f, slide_ids_f = parse_distribution_file(
        split_f_file_path, exclude_mode=exclude_mode, threshold=exclude_threshold)
    # create a list of class count matrix from different splits
    cls_cnt_mats = [cls_cnt_mat_a, cls_cnt_mat_b, cls_cnt_mat_c,
                    cls_cnt_mat_d, cls_cnt_mat_e, cls_cnt_mat_f]
    # create a list of label for each slide from different splits
    label_mats = [label_mat_a, label_mat_b, label_mat_c,
                  label_mat_d, label_mat_e, label_mat_f]
    # concat patch-level information
    patch_labels = patch_labels_a.tolist() + patch_labels_b.tolist() + patch_labels_c.tolist() + \
        patch_labels_d.tolist() + patch_labels_e.tolist() + patch_labels_f.tolist()
    patch_preds = patch_pred_a.tolist() + patch_pred_b.tolist() + patch_pred_c.tolist() + \
        patch_pred_d.tolist() + patch_pred_e.tolist() + patch_pred_f.tolist()
    patch_probs = np.vstack(
        [patch_prob_a, patch_prob_b, patch_prob_c, patch_prob_d, patch_prob_e, patch_prob_f])
    # concat slide ids
    slide_ids = [slide_ids_a] + [slide_ids_b] + [slide_ids_c] + \
        [slide_ids_d] + [slide_ids_e] + [slide_ids_f]
    return cls_cnt_mats, label_mats, patch_labels, patch_preds, patch_probs, slide_ids


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


def get_label_by_patch_id(patch_id, is_multiscale=False):
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
    label_idx = -3 if not is_multiscale else -4
    label = patch_id.split('/')[label_idx]
    label = SubtypeEnum[label.upper()]
    return label.value


def get_slide_by_patch_id(patch_id, is_multiscale=False):
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
    slide_idx = -2 if not is_multiscale else -3
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


def compute_metric(labels, preds, probs=None, verbose=False):
    """Function to compute the various metrics given predicted labels and ground truth labels

    Parameters
    ----------
    labels : numpy array
        A row contains the ground truth labels

    preds: numpy array
        A row contains the predicted labels

    probs: numpy array
        A matrix and each row is the probability for the predicted patches or slides

    verbose : bool
        Print detail of the computed metrics

    Returns
    -------
    overall_acc : float
        Accuracy

    overall_kappa : float
        Cohen's kappa

    overall_f1 : float
        F1 score

    overall_auc : float
        ROC AUC
    """
    overall_acc = accuracy_score(labels, preds)
    overall_kappa = cohen_kappa_score(labels, preds)
    overall_f1 = f1_score(labels, preds, average='macro')
    conf_mat = confusion_matrix(labels, preds).T
    acc_per_subtype = conf_mat.diagonal()/conf_mat.sum(axis=0)
    if not (probs is None):
        overall_auc = roc_auc_score(
            labels, probs, multi_class='ovo', average='macro')
    # disply results
    if verbose:
        print('Acc: {:.2f}\%'.format(overall_acc * 100))
        print('Kappa: {:.4f}'.format(overall_kappa))
        print('F1: {:.4f}'.format(overall_f1))
        if not (probs is None):
            print('AUC ROC: {:.4f}'.format(overall_auc))
        print_per_class_accuracy(
            acc_per_subtype, overall_acc, overall_kappa, overall_auc, overall_f1)

    # return results
    if not (probs is None):
        return overall_acc, overall_kappa, overall_f1, overall_auc
    else:
        return overall_acc, overall_kappa, overall_f1, None


def count_n_slides(slide_count_list):
    """Function to count the number of slides
    """
    total = 0
    for slide_count in slide_count_list:
        total += len(slide_count)
    return total


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
            patch_info[distribution_idx][1:-1], count=n_subtypes, sep=' ')
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
            gt_label = np.asarray(get_label_by_patch_id(
                patch_info[patch_info_idx], is_multiscale=True))
            pred_label = np.argmax(prob)
            pred_labels[idx] = pred_label
            gt_labels[idx] = gt_label
            slide_id = get_slide_by_patch_id(
                patch_info[patch_info_idx], is_multiscale=True)
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
    gt_labels = gt_labels.astype(np.int8)[empty_row_idx]
    pred_labels = pred_labels.astype(np.int8)[empty_row_idx]
    return cls_cnt_mat, label_mat, probs[empty_row_idx], gt_labels, pred_labels, slide_ids


def print_per_class_accuracy(per_class_acc, overall_acc, overall_kappa, overall_auc, overall_f1):
    formal_order = ['HGSC', 'CC', 'EC', 'LGSC', 'MC']
    subtype_dict = {}
    # for subtype in SubtypeEnum:
    #     subtype_dict[subtype.name] = subtype.value
    idx = [0, 1, 2, 3, 4]
    # for subtype in formal_order:
    #     idx += [subtype_dict[subtype]]

    per_class_acc = per_class_acc * 100

    print('|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|{:.2f}%|{:.4f}|{:.4f}|{:.4f}|{:.2f}%|'.format(
        per_class_acc[idx[0]], per_class_acc[idx[1]], per_class_acc[idx[2]], per_class_acc[idx[3]], per_class_acc[idx[4]],  overall_acc * 100, overall_kappa, overall_auc, overall_f1, per_class_acc.mean()))
