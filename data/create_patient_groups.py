import numpy as np
import re
import os
import glob
import json
import random
import argparse
import itertools

import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import utils.utils as utils
from utils.subtype_enum import SubtypeEnum

PATIENT_REGEX = utils.PATIENT_REGEX


def generate_groups(n_groups, patch_ids_path, out_path, min_patches, max_patches, seed, n_subtype=5):
    """Function to generate N groups that contain unique patients
    Parameters
    ----------
    n_groups : int
        Number of groups, each group contains patches from unique patients
    patch_ids_path : string
        Absoluate path to a txt file contains all patch ids
    out_path : string
        Absoluate path to store the group json files
    min_patches : int
        Minimum number of patches per each slide
    max_patches : int
        Maximum number of patches per each slide
    seed : int
        Random
    n_subtype: int
        Numbe of subtypes
    Returns
    -------
    None
    """
    groups = {}
    subtype_names = [s.name for s in SubtypeEnum]

    ignored_slides = []

    for group_idx in range(n_groups):
        groups['group_' + str(group_idx + 1)] = []

    if os.path.isfile(patch_ids_path):
        patches = utils.read_data_ids(patch_ids_path)
    else:
        raise NotImplementedError

    subtype_patient_slide_patch = utils.create_subtype_patient_slide_patch_dict(
        patches)

    for subtype, patient_slide_patch in subtype_patient_slide_patch.items():
        for patient, slide_patch in patient_slide_patch.items():
            for slide, patch in slide_patch.items():
                if len(patch) < min_patches or len(patch) > max_patches:
                    ignored_slides += ['/'.join([subtype, patient, slide])]

    for ignored_slide in ignored_slides:
        ignored_slide_subtype, ignored_slide_patient_num, ignored_slide_slide_id = ignored_slide.split(
            '/')
        del subtype_patient_slide_patch[ignored_slide_subtype][ignored_slide_patient_num][ignored_slide_slide_id]

    patient_subtype_dict = dict(
        zip(subtype_names, [[] for s in subtype_names]))

    for subtype, patients in subtype_patient_slide_patch.items():
        patient_subtype_dict[subtype] += list(patients.keys())

    patient_subtype_count = dict(
        zip(subtype_names, [len(patient_subtype_dict[s]) for s in subtype_names]))

    for subtype_name in subtype_names:
        random.seed(seed)
        random.shuffle(patient_subtype_dict[subtype_name])
        step_size = int(
            np.ceil(patient_subtype_count[subtype_name] / n_groups))
        for group_idx, patient_idx in enumerate(range(0, patient_subtype_count[subtype_name], step_size)):
            selected_patients = patient_subtype_dict[subtype_name][patient_idx:patient_idx+step_size]
            for selected_patient in selected_patients:
                for _, selected_patches in subtype_patient_slide_patch[subtype_name][selected_patient].items():
                    groups['group_' + str(group_idx + 1)] += selected_patches

    for group_idx in range(len(groups)):
        random.seed(seed)
        random.shuffle(groups['group_' + str(group_idx + 1)])

    with open(out_path, 'w') as f:
        json.dump(groups, f)

    print('Ignored Slides')
    print(ignored_slides)


def create_val_test_splits(eval_ids):
    """Function to create validation and test splits based on evaluation ids
       This function ensures unique patches in validation and testing sets
    Parameters
    ----------
    eval_ids : list
        List of patch ids in evaludation set
    Returns
    -------
    val_ids : list
        List of patch ids in validation set
    test_ids : list
        List of patch ids in testing set
    """
    subtype_names = [s.name for s in SubtypeEnum]
    subtype_patient_slide_patch = utils.create_subtype_patient_slide_patch_dict(
        eval_ids)
    val_ids = []
    test_ids = []
    for subtype in subtype_names:
        patient_slide_patch = subtype_patient_slide_patch[subtype]
        patients = list(patient_slide_patch.keys())
        patient_idx = len(patients) // 2
        for patient_in_val in patients[:patient_idx]:
            for patch_ids in patient_slide_patch[patient_in_val].values():
                val_ids += patch_ids
        for patient_in_test in patients[patient_idx:]:
            for patch_ids in patient_slide_patch[patient_in_test].values():
                test_ids += patch_ids

    # make sure testing set has more data than validation set
    if len(val_ids) > len(test_ids):
        swap_ids = test_ids[:]
        test_ids = val_ids[:]
        val_ids = swap_ids[:]

    return val_ids, test_ids


def create_train_val_test_splits(json_path, out_dir, n_groups, n_train_groups, seed):
    """Function to create training, validation and testing sets
    Parameters
    ----------
    json_path : string
        Absoluate path to group information json file
    out_dir : string
        Absoluate path to the directory that store the splits
    preifx : string
        For different experiments
    Returns
    -------
    None
    """

    with open(json_path, 'r') as f:
        groups = json.load(f)

    for train_group in list(itertools.combinations(list(range(1, n_groups + 1)), n_train_groups)):
        eval_group = list(set(range(1, n_groups + 1)) - set(train_group))
        train_ids = []
        eval_ids = []

        for train_group_idx in train_group:
            train_ids += groups['group_' + str(train_group_idx)][:]

        for eval_group_idx in eval_group:
            eval_ids += groups['group_' + str(eval_group_idx)][:]

        val_ids, test_ids = create_val_test_splits(eval_ids)

        train_group = [str(t) for t in train_group]
        eval_group = [str(t) for t in eval_group]

        group_name = '_'.join(train_group) + '_train_' + \
            '_'.join(eval_group) + '_eval'

        with open(os.path.join(out_dir, group_name + '_train_ids.txt'), 'w') as f:
            random.seed(seed)
            random.shuffle(train_ids)
            for train_id in train_ids:
                f.write('{}\n'.format(train_id))
        latex_formatter(utils.count_subtype(
            os.path.join(out_dir, group_name + '_train_ids.txt')), group_name + '_train_ids')

        with open(os.path.join(out_dir, group_name + '_eval_0_ids.txt'), 'w') as f:
            random.seed(seed)
            random.shuffle(val_ids)
            for val_id in val_ids:
                f.write('{}\n'.format(val_id))
        latex_formatter(utils.count_subtype(
            os.path.join(out_dir, group_name + '_eval_0_ids.txt')), group_name + '_eval_0_ids')

        with open(os.path.join(out_dir, group_name + '_eval_1_ids.txt'), 'w') as f:
            random.seed(seed)
            random.shuffle(test_ids)
            for test_id in test_ids:
                f.write('{}\n'.format(test_id))
        latex_formatter(utils.count_subtype(
            os.path.join(out_dir, group_name + '_eval_1_ids.txt')), group_name + '_eval_1_ids')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--n_groups", type=int, default=3)
    parser.add_argument("--n_train_groups", type=int, default=2)
    parser.add_argument("--patch_ids_path", type=str,
                        default='/projects/ovcare/classification/ywang/midl_dataset/test_dataset/patch_ids/patch_ids.txt')
    parser.add_argument("--out_path", type=str,
                        default='/projects/ovcare/classification/ywang/midl_dataset/test_dataset/patch_ids/patient_group.json')
    parser.add_argument("--min_patches", type=int, default=10)
    parser.add_argument("--max_patches", type=int, default=1000000)
    parser.add_argument("--split_dir", type=str,
                        default='/projects/ovcare/classification/ywang/midl_dataset/test_dataset/patch_ids/')

    args = parser.parse_args()

    generate_groups(args.n_groups, args.patch_ids_path, args.out_path,
                    args.min_patches, args.max_patches, seed=args.seed)
    create_train_val_test_splits(
        args.out_path, args.split_dir, args.n_groups, args.n_train_groups, seed=args.seed)
