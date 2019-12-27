from config import get_config, print_usage
from utils.subtype_enum import SubtypeEnum
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
#from plots.cm import plot_confusion_matrix_from_data
import utils.utils as utils
import models.models as models
import numpy as np
import os


def train(config):
    cls_cnt_mat_a, label_mat_a, _, patch_labels_a, patch_pred_a, slide_ids_a = utils.parse_distribution_file(
        '/Users/Andy/Desktop/results/testing_a_distribution_pw.txt', exclude_mode=config.count_exclude_mode, threshold=config.count_exclude_threshold)
    cls_cnt_mat_b, label_mat_b, _, patch_labels_b, patch_pred_b, slide_ids_b = utils.parse_distribution_file(
        '/Users/Andy/Desktop/results/testing_b_distribution_pw.txt', exclude_mode=config.count_exclude_mode, threshold=config.count_exclude_threshold)
    cls_cnt_mat_c, label_mat_c, _, patch_labels_c, patch_pred_c, slide_ids_c = utils.parse_distribution_file(
        '/Users/Andy/Desktop/results/testing_c_distribution_pw.txt', exclude_mode=config.count_exclude_mode, threshold=config.count_exclude_threshold)
    cls_cnt_mat_d, label_mat_d, _, patch_labels_d, patch_pred_d, slide_ids_d = utils.parse_distribution_file(
        '/Users/Andy/Desktop/results/testing_d_distribution_pw.txt', exclude_mode=config.count_exclude_mode, threshold=config.count_exclude_threshold)
    cls_cnt_mat_e, label_mat_e, _, patch_labels_e, patch_pred_e, slide_ids_e = utils.parse_distribution_file(
        '/Users/Andy/Desktop/results/testing_e_distribution_pw.txt', exclude_mode=config.count_exclude_mode, threshold=config.count_exclude_threshold)
    cls_cnt_mat_f, label_mat_f, _, patch_labels_f, patch_pred_f, slide_ids_f = utils.parse_distribution_file(
        '/Users/Andy/Desktop/results/testing_f_distribution_pw.txt', exclude_mode=config.count_exclude_mode, threshold=config.count_exclude_threshold)

    cls_cnt_mats = [cls_cnt_mat_a, cls_cnt_mat_b, cls_cnt_mat_c,
                    cls_cnt_mat_d, cls_cnt_mat_e, cls_cnt_mat_f]
    label_mats = [label_mat_a, label_mat_b, label_mat_c,
                  label_mat_d, label_mat_e, label_mat_f]
    print(utils.filtered_slide(cls_cnt_mats))
    patch_labels = patch_labels_a.tolist() + patch_labels_b.tolist() + patch_labels_c.tolist() + \
        patch_labels_d.tolist() + patch_labels_e.tolist() + patch_labels_f.tolist()
    patch_preds = patch_pred_a.tolist() + patch_pred_b.tolist() + patch_pred_c.tolist() + \
        patch_pred_d.tolist() + patch_pred_e.tolist() + patch_pred_f.tolist()
    print(len(patch_preds))
    slide_ids = [slide_ids_a] + [slide_ids_b] + [slide_ids_c] + \
        [slide_ids_d] + [slide_ids_e] + [slide_ids_f]

    wrong_slide = set()

    utils.compute_acc_and_kappa(patch_labels, patch_preds)
    combinations = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4], [0, 1, 4, 5, 2, 3], [
        0, 1, 4, 5, 3, 2], [2, 3, 4, 5, 0, 1], [2, 3, 4, 5, 1, 0]]

    majority_vote_average_acc = 0
    majority_vote_average_kappa = 0
    model_average_acc = 0
    model_average_kappa = 0
    max_model_acc = float('-inf')

    all_labels = np.array([])
    all_preds = np.array([])

    majority_vote_all_labels = np.array([])

    bad_slides = {'VOA-1754A', 'VOA-1179B',
                  'VOA-1931C', 'VOA-1931A', 'VOA-1179A'}
    for combination in combinations:
        train_cls_cnt_mat = np.vstack(
            [cls_cnt_mats[combination[0]], cls_cnt_mats[combination[1]], cls_cnt_mats[combination[2]], cls_cnt_mats[combination[3]], cls_cnt_mats[combination[4]]])
        train_label_mat = np.hstack(
            [label_mats[combination[0]], label_mats[combination[1]], label_mats[combination[2]], label_mats[combination[3]], label_mats[combination[4]]]).astype(np.int)

        val_cls_cnt_mat = cls_cnt_mats[combination[5]]
        val_label_mat = label_mats[combination[5]]
        val_slide_mat = slide_ids[combination[5]]

        test_cls_cnt_mat = cls_cnt_mats[combination[5]]
        test_label_mat = label_mats[combination[5]]

        majority_vote_labels = np.argmax(val_cls_cnt_mat, axis=1)
        majority_vote_all_labels = np.hstack(
            [majority_vote_all_labels, majority_vote_labels]).copy()
        majority_vote_acc = accuracy_score(val_label_mat, majority_vote_labels)
        majority_vote_kappa = cohen_kappa_score(
            majority_vote_labels, val_label_mat)

        print('Combination: {}'.format(str(combination)))
        print('Majority Vote Acc {} and Kappa {}'.format(
            str(majority_vote_acc), str(majority_vote_kappa)))
        majority_vote_average_kappa += majority_vote_kappa
        majority_vote_average_acc += majority_vote_acc

        model = models.CountBasedFusionModel(config)

        train_cls_cnt_mat_standard = model.preprocess(train_cls_cnt_mat)
        val_cls_cnt_mat_standard = model.preprocess(
            val_cls_cnt_mat, is_eval=True)

        model.optimize_parameters(train_cls_cnt_mat_standard, train_label_mat)

        preds, model_acc, model_kappa = model.forward(
            val_cls_cnt_mat_standard, val_label_mat)
        model_average_acc += model_acc
        model_average_kappa += model_kappa

        for idx, pred in enumerate(preds):
            if pred != val_label_mat[idx]:
                v = val_cls_cnt_mat[idx]
                v = [int(x) for x in v]
                # if val_slide_mat[idx] in bad_slides:
                print(v, SubtypeEnum(pred).name, SubtypeEnum(val_label_mat[idx].astype(
                    int)).name, val_cls_cnt_mat_standard[idx], val_slide_mat[idx])
                wrong_slide.add(val_slide_mat[idx])

        all_labels = np.hstack([all_labels, val_label_mat])
        all_preds = np.hstack([all_preds, preds])

        if max_model_acc < model_acc:
            best_model = model

        print('{} Acc: {} Kappa: {}'.format(
            model.name(), str(model_acc), str(model_kappa)))

    print('------ Average Acc and Kappa ------')
    print('Majority Vote Acc: {} Kappa: {}'.format(str(majority_vote_average_acc /
                                                       len(combinations)), str(majority_vote_average_kappa / len(combinations))))
    print('{} Acc: {} Kappa: {}'.format(model.name(), str(model_average_acc /
                                                          len(combinations)), str(model_average_kappa / len(combinations))))
    print('------ Complete Acc and Kappa --------')
    utils.compute_acc_and_kappa(all_labels, all_preds)
    utils.compute_acc_and_kappa(all_labels, majority_vote_all_labels)
    print(wrong_slide)
    # best_model.save('max_val_acc')
    # plot_confusion_matrix_from_data(all_labels, all_preds, columns=model.subtype_name,
    #                                 title=config.count_fusion_classifier + ' Count Based Slide Level CM')


def main(config):
    if config.mode == 'Training':
        train(config)
    elif config.mode == 'Validation':
        raise NotImplementedError
    elif config.mode == 'Testing':
        raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == '__main__':
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
