from config import get_config, print_usage
import utils.utils as utils
import models.models as models
import numpy as np


def main(config):
    # parse patch-level results from different splits
    cls_cnt_mats, label_mats, patch_labels, patch_preds, patch_probs, slide_ids = utils.parse_patch_level_info(
        './results/stage_2_patch_level/testing_a_distribution.txt',
        './results/stage_2_patch_level/testing_b_distribution.txt',
        './results/stage_2_patch_level/testing_c_distribution.txt',
        './results/stage_2_patch_level/testing_d_distribution.txt',
        './results/stage_2_patch_level/testing_e_distribution.txt',
        './results/stage_2_patch_level/testing_f_distribution.txt',
        exclude_mode=config.count_exclude_mode, exclude_threshold=config.count_exclude_threshold
    )
    # compute the number of slides used
    n_slides = utils.count_n_slides(cls_cnt_mats)
    print('{} WSIs are included'.format(n_slides))
    print('{} patches are included'.format(len(patch_preds)))
    # combination of each split
    combinations = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 4], [0, 1, 4, 5, 2, 3], [
        0, 1, 4, 5, 3, 2], [2, 3, 4, 5, 0, 1], [2, 3, 4, 5, 1, 0]]
    # place holders to store data
    slide_labels = np.array([])
    model_preds_slide_labels = np.array([])
    model_preds_slide_prob = np.array([]).reshape(0, 5)
    # start computing
    split_id = -1
    with open('./results/stage_2_slide_level/six_fold_cross_validation.txt', 'w') as f:
        for combination in combinations:
            split_id += 1
            # build slide-level model traning set
            train_cls_cnt_mat = np.vstack([
                cls_cnt_mats[combination[0]],
                cls_cnt_mats[combination[1]],
                cls_cnt_mats[combination[2]],
                cls_cnt_mats[combination[3]],
                cls_cnt_mats[combination[4]]])
            # build slide-level traning set ground truth
            train_label_mat = np.hstack([
                label_mats[combination[0]],
                label_mats[combination[1]],
                label_mats[combination[2]],
                label_mats[combination[3]],
                label_mats[combination[4]]]).astype(np.int)
            # build slide-level model test set
            test_cls_cnt_mat = cls_cnt_mats[combination[5]]
            # build slide-level model test set ground truth
            test_label_mat = label_mats[combination[5]]
            # obtain test set slide ids
            test_slide_mat = slide_ids[combination[5]]
            # compute labels using majority vote strategy
            cur_majority_vote_labels = np.argmax(test_cls_cnt_mat, axis=1)
            # initialize slide-level model
            model = models.CountBasedFusionModel(config)
            # preprocess training set
            preprocessed_train_cls_cnt_mat = model.preprocess(
                train_cls_cnt_mat)
            # preporcess test set
            preprocessed_test_cls_cnt_mat = model.preprocess(
                test_cls_cnt_mat, is_eval=True)
            # optimize slide-level model weights
            model.optimize_parameters(
                preprocessed_train_cls_cnt_mat, train_label_mat)
            # compute current split model accuracy and kappa
            cur_model_preds, cur_model_probs, _, _, _, _ = model.forward(
                preprocessed_test_cls_cnt_mat, test_label_mat)
            # store model predicted probability and predicted labels
            model_preds_slide_prob = np.vstack(
                [model_preds_slide_prob, cur_model_probs])
            model_preds_slide_labels = np.hstack(
                [model_preds_slide_labels, cur_model_preds])
            for idx, slide_id in enumerate(test_slide_mat):
                f.write('{}\n'.format(test_slide_mat[idx]))
                f.write('{}\n'.format(test_label_mat[idx]))
                f.write('{}\n'.format(
                    str(cur_model_probs[idx]).replace('\n', '')))
                f.write('{}\n'.format(
                    str(train_cls_cnt_mat[idx]).replace('\n', '')))
                f.write('---\n')
            # store the current split slide labels
            slide_labels = np.hstack([slide_labels, test_label_mat])
            # save current model
            model.save(chr(97 + split_id))

    print('------------ Slide-Level 6-Fold Cross-Validation -------------')
    utils.compute_metric(
        slide_labels, model_preds_slide_labels, model_preds_slide_prob, verbose=True)


if __name__ == '__main__':
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
