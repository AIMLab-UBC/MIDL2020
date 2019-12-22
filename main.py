#!/usr/bin/env python3

from config import get_config, print_usage
from tqdm import tqdm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score
import models.models as models
import numpy as np
import utils.utils as utils
import data.patch_dataset as patch_dataset
import data.postprocess as postprocess
import torch
import os


def evaluate(config):
    assert(config.eval_batch_size == 1)

    patches = patch_dataset.SubtypePatchDataset(
        config, apply_color_jitter=False)
    print('{} mode and there are {} patches...'.format(
        config.mode, str(len(patches))))

    data_loader = torch.utils.data.DataLoader(
        patches,
        batch_size=config.eval_batch_size)

    model = models.DeepModel(config, is_eval=True)

    prefix = config.mode + ' Epoch: '
    n_correct = np.zeros(config.n_subtypes).astype(np.float64)
    n_idx = np.zeros(config.n_subtypes).astype(np.float64)
    conf_mat = np.zeros((config.n_subtypes, config.n_subtypes))

    with open(os.path.join(config.dataset_dir, config.mode + '_' + config.testing_output_file_name), 'w') as f:
        list_pred_labels = []
        list_gt_labels = []
        for data in tqdm(data_loader, desc=prefix):
            cur_data, cur_label, orig_patch = data
            with torch.no_grad():
                _, pred_prob, _ = model.forward(cur_data)
                pred_label = torch.argmax(pred_prob).item()
                n_idx[cur_label] = n_idx[cur_label] + 1.
                list_gt_labels += [cur_label.item()]
                list_pred_labels += [pred_label]
                conf_mat[pred_label][cur_label] = conf_mat[pred_label][cur_label] + 1.
                if pred_label == cur_label:
                    n_correct[pred_label] = n_correct[pred_label] + 1.
                # write to into distribution output file
                pred_prob = pred_prob.cpu().numpy()
                f.write('{}\n'.format(cur_slide_id))
                f.write('{}\n'.format(str(pred_prob).replace('\n', '')))
                f.write('{}\n'.format(patch_id[0]))
                f.write('---\n')

    overall_acc = n_correct.sum()/n_idx.sum()
    overall_kappa = cohen_kappa_score(list_pred_labels, list_gt_labels)
    acc_per_subype = n_correct / n_idx
    print("Overall {} Acc: {}".format(config.mode, str(overall_acc)))
    print("Overall {} Kappa: {}".format(config.mode, str(overall_kappa)))
    print('{} Acc Per Subtype: {}'.format(config.mode, str(acc_per_subype)))
    print('Confusion Matrix')
    print(repr(conf_mat))
    print('--- Latex Table Format ---')
    utils.latex_table_formatter(acc_per_subype, overall_acc, overall_kappa)


def train(config):
    patches = patch_dataset.SubtypePatchDataset(config)

    print('{} mode and there are {} patches...'.format(
        config.mode, str(len(patches))))

    data_loader = torch.utils.data.DataLoader(
        patches,
        batch_size=config.batch_size)

    model = models.DeepModel(config, is_eval=False)

    writer = SummaryWriter(log_dir=os.path.join(config.log_dir, model.name()))
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    iter_idx = -1
    max_val_acc = float('-inf')
    intv_loss = 0
    for epoch in range(config.epoch):
        prefix = 'Training Epoch {:3d}: '.format(epoch)
        for data in tqdm(data_loader, desc=prefix):
            iter_idx += 1

            train_data, train_labels, orig_patch = data

            pred_labels_logits, pred_labels_probs, output = model.forward(
                train_data)
            model.optimize_parameters(pred_labels_logits, train_labels, output)

            intv_loss += model.get_current_errors()

            if iter_idx % config.rep_intv == 0:
                writer.add_scalar('Training CrossEntropyLoss', intv_loss / config.rep_intv,
                                  global_step=iter_idx)
                intv_loss = 0

                val_acc = model.eval(eval_data_ids=patches.eval_data_ids)

                if max_val_acc < val_acc:
                    max_val_acc = val_acc
                    model.save_state(model_id='max_val_acc')

                writer.add_scalar('Validation Accuracy',
                                  val_acc, global_step=iter_idx)

                if config.log_patches:
                    concat_patches = postprocess.hori_concat_img(
                        orig_patch.numpy().transpose(0, 3, 1, 2))
                    writer.add_images('Patches', concat_patches,
                                      global_step=iter_idx, dataformats='HWC')


def main(config):
    utils.set_gpus(n_gpus=1)
    if config.mode == 'Training':
        train(config)
    elif config.mode == 'Validation':
        evaluate(config)
    elif config.mode == 'Testing':
        evaluate(config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
