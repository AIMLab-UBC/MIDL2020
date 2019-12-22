from PIL import Image
from sklearn.metrics import cohen_kappa_score
import utils.utils as utils
import data.preprocess as preprocess
import numpy as np
import os
import h5py
import torch


class BaseModel():
    def name(self):
        n = [self.model_name_prefix, self.deep_classifier, self.deep_model, self.optim, 'lr' +
             str(self.lr), 'bs' + str(self.batch_size), 'e' + str(self.epoch)]
        if self.n_eval_samples != 500:
            n += ['neval'+str(self.n_eval_samples)]
        if self.use_pretrained:
            n += ['pw']
        if self.use_equalized_batch:
            n += ['eb']
        if self.l2_decay != 0:
            n += ['l2' + str(self.l2_decay)]
        if self.use_kappa_select_model:
            n += ['kappa']
        return '_'.join(n).lower()

    def __init__(self, config):
        self.config = config
        self.deep_classifier = config.deep_classifier
        self.count_fusion_classifier = config.count_fusion_classifier
        self.model_name_prefix = config.model_name_prefix
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.patch_size = config.patch_size
        self.n_subtypes = config.n_subtypes
        self.save_dir = config.save_dir
        self.n_eval_samples = config.n_eval_samples
        self.use_pretrained = config.use_pretrained
        self.l2_decay = config.l2_decay
        self.continue_train = config.continue_train
        self.log_patches = config.log_patches
        self.optim = config.optim
        self.use_equalized_batch = config.use_equalized_batch
        self.use_kappa_select_model = config.use_kappa_select_model
        self.load_model_id = config.load_model_id
        self.eval_images = h5py.File(os.path.join(
            config.dataset_dir, config.preload_image_file_name), 'r')
        self.eval_data_labels = []

    def eval(self, eval_data_ids):
        # set model to eval mode to disable BatchNorm and DropOut
        self.model.eval()
        # set the number of evaluation samples
        if self.n_eval_samples == -1:
            # -1 means eval model on all validation set
            cur_eval_ids = eval_data_ids[:]
        else:
            # convert to numpy array for fast indexing
            eval_data_ids = np.asarray(eval_data_ids)
            # start to extract patch label beforehand
            if len(self.eval_data_labels) == 0:
                for eval_data_id in eval_data_ids:
                    self.eval_data_labels += [
                        utils.get_label_by_patch_id(eval_data_id)]
                # convert to numpy array for fast indexing
                self.eval_data_labels = np.asarray(self.eval_data_labels)
            # store the evaluation patch ids
            cur_eval_ids = []
            # compute the number of patches per subtype for evaluation
            per_subtype_samples = self.n_eval_samples // self.n_subtypes
            # select patch ids by subtype
            for subtype in range(self.n_subtypes):
                # numpy advanced indexing
                cur_subtype_idx = self.eval_data_labels == subtype
                # randomly pick patch ids
                try:
                    # pick without replacement
                    cur_eval_ids += np.random.choice(
                        eval_data_ids[cur_subtype_idx], per_subtype_samples, replace=False).tolist()
                except ValueError:
                    # if has less samples in a subtype, pick with replacement
                    cur_eval_ids += np.random.choice(
                        eval_data_ids[cur_subtype_idx], per_subtype_samples).tolist()
        # evaluation during training
        n_correct = 0
        if self.use_kappa_select_model:
            pred_list = []
            gt_list = []
        # now we go through all eval data ids
        for cur_eval_id in cur_eval_ids:
            # generate patch id in format: subtype/slide_id/patch_location
            patch_id = utils.create_patch_id(cur_eval_id)
            gt_label = utils.get_label_by_patch_id(cur_eval_id)
            # obtain image from h5 file
            cur_image = self.eval_images[patch_id]['image_data'][()]
            # convert to PIL image
            cur_image = Image.fromarray(cur_image)
            # preprocess (disable color jitter in validation and testing)
            cur_tensor = preprocess.raw(
                cur_image, is_eval=True, apply_color_jitter=False)
            # disable gradient computation
            with torch.no_grad():
                _, prob, _ = self.forward((cur_tensor, ))
                pred_label = torch.argmax(prob).item()
            # if use kappa to select weights, store ground truth and prediction to a list
            if self.use_kappa_select_model:
                pred_list += [pred_label]
                gt_list += [gt_label]
            else:
                if pred_label == gt_label:
                    n_correct += 1
        # set model back to train mode
        self.model.train()
        if self.use_kappa_select_model:
            return cohen_kappa_score(pred_list, gt_list)
        else:
            return n_correct / len(cur_eval_ids)

    def load_state(self, model_id, load_pretrained=False):
        if load_pretrained:
            filename = model_id + '.pth'
        else:
            filename = '{}_{}.pth'.format(self.name(), str(model_id))
        save_path = os.path.join(self.save_dir, filename)

        if torch.cuda.is_available():
            state = torch.load(save_path)
        else:
            state = torch.load(save_path, map_location='cpu')

        try:
            self.model.load_state_dict(state['state_dict'])
        except RuntimeError:
            pretrained_dict = state['state_dict']
            model_dict = self.model.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # load the new state dict
            self.model.load_state_dict(pretrained_dict)

        self.optimizer.load_state_dict(state['optimizer'])

        model_id = state['iter_idx']
        return model_id

    def save_state(self, model_id):
        filename = '{}_{}.pth'.format(self.name(), str(model_id))
        save_path = os.path.join(self.save_dir, filename)
        state = {
            'iter_idx': model_id,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, save_path)

    def get_current_errors(self):
        return self.loss.item()

    def optimize_parameters(self, logits, labels, output=None):
        if type(output).__name__ == 'GoogLeNetOutputs':
            self.loss = self.criterion(logits.type(torch.float), labels.type(torch.long)) + 0.4 * (self.criterion(output.aux_logits1.type(
                torch.float), labels.type(torch.long)) + self.criterion(output.aux_logits2.type(torch.float), labels.type(torch.long)))
        elif type(output).__name__ == 'InceptionOutputs':
            self.loss = self.criterion(logits.type(torch.float), labels.type(
                torch.long)) + 0.4 * self.criterion(output.aux_logits.type(torch.float), labels.type(torch.long))
        else:
            self.loss = self.criterion(logits.type(
                torch.float), labels.type(torch.long))
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def forward(self):
        pass
