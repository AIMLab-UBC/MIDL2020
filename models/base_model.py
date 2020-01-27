from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import utils.utils as utils
import data.preprocess as preprocess
import numpy as np
import torch
import os
import h5py


class BaseModel():
    def name(self):
        n = [self.model_name_prefix, self.deep_classifier, self.deep_model, self.optim, 'lr' +
             str(self.lr), 'bs' + str(self.batch_size), 'e' + str(self.epoch)]
        if self.n_eval_samples != 100:
            n += ['neval'+str(self.n_eval_samples)]
        if self.use_pretrained:
            n += ['pw']
        if self.use_equalized_batch:
            n += ['eb']
        if self.is_multiscale_expert:
            n += ['mscale_exp']
            n += [self.expert_magnification]
        return '_'.join(n).lower()

    def __init__(self, config):
        self.config = config
        self.deep_classifier = config.deep_classifier
        # avoid same hyperparameters setup result in the same name
        self.model_name_prefix = config.model_name_prefix
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.epoch = config.epoch
        self.n_subtypes = config.n_subtypes
        self.save_dir = config.save_dir
        self.n_eval_samples = config.n_eval_samples
        self.use_pretrained = config.use_pretrained
        self.continue_train = config.continue_train
        self.optim = config.optim
        self.use_equalized_batch = config.use_equalized_batch
        self.is_multiscale_expert = config.is_multiscale_expert
        self.expert_magnification = config.expert_magnification
        self.is_multiscale = config.is_multiscale_expert
        # store evaluation data labels
        self.eval_data_labels = []

    def eval(self, eval_data_ids):
        self.model.eval()
        # set the number of evaluation samples
        if self.n_eval_samples == -1:
            # -1 means eval model on all validation set
            cur_eval_ids = eval_data_ids
        else:
            # convert to numpy array for fast indexing
            eval_data_ids = np.asarray(eval_data_ids)
            # start to extract patch label beforehand
            if len(self.eval_data_labels) == 0:
                for eval_data_id in eval_data_ids:
                    self.eval_data_labels += [
                        utils.get_label_by_patch_id(eval_data_id, self.is_multiscale)]
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
                    # pick without replacement to enlarge diversity
                    cur_eval_ids += np.random.choice(
                        eval_data_ids[cur_subtype_idx], per_subtype_samples, replace=False).tolist()
                except ValueError:
                    # if has less samples in a subtype, pick with replacement
                    cur_eval_ids += np.random.choice(
                        eval_data_ids[cur_subtype_idx], per_subtype_samples).tolist()
        # evaluation during training
        pred_labels = []
        gt_labels = []
        # go through all patch ids
        for cur_eval_id in cur_eval_ids:
            # generate patch id in format: subtype/slide_id/patch_downsample_size/patch_location
            patch_id = utils.create_patch_id(
                cur_eval_id, is_multiscale=self.is_multiscale)
            gt_label = utils.get_label_by_patch_id(
                cur_eval_id, is_multiscale=self.is_multiscale)
            # preprocess images
            cur_image = self.eval_images[patch_id]['image_data'][()]
            cur_image = Image.fromarray(cur_image)
            cur_tensor = preprocess.raw(
                cur_image, is_eval=True, apply_color_jitter=False)
            # forward pass without gradient computation
            with torch.no_grad():
                _, prob = self.forward(cur_tensor)
                pred_label = torch.argmax(prob).item()
            # store ground truth and predicted labels
            pred_labels += [pred_label]
            gt_labels += [gt_label]
        # set model back to train model
        self.model.train()
        # report metric
        return accuracy_score(gt_labels, pred_labels)

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

    def optimize_parameters(self, logits, labels):
        self.loss = self.criterion(logits.type(
            torch.float), labels.type(torch.long))
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def forward(self):
        pass
