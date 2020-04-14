from models.base_model import BaseModel
from utils.subtype_enum import SubtypeEnum
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
import models.networks as networks
import utils.utils as utils
import numpy as np
import torch
import os
import joblib
import h5py


class CountBasedFusionModel(BaseModel):
    """Count-based slide-level fusion classifier

    For the patches in each slide, we apply the stage-2 patch-level classifier to obtain
    the probability distribution and take the largest probability as the patch-level label. 
    Afterwards, in each slide, we count the predicted patch-level labels in each class. 
    Therefore, the assumed input has shape N * C, where N is the number of slides, and
    C is the number of classes. 
    """

    def name(self):
        return self.count_fusion_classifier

    def __init__(self, config):
        super().__init__(config)
        # obtain slide-level classifier from config
        self.count_fusion_classifier = config.count_fusion_classifier
        # initialize preprocess scaler
        self.scaler = StandardScaler()
        # obtain the string type class name
        self.subtype_name = [s.name for s in SubtypeEnum]
        # initialize slide-level classifier
        if self.count_fusion_classifier == 'RandomForest':
            self.classifier = ensemble.RandomForestClassifier(
                criterion='gini', n_estimators=100, max_features='log2', class_weight='balanced', random_state=3242)
        else:
            raise NotImplementedError

    def preprocess(self, data, is_eval=False):
        # standardize the input data to zero mean and unit variance
        if is_eval:
            data = self.scaler.transform(data)
        else:
            data = self.scaler.fit_transform(data)
        return data

    def forward(self, cls_cnt_mat, labels):
        # predict
        preds = self.classifier.predict(cls_cnt_mat)
        probs = self.classifier.predict_proba(cls_cnt_mat)
        # compute metric
        acc, kappa, f1, auc = utils.compute_metric(labels, preds, probs)
        return preds, probs, acc, kappa, f1, auc

    def optimize_parameters(self, cls_cnt_mat, labels):
        self.classifier = self.classifier.fit(cls_cnt_mat, labels)

    def save(self, model_id):
        # save model parameters and preprocess scaler
        state = {
            'model': self.classifier,
            'scaler': self.scaler,
        }
        joblib.dump(state, os.path.join(self.save_dir,
                                        '_'.join([self.name(), model_id]) + '.sav'))

    def load(self, model_id):
        # load model parameters and preprocess scaler
        state = joblib.load(model_id)
        self.classifier = state['model']
        self.scaler = state['scaler']


class DeepModel(BaseModel):
    """Deep learning-based patch-level classifier

    We apply the baseline and the two-stage classifier inside this class. 
    """

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

    def __init__(self, config, is_eval=False):
        super().__init__(config)
        # load images for validation during training
        self.eval_images = h5py.File(os.path.join(
            config.dataset_dir, config.preload_image_file_name), 'r')
        # flag indicates model.train() or model.eval()
        self.is_eval = is_eval
        # init models
        if self.deep_classifier.lower() == 'baseline':
            # init baseline models, i.e., vgg19_bn
            self.model = networks.Baseline(
                num_classes=self.n_subtypes, use_pretrained=self.use_pretrained)
        elif self.deep_classifier.lower() == 'multi_stage':
            # init multi-stage model
            load_name = self.name()
            if self.expert_magnification == '512':
                prior_scale = str(int(self.expert_magnification) // 2)
                prior_batch_size = self.batch_size * 2
                load_name = load_name.replace('mscale_exp_' + self.expert_magnification, 'mscale_exp_' +
                                              prior_scale).replace('bs' + str(self.batch_size), 'bs' + str(prior_batch_size))
            # create multi-stage model
            self.model = networks.TwoStageCNN(num_classes=self.n_subtypes,
                                              use_pretrained=self.use_pretrained,
                                              patch_size=int(
                                                  self.expert_magnification),
                                              weights_save_path=os.path.join(self.save_dir, load_name + '_' + self.load_model_id + '.pth'))
        else:
            raise NotImplementedError
        # use cuda
        self.model.cuda()
        # init cross-entropy loss and optimizer
        if not self.is_eval:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if self.optim == 'Adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.lr, weight_decay=self.l2_decay)
            else:
                raise NotImplementedError
        # continue train models
        if self.continue_train:
            self.load_state(config.load_model_id,
                            load_pretrained=self.load_pretrained)
            self.model.train()
        # set to eval mode
        if self.is_eval:
            self.load_state(config.load_model_id)
            self.model.eval()

    def forward(self, x):
        # forward pass
        logits = self.model.forward(x)
        # compute probility distribution
        probs = torch.softmax(logits, dim=1)
        return logits, probs

    def optimize_parameters(self, logits, labels):
        self.loss = self.criterion(logits.type(
            torch.float), labels.type(torch.long))
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_current_errors(self):
        return self.loss.item()

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
