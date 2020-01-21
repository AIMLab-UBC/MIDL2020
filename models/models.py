from models.base_model import BaseModel
from utils.subtype_enum import SubtypeEnum
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import models.networks as networks
import utils.utils as utils
import numpy as np
import torch
import torchvision
import os
import joblib
import h5py


class CountBasedFusionModel(BaseModel):
    def name(self):
        return self.count_fusion_classifier

    def __init__(self, config):
        super().__init__(config)
        self.count_fusion_classifier = config.count_fusion_classifier
        self.scaler = StandardScaler()
        self.subtype_name = [s.name for s in SubtypeEnum]
        # init slide-level classifier
        if self.count_fusion_classifier == 'RandomForest':
            self.classifier = ensemble.RandomForestClassifier(
                criterion='gini', n_estimators=100, max_features='log2', class_weight='balanced', random_state=3242)
        else:
            raise NotImplementedError

    def forward(self, cls_cnt_mat, labels):
        # forward pass
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

    def preprocess(self, data, is_eval=False):
        # preprocess normalization
        if is_eval:
            data = self.scaler.transform(data)
        else:
            data = self.scaler.fit_transform(data)
        return data


class DeepModel(BaseModel):
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
            self.model = networks.MultiStage(num_classes=self.n_subtypes,
                                             use_pretrained=self.use_pretrained,
                                             progressive_size=int(
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
