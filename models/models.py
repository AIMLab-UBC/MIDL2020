from models.base_model import BaseModel
from utils.subtype_enum import SubtypeEnum
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
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
        if self.count_fusion_classifier == 'RandomForest':
            self.classifier = ensemble.RandomForestClassifier(
                criterion='gini', n_estimators=100, max_features='log2', class_weight='balanced', random_state=3242)
        else:
            raise NotImplementedError

    def forward(self, cls_cnt_mat, labels):
        preds = self.classifier.predict(cls_cnt_mat)
        probs = self.classifier.predict_proba(cls_cnt_mat)
        acc, kappa, f1, auc = utils.compute_metric(labels, preds, probs)
        return preds, probs, acc, kappa, f1, auc

    def optimize_parameters(self, cls_cnt_mat, labels):
        self.classifier = self.classifier.fit(cls_cnt_mat, labels)

    def save(self, model_id):
        state = {
            'model': self.classifier,
            'scaler': self.scaler,
        }
        joblib.dump(state, os.path.join(self.save_dir,
                                        '_'.join([self.name(), model_id]) + '.sav'))

    def load(self, model_id):
        state = joblib.load(model_id)
        self.classifier = state['model']
        self.scaler = state['scaler']

    def preprocess(self, data, is_eval=False):
        if is_eval:
            data = self.scaler.transform(data)
        else:
            data = self.scaler.fit_transform(data)
        return data


class DeepModel(BaseModel):
    def __init__(self, config, is_eval=False):
        super().__init__(config)
        self.eval_images = h5py.File(os.path.join(
            config.dataset_dir, config.preload_image_file_name), 'r')
        self.is_eval = is_eval

        model = getattr(torchvision.models, self.deep_classifier)
        if 'res' in self.deep_classifier:
            model = model(pretrained=self.use_pretrained)
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, self.config.n_subtypes)
        elif 'vgg' in self.deep_classifier:
            model = model(pretrained=self.use_pretrained)
            model.classifier._modules['6'] = torch.nn.Linear(
                4096, self.n_subtypes)
        elif self.deep_classifier == 'alexnet':
            model = model(pretrained=self.use_pretrained)
            model.classifier._modules['6'] = torch.nn.Linear(
                4096, self.n_subtypes)
        elif 'densenet' in self.deep_classifier:
            model = model(num_classes=self.n_subtypes,
                          pretrained=self.use_pretrained)
        elif 'mobilenet' in self.deep_classifier:
            model = model(num_classes=self.n_subtypes,
                          pretrained=self.use_pretrained)
        else:
            raise NotImplementedError
        self.model = model.cuda()

        if not self.is_eval:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.l2_decay)
        elif self.optim == 'Adamax':
            self.optimizer = torch.optim.Adamax(
                self.model.parameters(), lr=self.lr)
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.Adamax(
                self.model.parameters(), lr=self.lr)
        elif self.optim == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

        if self.continue_train:
            self.load_state(config.load_model_id,
                            load_pretrained=self.load_pretrained)

        if self.is_eval:
            self.load_state(config.load_model_id)
            self.model.eval()

    def forward(self, x):
        output = self.model.forward(*x)
        if type(output).__name__ == 'GoogLeNetOutputs':
            # GoogLeNet output has one logit and two auxiliary logit
            logits = output.logits
        elif type(output).__name__ == 'InceptionOutputs':
            # Inception output has one logit and one auxiliary logit
            logits = output.logits
        else:
            logits = output
        probs = torch.softmax(logits, dim=1)
        return logits, probs, output
