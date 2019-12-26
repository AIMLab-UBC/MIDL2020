from models.base_model import BaseModel
from utils.subtype_enum import SubtypeEnum
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import neural_network
from sklearn import ensemble
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import utils.utils as utils
import numpy as np
import torch
import torchvision
import os
import joblib


class CountBasedFusionModel(BaseModel):
    def name(self):
        return self.count_fusion_classifier

    def __init__(self, config):
        super().__init__(config)
        self.count_fusion_classifier = config.count_fusion_classifier
        self.scaler = StandardScaler()
        self.subtype_name = [s.name for s in SubtypeEnum]

        if self.count_fusion_classifier == 'MLP':
            self.classifier = neural_network.MLPClassifier(hidden_layer_sizes=(
                200, 200, 200), activation='relu', solver='adam', max_iter=1000, learning_rate='constant', learning_rate_init=0.0002, batch_size=16, alpha=1e-5)
        elif self.count_fusion_classifier == 'SVC':
            # Majority Vote Acc: 0.8505070749989979 Kappa: 0.7701683081408705
            # SVC Acc: 0.8614960859880089 Kappa: 0.7855853924285369
            # python3 count_fusion.py --count_fusion_classifier SVC --count_exclude_mode gap --count_exclude_threshold 0.99
            self.classifier = svm.SVC(
                kernel='rbf', gamma='auto', shrinking=False)
        elif self.count_fusion_classifier == 'ExtraTree':
            self.classifier = tree.ExtraTreeClassifier(
                criterion='entropy', class_weight='balanced', max_features='log2')
        elif self.count_fusion_classifier == 'RandomForest':
            self.classifier = ensemble.RandomForestClassifier(
                criterion='gini', n_estimators=100, max_features='log2', class_weight='balanced')
        elif self.count_fusion_classifier == 'KNeighbors':
            self.classifier = neighbors.KNeighborsClassifier(
                n_neighbors=6, weights='distance', algorithm='auto')
        elif self.count_fusion_classifier == 'GradientBoosting':
            self.classifier = ensemble.GradientBoostingClassifier(
                learning_rate=0.001, max_depth=None, n_estimators=1000, max_leaf_nodes=4, min_samples_split=5)
        elif self.count_fusion_classifier == 'HistGradientBoosting':
            self.classifier = ensemble.HistGradientBoostingClassifier(
                learning_rate=0.2, loss='categorical_crossentropy', l2_regularization=0.0001)
        elif self.count_fusion_classifier == 'GaussianProcess':
            self.classifier = gaussian_process.GaussianProcessClassifier()
        elif self.count_fusion_classifier == 'BernoulliNB':
            self.classifier = naive_bayes.BernoulliNB()
        else:
            raise NotImplementedError

    def forward(self, cls_cnt_mat, labels):
        preds = self.classifier.predict(cls_cnt_mat)
        acc = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        return preds, acc, kappa

    def optimize_parameters(self, cls_cnt_mat, labels):
        self.classifier = self.classifier.fit(cls_cnt_mat, labels)

    def save(self, model_id):
        state = {
            'model': self.classifier,
            'scaler': self.scaler,
        }
        joblib.dump(state, os.path.join(self.save_dir,
                                        '_'.join([self.name(), model_id]) + '.sav'))

    def preprocess(self, data, is_eval=False):
        data = softmax(data, axis=1)
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
