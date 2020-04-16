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
        return 'base_model'

    def __init__(self, config):
        self.config = config
        self.deep_classifier = config.deep_classifier
        self.deep_model = config.deep_model
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
        self.load_model_id = config.load_model_id
        # store evaluation data labels
        self.eval_data_labels = []

    def eval(self, eval_data_ids):
        pass

    def load_state(self, model_id, load_pretrained=False):
        pass

    def save_state(self, model_id):
        pass

    def get_current_errors(self):
        pass

    def optimize_parameters(self, logits, labels):
        pass

    def forward(self):
        pass
