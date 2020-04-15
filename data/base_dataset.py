from torch.utils.data import Dataset
import numpy as np
import utils.utils as utils
import h5py
import os


class BaseDataset(Dataset):
    def name(self):
        return 'base_dataset'

    def __init__(self, config):
        self.config = config
        self.mode = config.mode
        self.is_eval = config.mode != 'Training'
        self.n_eval_samples = config.n_eval_samples
        self.preload_images = h5py.File(os.path.join(
            config.dataset_dir, config.preload_image_file_name), 'r')
        self.n_subtypes = config.n_subtypes
        self.use_equalized_batch = config.use_equalized_batch
        self.is_multiscale_expert = config.is_multiscale_expert
        self.expert_magnification = config.expert_magnification
        self.is_multiscale = config.is_multiscale_expert

        if config.mode == 'Testing':
            # Testing mode is used for evaluating model in the test dataset
            self.cur_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.test_ids_file_name))
        elif config.mode == 'Validation':
            # Validation mode is used for evaluating model in the validation dataset
            self.cur_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.val_ids_file_name))
        else:
            # Training mode is used for training model in the training dataset
            self.cur_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.train_ids_file_name))
            # also pass the validation set to evaluate model during training
            self.eval_data_ids = utils.read_data_ids(
                os.path.join(config.dataset_dir, config.val_ids_file_name))
        # modify patch ids for h5
        if self.is_multiscale_expert:
            # replace data id to the corresponding expert magnification
            for idx, cur_data_id in enumerate(self.cur_data_ids):
                data_info = cur_data_id.split('/')
                data_info[-2] = str(self.expert_magnification)
                self.cur_data_ids[idx] = '/'.join(data_info)
            if self.config.mode == 'Training':
                for idx, cur_data_id in enumerate(self.eval_data_ids):
                    data_info = cur_data_id.split('/')
                    data_info[-2] = str(self.expert_magnification)
                    self.eval_data_ids[idx] = '/'.join(data_info)
        # employ balanced batch during training
        if self.use_equalized_batch:
            # extract patch label beforehand
            self.label_list = []
            for cur_data_id in self.cur_data_ids:
                cur_label = utils.get_label_by_patch_id(cur_data_id)
                self.label_list += [cur_label]
            # for fast numpy indexing
            self.label_list = np.asarray(self.label_list)
            self.cur_data_ids = np.asarray(self.cur_data_ids)

    def __len__(self):
        return len(self.cur_data_ids)

    def __getitem__(self, index):
        pass
