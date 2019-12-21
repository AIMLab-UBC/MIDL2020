from data.base_dataset import BaseDataset
from PIL import Image
import utils.utils as utils
import data.preprocess as preprocess
import numpy as np
import torch


class SubtypePatchDataset(BaseDataset):
    def __init__(self, config, apply_color_jitter=True):
        super().__init__(config)
        self.apply_color_jitter = apply_color_jitter

    def __getitem__(self, index):
        if self.use_equalized_batch and not self.is_eval:
            # compute current patch label
            sample_label = index % self.n_subtypes
            # obatin all the the patch has the same label
            label_idx = self.label_list == sample_label
            # randomly select one patch
            cur_data_id = np.random.choice(self.cur_data_ids[label_idx])
        else:
            cur_data_id = self.cur_data_ids[index]

        # create patch id by stripping the image extension
        # and only obtain `/subtype/slide_id/image_location` format
        # for h5 file to obtain the image data
        patch_id = utils.create_patch_id(cur_data_id)
        cur_label = utils.get_label_by_patch_id(patch_id)
        cur_image = Image.fromarray(cur_image)
        cur_tensor = preprocess.raw(
            cur_image, apply_color_jitter=self.apply_color_jitter)
        cur_label = torch.tensor(cur_label).type(torch.LongTensor).cuda()
        return (cur_tensor, ), cur_label, patch_id
