# MIDL2020

<p align="center">
<img src="./docs/schematic.png" width="600"/>
</p>


This is the implementation of the [Classification of Epithelial Ovarian Carcinoma Whole-Slide Pathology Images Using Deep Transfer Learning](https://openreview.net/forum?id=VXdQD8B307). The code was written by [Yiping Wang](http://yiping.wang.vision). If you use this code for your research, please cite:

```
@inproceedings{
wang2020classification,
title={Classification of Epithelial Ovarian Carcinoma Whole-Slide Pathology Images Using Deep Transfer Learning},
author={Yiping Wang, David Farnell, Hossein Farahani, Mitchell Nursey, Basile Tessier-Cloutier, Steven J.M. Jones, David G. Huntsman, C. Blake Gilks, Ali Bashashati},
booktitle={Submitted to Medical Imaging with Deep Learning},
year={2020},
url={https://openreview.net/forum?id=VXdQD8B307},
note={under review}
}
```

Our work is inspired by [ProGAN](https://github.com/tkarras/progressive_growing_of_gans) and [fast.ai](https://www.fast.ai/2018/04/30/dawnbench-fastai/).

# Prerequisites
- Linux or macOS
- Python 3.5.2
- PyTorch 1.0.0
- scikit-learn 0.22.2.post1
- NVIDIA GPU + CUDA CuDNN

# Get Started
### Installation
- Install the required packages
    - `pip install torch==1.3.0+cu92 torchvision==0.4.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html`
    - `pip install -r requirements.txt`

- Clone this repo
```
mkdir wsi_classification
cd wsi_classification
git clone https://github.com/AliBashashati/MIDL2020
cd MIDL2020
```

### Repo Structure

- `patch_level.py` is the patch-level classifier entry. Inside this file, function `train` initialize models, create data loader, optimize model weights, log training information, etc. `evaluate` simply loads the trained model and apply the model on validation or testing sets. 

- `config.py` is the program that reads arguments from the user. Therefore, you can custom any hyperparameters or settings through `config.py`

- `models` is the sub-module that contains implementation involves models.

    - Inside `models` sub-module, `models/base_model.py` is the template class for `models/models.py`. It defines various expected behaviours of a model, such as `forward`, `optimize_weights`, `load_state`, `save_state`, etc. Any models should inherit `BaseModel`. 

    - Inside `models` sub-module, `models/models.py` is the `models/networks.py` interface. It initialize `models/networks.py` and optimize the weights of `models/networks.py`. Different models in `models/models.py` are summaized in the following table. 

        | Models | Usage |
        | ------------- |:-------------:|
        | `CountBasedFusionModel` | Slide-Level model wrapper. It assumes an input of N*C matrix, where N represents the number of slides, C represents the number of classes. | 
        | `DeepModel` | Patch-Level model wrapper. It is a simple wrapper for out-of-box deep learning models. It takes one patch for each forward pass. It follows the standard deep learning training protocol. |

    - Inside `models` sub-module, `models/networks.py` is the implementation of various networks. It simply defines network architecture and `forward` functions. Keep it as simple as possible. 

- `data` is the sub-module that contains data loader, preprocess, and post-process functions. 

    - Inside `data` sub-module, `data/base_dataset.py` is the template class for other data loaders. It simply defines data loader behaviour, and assign `config.py` settings into the current data loader. Moreover, it also modifies the patch ids if requires, such as change multi-scale model scales, etc. 

    - Inside `data` sub-module, `data/patch_dataset.py` is the main patch images or patch features data loader. 

        | Dataset | Usage |
        | ------------- |:-------------:|
        | `SubtypePatchDataset` | It loads patch images from H5 files and applies preprocess steps.|

    - Inside `data` sub-module, `data/create_patient_groups.py` is the helper script to split the dataset by patients. 


- `utils` is the sub-module that contains simple but useful function snippet. 

    - Inside `utils` sub-module, `utils/utils.py` contains a wide range of useful small functions. Contributors should also add useful functions in here, and add docstring. Moreover, functions in here should be self-explained and as general as possible. 

    - Inside `utils` sub-module, `utils/subtype_enum.py` defines enum for classes. 


### Patch extraction
First of all, the `enum` in `utils/subtype_enum.py` should be defined.

Afterwards, we extract the 1024 * 1024 patches and then downsampled to 512 * 512 and 256 * 256 using the `extract_patches.py`. This script not only extract patches, but also store the patches into a H5 file for easy data transfer and management. However, we use our own data annotation file so the annotation parse and check portion needs to changed for other dataset. 

We store the patches in the h5 files who has the format `class_name/slide_id/patch_locaton_x_y` and use `.txt` files to store the data entry ids. 

### Patch-level: train, validation and test

The following bash script is used to invokve training, validation and test:
```
#!/bin/bash
chmod 775 ./maestro.py
echo 'Two-stage model using split A'

echo 'Stage 1 - Patch Size 256 * 256 Training'
./patch_level.py  --deep_model DeepModel --deep_classifier multi_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 64 --epoch 20 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 256 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_resize --preload_image_file_name 1024_resize.h5 --train_ids_file_name patch_ids/1_2_train_3_eval_train_ids.txt  --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/1024_resize_log/ --save_dir /projects/ovcare/classification/ywang/project_save/1024_resize_save/
echo 'Stage 1 - Patch Size 256 * 256 Validation'
./patch_level.py  --mode Validation --deep_model DeepModel --deep_classifier multi_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 64 --epoch 20 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 256 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_resize --preload_image_file_name 1024_resize.h5 --train_ids_file_name patch_ids/1_2_train_3_eval_train_ids.txt  --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/1024_resize_log/ --save_dir /projects/ovcare/classification/ywang/project_save/1024_resize_save/

echo 'Stage 2 - Patch Size 512 * 512 Training'
./patch_level.py  --deep_model DeepModel --deep_classifier multi_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 32 --epoch 20 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 512 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_resize --preload_image_file_name 1024_resize.h5 --train_ids_file_name patch_ids/1_2_train_3_eval_train_ids.txt  --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/1024_resize_log/ --save_dir /projects/ovcare/classification/ywang/project_save/1024_resize_save/
echo 'Stage 2 - Patch Size 512 * 512 Validation'
./patch_level.py  --mode Validation --deep_model DeepModel --deep_classifier multi_stage --model_name_prefix split_a --use_pretrained --lr 0.0002 --batch_size 32 --epoch 20 --rep_intv 250 --use_equalized_batch --n_eval_samples 2000 --is_multiscale_expert --expert_magnification 512 --dataset_dir /projects/ovcare/classification/ywang/midl_dataset/1024_resize --preload_image_file_name 1024_resize.h5 --train_ids_file_name patch_ids/1_2_train_3_eval_train_ids.txt  --val_ids_file_name patch_ids/1_2_train_3_eval_eval_0_ids.txt --log_dir /projects/ovcare/classification/ywang/project_log/1024_resize_log/ --save_dir /projects/ovcare/classification/ywang/project_save/1024_resize_save/
```

### Slide-level: train and test
We train Random Forests using 6-fold cross validation on the results of six patch-level test set. 

After changing the path to the six patch-level results in the `slide_level.py`, simply run `python3 slide_level.py` and it will output the slide-level results as well as save the trained model. 

### Our results
We include our patch-level and slide-level results in `./results/`. 

# Datasets and Detailed Results
The epithelial ovarian carcinoma whole-slide pathology images used in this study are available from the [corresponding author](mailto:ali.bashashati@ubc.ca) upon reasonable request. Moreover, the patch-level and slide-level model trained weights are also available from the [corresponding author](mailto:ali.bashashati@ubc.ca) upon reasonable request. 

Our dataset has the following distribution in terms of patients, slides, and 1024 * 1024 tumor patches:
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|32|14|28|9|76|159|
|Slide|53|29|55|11|157|305|
|Patch|16.49%|16.31%|12.93%|10.96%|43.31%|161516|

We first randomly divided the datasets by patients into three groups. We denote these three groups as Group 1, Group 2, and Group 3. 

Group 1 has the following distributions:
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|11|5|10|3|26|55|
|Slide|20|8|14|4|54|100|
|Patch|15.23%|10.43%|11.44%|13.93%|48.98%|56034|

Group 2 has the following distributions:
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|11|5|10|3|26|55|
|Slide|16|8|26|3|60|113|
|Patch|12.30%|17.80%|15.51%|7.09%|47.30%|64855|

Group 3 has the following distributions:
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|10|4|8|3|24|49|
|Slide|17|13|15|4|43|92|
|Patch|24.91%|22.05%|10.89%|13.03%|29.12%|40627|

## Slide-level and Patch-Level Results

<p align="center">
<img src="./docs/nested-cv.png" width="600" class="center"/>
</p>

For slide-level classification, we only use the patch-level test set results to build the input matrix to train random forests, and we report the 6-fold cross-validation slide-level results.

|Split|CC|LGSC|EC|MC|HGSC|Weighted Accuracy|Kappa|AUC|F1 Score|Average Accuracy|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|baseline 6-fold cross-validation|83.02%|65.52%|54.55%|54.55%|80.25%|73.77%|0.5993|0.9391|0.6855|67.58%|
|stage-1 6-fold cross-validation|79.25%|79.31%|61.82%|54.55%|85.99%|78.69%|0.6730|0.9375|0.7414|72.18%|
|stage-2 6-fold cross-validation|86.79%|100.00%|74.55%|81.82%|90.45%|87.54%|0.8106|0.9641|0.8718|86.72%|

For patch-level classification, we employ a 3-fold cross-validation scheme with a tweak. We use two of three patient groups as the training set and divide the remaining group equally by patient into two subgroups, one of the subgroups will be used as validation *or* test set. Therefore, we eventually have 6 different training, validation and test set.


### Split A Distribution and Patch-level Classifier Test Results

#### Training set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|22|10|20|6|52|110|
|Slide|36|16|40|7|114|213|
|Patch|13.66%|14.39%|13.62%|10.26%|48.08%|120889|

#### Validation set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|4|2|12|25|
|Slide|7|3|9|3|19|41|
|Patch|17.31%|27.79%|7.85%|31.73%|15.33%|14594|

#### Test set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|4|1|12|24|
|Slide|10|10|6|1|24|51|
|Patch|29.18%|18.83%|12.59%|2.55%|36.85%|26033|

#### Baseline, Stage-1 and Stage-2 test results
|Model|CC|LGSC|EC|MC|HGSC|Weighted Accuracy|Kappa|AUC|F1 Score|Average Accuracy|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Baseline|99.22%|89.60%|73.89%|100.00%|75.05%|85.33%|0.8015|0.9739|0.8546|87.55%|
|Stage-1|99.42%|79.05%|63.58%|99.85%|74.70%|81.97%|0.7543|0.9651|0.8105|83.32%|
|Stage-2|99.50%|77.34%|72.70%|99.55%|72.64%|82.05%|0.7568|0.9658|0.8243|84.34%|

### Split B Distribution and Patch-level Classifier Test Results

#### Training set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|22|10|20|6|52|110|
|Slide|36|16|40|7|114|213|
|Patch|13.66%|14.39%|13.62%|10.26%|48.08%|120889|

#### Validation set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|4|1|12|24|
|Slide|10|10|6|1|24|51|
|Patch|29.18%|18.83%|12.59%|2.55%|36.85%|26033|

#### Test set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|4|2|12|25|
|Slide|7|3|9|3|19|41|
|Patch|17.31%|27.79%|7.85%|31.73%|15.33%|14594|

#### Baseline, Stage-1 and Stage-2 test results
|Model|CC|LGSC|EC|MC|HGSC|Weighted Accuracy|Kappa|AUC|F1 Score|Average Accuracy|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Baseline|89.90%|76.58%|81.05%|20.89%|58.78%|58.84%|0.4986|0.8969|0.5698|65.44%|
|Stage-1|89.59%|74.04%|75.72%|44.58%|50.38%|63.89%|0.5521|0.8997|0.6122|66.86%|
|Stage-2|86.26%|64.32%|63.23%|37.93%|60.84%|59.13%|0.4965|0.8823|0.5711|62.52%|

### Split C Distribution and Patch-level Classifier Test Results

#### Training set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|21|9|18|6|50|104|
|Slide|37|21|29|8|97|192|
|Patch|19.30%|15.31%|11.21%|13.55%|40.63%|96661|

#### Validation set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|6|3|5|2|13|29|
|Slide|7|3|13|2|25|50|
|Patch|6.87%|12.82%|12.82%|12.84%|54.64%|18990|

#### Test set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|5|1|13|26|
|Slide|9|5|13|1|35|63|
|Patch|14.55%|19.86%|16.62%|4.71%|44.26%|45865|

#### Baseline, Stage-1 and Stage-2 test results
|Model|CC|LGSC|EC|MC|HGSC|Weighted Accuracy|Kappa|AUC|F1 Score|Average Accuracy|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Baseline|95.11%|54.81%|80.17%|96.76%|63.07%|70.52%|0.6035|0.9335|0.7300|77.99%|
|Stage-1|84.44%|32.51%|77.52%|97.45%|81.99%|72.51%|0.6132|0.9276|0.7162|74.78%|
|Stage-2|97.06%|58.49%|78.55%|97.64%|68.81%|73.84%|0.6452|0.9507|0.7532|80.11%|

### Split D Distribution and Patch-level Classifier Test Results

#### Training set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|21|9|18|6|50|104|
|Slide|37|21|29|8|97|192|
|Patch|19.30%|15.31%|11.21%|13.55%|40.63%|96661|

#### Validation set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|5|1|13|26|
|Slide|9|5|13|1|35|63|
|Patch|14.55%|19.86%|16.62%|4.71%|44.26%|45865|

#### Test set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|6|3|5|2|13|29|
|Slide|7|3|13|2|25|50|
|Patch|6.87%|12.82%|12.82%|12.84%|54.64%|18990|

#### Baseline, Stage-1 and Stage-2 test results
|Model|CC|LGSC|EC|MC|HGSC|Weighted Accuracy|Kappa|AUC|F1 Score|Average Accuracy|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Baseline|64.29%|83.37%|48.71%|99.84%|80.76%|78.30%|0.6790|0.9473|0.7212|75.39%|
|Stage-1|76.55%|79.10%|40.04%|98.24%|87.14%|80.77%|0.7056|0.9423|0.7383|76.21%|
|Stage-2|65.06%|84.11%|33.76%|95.41%|88.33%|80.10%|0.6881|0.9277|0.7261|73.33%|

### Split E Distribution and Patch-level Classifier Test Results

#### Training set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|21|9|18|6|50|104|
|Slide|33|21|41|7|103|205|
|Patch|17.16%|19.44%|13.73%|9.38%|40.30%|105482|

#### Validation set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|6|3|5|2|13|29|
|Slide|11|3|7|3|27|51|
|Patch|12.62%|10.19%|11.73%|39.90%|25.57%|16690|

#### Test set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|5|1|13|26|
|Slide|9|5|7|1|27|49|
|Patch|16.33%|10.54%|11.32%|2.91%|58.91%|39344|

#### Baseline, Stage-1 and Stage-2 test results
|Model|CC|LGSC|EC|MC|HGSC|Weighted Accuracy|Kappa|AUC|F1 Score|Average Accuracy|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Baseline|79.29%|88.15%|26.77%|99.74%|65.01%|66.47%|0.5003|0.9149|0.6336|71.79%|
|Stage-1|61.00%|95.37%|58.06%|98.78%|68.85%|70.01%|0.5490|0.9371|0.7058|76.41%|
|Stage-2|70.92%|90.98%|47.80%|97.81%|66.74%|68.74%|0.5285|0.9156|0.7067|74.85%|

### Split F Distribution and Patch-level Classifier Test Results

#### Training set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|21|9|18|6|50|104|
|Slide|33|21|41|7|103|205|
|Patch|17.16%|19.44%|13.73%|9.38%|40.30%|105482|

#### Validation set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|5|2|5|1|13|26|
|Slide|9|5|7|1|27|49|
|Patch|16.33%|10.54%|11.32%|2.91%|58.91%|39344|

#### Test set
|Data Type|CC|LGSC|EC|MC|HGSC|Total|
| --- | --- | --- | --- | --- | --- | --- |
|Patient|6|3|5|2|13|29|
|Slide|11|3|7|3|27|51|
|Patch|12.62%|10.19%|11.73%|39.90%|25.57%|16690|

#### Baseline, Stage-1 and Stage-2 test results
|Model|CC|LGSC|EC|MC|HGSC|Weighted Accuracy|Kappa|AUC|F1 Score|Average Accuracy|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|Baseline|94.78%|31.24%|65.25%|39.14%|51.64%|51.61%|0.3993|0.8751|0.5124|56.41%|
|Stage-1|95.58%|33.35%|90.04%|33.88%|58.13%|54.40%|0.4404|0.8893|0.5464|62.20%|
|Stage-2|95.96%|28.12%|53.65%|41.63%|75.05%|57.06%|0.4615|0.8287|0.5492|58.88%|



