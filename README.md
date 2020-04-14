# MIDL2020

This is the implementation of the [Classification of Epithelial Ovarian Carcinoma Whole-Slide Pathology Images Using Deep Transfer Learning](https://openreview.net/forum?id=VXdQD8B307). The code was written by Yiping Wang. If you use this code for your research, please cite:

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

# Prerequisites
- Linux or macOS
- Python 3.5.2
- PyTorch 1.0.0
- scikit-learn 0.22.2.post1
- NVIDIA GPU + CUDA CuDNN

# Datasets
The epithelial ovarian carcinoma whole-slide pathology images used in this study are available from the [corresponding author](mailto:ali.bashashati@ubc.ca) upon reasonable request.

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


