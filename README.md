![Python 3.6](https://img.shields.io/badge/python-3.6.13-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Amazingren/CIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Amazingren/CIT/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

## CrossMLP
**[Cascaded Cross MLP-Mixer GANs for Cross-View Image Translation](https://arxiv.org/abs/2110.10183)**  <br> 
[Bin Ren](https://scholar.google.com/citations?user=Md9maLYAAAAJ&hl=en)<sup>1</sup>, [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en)<sup>2</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>1</sup>. <br> 
<sup>1</sup>University of Trento, Italy, <sup>2</sup>ETH, Switzerland.<br>
In [BMVC 2021](https://www.bmvc2021.com/) **Oral**. <br>
The repository offers the official implementation of our paper in PyTorch.


## Installation
- Step1: Create a new virtual environment with anaconda
 ```
 conda create -n crossmlp python=3.6
 ``` 

- Step2: Install the required libraries
```
pip install -r requirement.txt
```


## Dataset Preparation
For Dayton and CVUSA, the datasets must be downloaded beforehand. Please download them on the respective webpages. In addition, we put a few sample images in this code repo [data samples](https://github.com/Amazingren/CrossMLP/tree/main/datasets/samples). Please cite their papers if you use the data.

Preparing Ablation Dataset. We conduct ablation study in a2g (aerialto-ground) direction on Dayton dataset. To reduce the training time, we randomly select 1/3 samples from the whole 55,000/21,048 samples i.e. around 18,334 samples for training and 7,017 samples for testing. The trianing and testing splits can be downloaded [here](https://github.com/Amazingren/CrossMLP/tree/main/datasets/dayton_ablation_split).

Preparing Dayton Dataset. The dataset can be downloaded [here](https://github.com/lugiavn/gt-crossview). In particular, you will need to download dayton.zip. Ground Truth semantic maps are not available for this datasets. We adopt [RefineNet](https://github.com/guosheng/refinenet) trained on CityScapes dataset for generating semantic maps and use them as training data in our experiments. Please cite their papers if you use this dataset. Train/Test splits for Dayton dataset can be downloaded from [here](https://github.com/Amazingren/CrossMLP/tree/main/datasets/dayton_split).

Preparing CVUSA Dataset. The dataset can be downloaded [here](http://mvrl.cs.uky.edu/datasets/cvusa/). After unzipping the dataset, prepare the training and testing data as discussed in our CrossMLP. We also convert semantic maps to the color ones by using this [script](https://github.com/Amazingren/CrossMLP/blob/main/scripts/convert_semantic_map_cvusa.m). Since there is no semantic maps for the aerial images on this dataset, we use black images as aerial semantic maps for placehold purposes.

🌲 Note that for your convenience we also provide download scripts:
```
bash ./datasets/download_selectiongan_dataset.sh [dataset_name]
```
[dataset_name] can be:
- `dayton_ablation` : 5.7 GB
- `dayton`: 17.0 GB
- `cvusa`: 1.3 GB

## Training


## Testing


## Evaluation

## Generating Images Using Pretrained Model


## Contributions

If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Bin Ren ([bin.ren@unitn.it](bin.ren@unitn.it)).


## Acknowledgments
This source code borrows heavily from Pix2pix and SelectionGAN. We also thank the authors X-Fork & X-Seq for providing the evaluation codes. This work was supported by the EU H2020 AI4Media No.951911project and by the PRIN project PREVUE.