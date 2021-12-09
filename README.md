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

:t-rex:News!  We have updated the proposed CrossMLP(December 9th, 2021)!

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
Run the `train_crossMlp.sh`, whose content is shown as follows

```
python train.py --dataroot [path_to_dataset] \
	--name [experiment_name] \
	--model crossmlpgan \
	--which_model_netG unet_256 \
	--which_direction AtoB \
	--dataset_mode aligned \
	--norm batch \
	--gpu_ids 0 \
	--batchSize [BS] \
	--loadSize [LS] \
	--fineSize [FS] \
	--no_flip \
	--display_id 0 \
	--lambda_L1 100 \
	--lambda_L1_seg 1
```
- For dayton or dayton_ablation dataset, [BS,LS,FS]=[4,286,256], set `--niter 20 --niter_decay 15`
- For cvusa dataset, [BS,LS,FS]=[4,286,256], set `--niter 15 --niter_decay 15`

There are many options you can specify. Please use `python train.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `export CUDA_VISIBLE_DEVICES=[GPU_ID]`. Training will cost about 3 days for `dayton` , less than 2 days for `dayton_ablation`, and less than 3 days for `cvusa`  with the default `--batchSize` on one TITAN Xp GPU (12G). So we suggest you use a larger --batchSize, while performance is not tested using a larger --batchSize

To view training results and loss plots on local computers, set --display_id to a non-zero value and run python -m visdom.server on a new terminal and click the URL http://localhost:8097. On a remote server, replace localhost with your server's name, such as http://server.trento.cs.edu:8097.

## Testing
Run the `test_crossMlp.sh`, whose content is shown as follows:
```
python test.py --dataroot [path_to_dataset] \
--name crossMlp_dayton_ablation \
--model crossmlpgan \
--which_model_netG unet_256 \
--which_direction AtoB \
--dataset_mode aligned \
--norm batch \
--gpu_ids 0 \
--batchSize 8 \
--loadSize 286 \
--fineSize 256 \
--saveDisk  \ 
--no_flip --eval
```
By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

We also provide image IDs used in our paper [here](https://github.com/Amazingren/CrossMLP/blob/main/scripts/Image_ids.txt) for further qualitative comparsion.

## Evaluation

Coming soon

## Generating Images Using Pretrained Model

Coming soon

## Contributions

If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Bin Ren ([bin.ren@unitn.it](bin.ren@unitn.it)).


## Acknowledgments
This source code borrows heavily from Pix2pix and SelectionGAN. We also thank the authors X-Fork & X-Seq for providing the evaluation codes. This work was supported by the EU H2020 AI4Media No.951911project and by the PRIN project PREVUE.