# PPL with CRI Loss for Imbalanced Classification

Our code is based on [MisLAS](https://arxiv.org/pdf/2104.00466.pdf) and [RIDE](https://people.eecs.berkeley.edu/~xdwang/papers/RIDE.pdf) models.

## Installation

### Requirements

* numpy==1.22.0  
* python==3.9
* pytorch==1.10.1
* torchvision==0.11.2
* tqdm==4.62.3
* pillow==8.4.0

### Dataset Preparation

* [cifar10 & cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)

* [ImageNet](http://image-net.org/index)

* [iNaturalist2018](https://github.com/visipedia/inat_comp/tree/master/2018)

For ImageNet-LT and iNaturalist2018, please prepare data in the data directory:
```
datasets
├── data_txt
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_val.txt
    ├── ImageNet_LT_train.txt
    └── ImageNet_LT_test.txt

```

getting the txt files from [data_txt file Link](https://drive.google.com/drive/folders/1ssoFLGNB_TM-j4VNYtgx9lxfqvACz-8V?usp=sharing)

For CRI+PPW+PPmix, change the `data_path` in `config/.../.yaml`;

For CRI+PPW#, change the `data_loader:{data_dir} in `./config/...json`.


## Training

one GPU for Imbalance cifar10 & cifar100, two GPUs for ImageNet-LT, and eight GPUs iNaturalist2018.

Backbone network can be resnet32 for Imbalance cifar10 & cifar100, resnet10 for ImageNet-LT, and resnet50 for iNaturalist2018.

### CRI+PPW+PPmix

#### Imbalance cifar10 & cifar100:

`python train.py --cfg ./config/cifar10/cifar10_CRI.yaml`

`python train.py --cfg ./config/cifar100/cifar100_CRI.yaml`

#### ImageNet-LT:

`python train.py --cfg ./config/imagenet/imagenet_CRI.yaml`

#### ina2018:

`python train.py --cfg ./config/ina2018/ina2018_CRI.yaml`

### CRI+PPW#

#### Imbalance cifar10 & cifar100:

`python train.py --cfg ./config/cifar10.json`

`python train.py --cfg ./config/cifar100.json`

#### ImageNet-LT:

`python train.py --cfg ./config/imagenet.json`

#### ina2018:

`python train.py --cfg ./config/ina2018.json`


## Validation

### CRI+PPW+PPmix

`python eval.py --cfg ./config/....yaml resume /path/ckps/...pth.tar`

### CRI+PPW+RIDE

`python eval.py --cfg ./config/....json --resume /path/...pth`


## Results and Models

### CRI+PPW+PPmix

[Links to models](https://drive.google.com/drive/folders/1b932TjGm_-GcuN9Mq24aExk2uZK64LWy?usp=sharing)

### CRI+PPW+RIDE

[Links to models](https://drive.google.com/drive/folders/1Dqh0Jcs-lqKv0BkEJmMX8JJwnhCL7mhx?usp=sharing)









