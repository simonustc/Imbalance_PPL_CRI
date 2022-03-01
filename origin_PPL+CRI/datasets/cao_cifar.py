# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import os
import cv2
import time
import json
import copy

import math
import torchvision.datasets
import numpy as np

import torch
import torchvision
from torchvision import transforms
import torchvision.datasets

def get_category_list(annotations, num_classes):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list,cat_list


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    def __init__(self, root, imb_ratio,imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)

        #self.cfg = cfg
        self.input_size = (32,32)
        self.color_space = 'RGB'
        self.imb_type =imb_type
        self.epoch_min=150
        self.epoch_max=160
        self.alpha=5.0
        self.ratio=imb_ratio
        imb_factor =imb_factor

        print("Use {} Mode to train network".format(self.color_space))

        rand_number = 0

        np.random.seed(rand_number)
        random.seed(rand_number)

        img_num_list = self.get_img_num_per_cls(self.cls_num, self.imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list,self.ratio)
        self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.data = self.all_info

        #print("{} Mode: Contain {} images".format(mode, len(self.data)))


        self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
        self.class_dict = self._get_class_dict()


        print('-'*20+'in imbalance cifar dataset'+'-'*20)
        print('class_dict is: ')
        print(self.class_dict)
        print('class_weight is: ')
        print(self.class_weight)


        num_list, cat_list = get_category_list(self.get_annotations(), self.cls_num)
        self.num_list=num_list
        self.instance_p = np.array([num / sum(num_list) for num in num_list])
        self.class_p = np.array([1/self.cls_num for _ in num_list])
        num_list_square = [math.sqrt(num) for num in num_list]
        self.square_p = np.array([num / sum(num_list_square) for num in num_list_square])
            #self.class_dict, self.origin_class_dict = self._get_class_dict()
        self.class_dict=self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        print('epoch in dataset', self.epoch)
        # if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive":
        #     self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
        #     print('self.progress_p', self.progress_p)
        # if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "spro":
        #     self.spro = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.square_p
        #     print('self.sprop', self.spro)

        if self.epoch<self.epoch_min:
            self.pp=self.instance_p
        elif self.epoch>=self.epoch_max:
            self.pp=self.class_p
        else:
            self.pp= (((self.epoch-self.epoch_min)/(self.epoch_max-self.epoch_min))**self.alpha) * self.class_p + (1-(((self.epoch-self.epoch_min)/(self.epoch_max-self.epoch_min))**self.alpha))*self.instance_p
        #print('self.pp', self.pp)
        # if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "spp":
        #     if self.epoch<self.epoch_min:
        #         self.spp=self.square_p
        #     elif self.epoch>self.epoch_max:
        #         self.spp=self.class_p
        #     else:
        #         self.spp= (((self.epoch-self.epoch_min)/(self.epoch_max-self.epoch_min))**self.alpha) * self.class_p + (1-(((self.epoch-self.epoch_min)/(self.epoch_max-self.epoch_min))**self.alpha))*self.square_p
        #     print('self.spp', self.spp)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train:
        #     #assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", 'square', 'progressive']
        #     if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
        #         sample_class = random.randint(0, self.cls_num - 1)
        #     elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "square":
        #         sample_class = np.random.choice(np.arange(self.cls_num), p=self.square_p)
        #     elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "progressive":
        #         sample_class = np.random.choice(np.arange(self.cls_num), p=self.progress_p)
        #     elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "spro":
        #         sample_class = np.random.choice(np.arange(self.cls_num), p=self.spro)
        #     elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "pp":
        #         sample_class = np.random.choice(np.arange(self.cls_num), p=self.pp)
        #     elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "spp":
        #         sample_class = np.random.choice(np.arange(self.cls_num), p=self.spp)

        # sample_class = np.random.choice(np.arange(self.cls_num), p=self.pp)
        # sample_indexes = self.class_dict[sample_class]
        #index = random.choice(sample_indexes)

        img, target = self.data[index]['image'], self.data[index]['category_id']
        meta = dict()


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight


    def _get_trans_image(self, img_idx):
        now_info = self.data[img_idx]
        img = now_info['image']
        img = Image.fromarray(img)
        return self.transform(img)[None, :, :, :]

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for d in self.all_info:
            annos.append({'category_id': int(d['category_id'])})
        return annos

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img =cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img#.convert('RGB')
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)

        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def gen_imbalanced_data(self, img_num_per_cls,ratio):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            the_img_num=int(the_img_num*ratio)
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            for img in self.data[selec_idx, ...]:
                new_data.append({
                    'image': img,
                    'category_id': the_class
                })
        self.all_info = new_data

    def data_format_transform(self):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        assert len(targets_np) == len(self.data)
        for i in range(len(self.data)):
            new_data.append({
                'image': self.data[i],
                'category_id': targets_np[i],
            })
        self.all_info = new_data


    def __len__(self):
        return len(self.data)


class CIFAR10_LT(object):

    def __init__(self,root='./data/imbalance_cifar10',imb_ratio=1.0, imb_type='exp',
                 imb_factor=0.01, batch_size=128, num_works=40):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = IMBALANCECIFAR10(root=root, imb_ratio=imb_ratio,imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True,
                                         download=True, transform=train_transform)
        eval_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)

        self.cls_num_list=train_dataset.num_list

        self.dist_sampler = None
        self.train_instance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        self.eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)

