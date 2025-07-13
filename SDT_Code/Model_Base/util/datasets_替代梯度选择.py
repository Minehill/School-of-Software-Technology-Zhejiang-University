# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if not hasattr(args, 'data_set'):
        args.data_set = 'IMNET'

    if args.data_set == 'CIFAR10':
        root = os.path.join(args.data_path) 
        dataset = datasets.CIFAR10(root=root, train=is_train, transform=transform, download=False)
        args.nb_classes = 10 
        print(f"Loading local CIFAR-10 dataset. Train: {is_train}. Number of classes: {args.nb_classes}")
        return dataset
        
    elif args.data_set == 'CIFAR100':
        root = os.path.join(args.data_path)
        dataset = datasets.CIFAR100(root=root, train=is_train, transform=transform, download=False)
        args.nb_classes = 100
        print(f"Loading local CIFAR-100 dataset. Train: {is_train}. Number of classes: {args.nb_classes}")
        return dataset
        
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        args.nb_classes = 1000
        print(f"Loading ImageNet dataset. Train: {is_train}. Number of classes: {args.nb_classes}")
        return dataset
        
    else:
        raise ValueError(f"Dataset '{args.data_set}' is not supported.")

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 232
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)