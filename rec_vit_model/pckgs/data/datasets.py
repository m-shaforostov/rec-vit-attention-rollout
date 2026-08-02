import numpy as np
import torchvision
from torchvision import transforms
import torch
import os
from config.definitions import DATASET_DIR
from torchvision.transforms import functional


class Transformation:
    """
    Defines data transformations and its inverse
    """
    def __init__(self, transformation_type):
        predefined = {'imagenet_statistics', 'identity'}
        if transformation_type not in predefined:
            raise ValueError(f'Transformation type {transformation_type} is not defined!')
        self.transformation_type = transformation_type

    def transformation(self):
        if self.transformation_type == 'imagenet_statistics':
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        elif self.transformation_type == 'identity':
            return transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

    def inverse_transformation(self):
        if self.transformation_type == 'imagenet_statistics':
            transformations = [transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])]
            return transforms.Compose(transformations)

        elif self.transformation_type == 'identity':
            return transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])


class Squarify:
    """
    Complete data (prevalently image data) to a square shape (it can be off 1px due to symmetrical padding)
    """
    def __call__(self, im):
        w, h = im.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return functional.pad(im, padding, 0, 'constant')


def load_data(dataset, transform=None, batch_size_train=64, batch_size_test=64, imsize=32, get_seg_mask=False):
    if dataset == 'CIFAR10':
        if transform is None or transform == 'basic':
            transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                              (0.5, 0.5, 0.5))])
            transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                             (0.5, 0.5, 0.5))])
        elif transform == 'match_imagenet_statistics':
            # for networks pretrained on ImageNet
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

        elif transform == 'toN01':
            transform_train = transforms.Compose(
                [transforms.Resize((imsize, imsize)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                      std=[0.24703233, 0.24348505, 0.26158768])])
            transform_test = transforms.Compose(
                [transforms.Resize((imsize, imsize)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                      std=[0.24703233, 0.24348505, 0.26158768])])

        elif transform == 'special_2':
            tr = [transforms.RandomCrop(32, padding=4),
                  transforms.Resize((imsize, imsize)),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

            transform_train = transforms.Compose(tr)

            transform_test = transforms.Compose([
                transforms.Resize((imsize, imsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        else:
            raise ValueError('Transform unknown')

        train_set = torchvision.datasets.CIFAR10(root=os.path.join(DATASET_DIR, 'CIFAR10'), train=True, download=True,
                                                 transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=0)

        test_set = torchvision.datasets.CIFAR10(root=os.path.join(DATASET_DIR, 'CIFAR10'), train=False, download=True,
                                                transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True, num_workers=0)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        input_shape = (3, 32, 32)

    elif dataset == 'PET':
        # TRAINING DATA
        data_transform = Transformation(transform).transformation()

        transform_train = transforms.Compose([
            Squarify(),
            transforms.Resize((imsize, imsize)),
            transforms.RandomCrop(imsize, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            data_transform,
        ])

        train_set = torchvision.datasets.OxfordIIITPet(root=os.path.join(DATASET_DIR, 'PET'), split='trainval',
                                                       target_types='category', transform=transform_train,
                                                       download=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True, num_workers=0)

        # TESTING DATA
        transform_test = transforms.Compose([
            Squarify(),
            transforms.Resize((imsize, imsize)),
            transforms.ToTensor(),
            data_transform,
        ])

        test_set_targ = torchvision.datasets.OxfordIIITPet(root=os.path.join(DATASET_DIR, 'PET'), split='test',
                                                           target_types='category', transform=transform_test,
                                                           download=True)

        # RETURN TEST LOADER IN THE FORM OF ZIPPED LOADERS (one with category targets and the other with segmentations)
        if get_seg_mask:
            transform_seg_mask = transforms.Compose(
                [Squarify(), transforms.Resize((imsize, imsize)), transforms.ToTensor()])

            test_set_seg = torchvision.datasets.OxfordIIITPet(root=os.path.join(DATASET_DIR, 'PET'), split='test',
                                                              target_types='segmentation', transform=transform_test,
                                                              target_transform=transform_seg_mask, download=True)

            test_loader_seg = torch.utils.data.DataLoader(test_set_seg, batch_size=batch_size_test, shuffle=False,
                                                          num_workers=0)

            test_loader_targ = torch.utils.data.DataLoader(test_set_targ, batch_size=batch_size_test, shuffle=False,
                                                           num_workers=0)

            test_loader = zip(test_loader_targ, test_loader_seg)

        else:
            test_loader = torch.utils.data.DataLoader(test_set_targ, batch_size=batch_size_test, shuffle=True,
                                                      num_workers=0)

        classes = [str(i) for i in range(37)]
        input_shape = (3, 224, 224)

    else:
        raise ValueError(f'the dataset {dataset} is not in the list of supported datasets!')

    return train_loader, test_loader, input_shape, classes
