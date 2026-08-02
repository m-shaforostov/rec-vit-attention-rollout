import pickle
import numpy as np
import torch.nn as nn
import torch
import os
import sys
from .dataset_utils import load_data_for_art
from art.attacks.evasion import ProjectedGradientDescent, ElasticNet
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))))
from config.definitions import AES_DIR


def compute_attack(dataset, model, pretrained_model, attack, optimizer, filename, device, num_inputs=10000,
                   loss=nn.CrossEntropyLoss(), exclude_incorrect=True, conf=0.0, bin_s_steps=50, max_iter=1000,
                   init_const=0.001, batch=512, beta=0.001, eps=0.1, eps_step=0.01, num_rand_init=200, target=None,
                   adv_conf=0.95, high_conf=False, imsize=32, transform='default'):

    assert attack in ['L2', 'Linf']

    segmentation = True if dataset == 'PET' else False

    (_, _), (x_test, y_test, y_seg), input_shape, classes = load_data_for_art(dataset, batch_size_train=200,
                                                                              batch_size_test=200, transform=transform,
                                                                              imsize=imsize, seg_mask=segmentation)

    perturbation_mask = np.zeros(y_seg.shape)
    perturbation_mask[np.int32(y_seg*255) != 0] = 1

    perturbation_mask = np.broadcast_to(perturbation_mask, (perturbation_mask.shape[0], 3,
                                                            perturbation_mask.shape[2], perturbation_mask.shape[3]))

    random_indices = np.arange(x_test.shape[0])
    random_indices = np.random.permutation(random_indices)
    random_indices = random_indices[:num_inputs]

    x_test = x_test[random_indices, :, :, :]
    y_test = y_test[random_indices, :]
    perturbation_mask = perturbation_mask[random_indices, :, :, :]
    y_seg = y_seg[random_indices, :, :, :]

    model.load_state_dict(torch.load(pretrained_model))
    model.eval()

    classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=len(classes),
        device_type=device,
        preprocessing=(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
        clip_values=(0, 1)
    )

    if attack == 'L2':
        attack = ElasticNet(classifier, confidence=conf, binary_search_steps=bin_s_steps, max_iter=max_iter,
                            beta=beta, initial_const=init_const, batch_size=batch, decision_rule='L2')
    elif attack == 'Linf':
        attack = ProjectedGradientDescent(classifier, norm='inf', eps=eps, eps_step=eps_step, max_iter=max_iter,
                                          num_random_init=num_rand_init, batch_size=batch)
    else:
        raise ValueError(f'Attack {attack} is not defined!')

    if exclude_incorrect:
        prediction = classifier.predict(x_test)
        init_mask = np.argmax(prediction, axis=1) == np.argmax(y_test, axis=1)
        x_test = x_test[init_mask, :, :, :]
        y_test = y_test[init_mask, :]
        perturbation_mask = perturbation_mask[init_mask, :, :, :]
        y_seg = y_seg[init_mask, :, :, :]

    if target is not None:
        x_test_adv = attack.generate(x=x_test, y=target)
        data_mask = np.argmax(classifier.predict(x_test_adv), axis=1) == np.argmax(target, axis=1)
    else:
        x_test_adv = attack.generate(x=x_test, mask=perturbation_mask)
        data_mask = np.argmax(classifier.predict(x_test_adv), axis=1) != np.argmax(y_test, axis=1)

    success_rate = np.sum(data_mask)/len(data_mask)

    x_test_adv = x_test_adv[data_mask, :, :, :]
    original = x_test[data_mask, :, :, :]
    y_test_adv = y_test[data_mask, :]
    y_seg = y_seg[data_mask, :, :, :]

    if high_conf:
        soft_max = torch.softmax(torch.tensor(classifier.predict(x_test_adv)), dim=1)
        data_mask = np.max(soft_max.numpy(), axis=1) > adv_conf
        x_test_adv = x_test_adv[data_mask, :, :, :]
        original = original[data_mask, :, :, :]
        y_test_adv = y_test[data_mask, :]
        y_seg = y_seg[data_mask, :, :, :]
        if len(data_mask) != 0:
            success_rate *= np.sum(data_mask) / len(data_mask)

    print("Success of the attack: {}%".format(success_rate * 100))

    with open(filename, 'wb') as f:
        pickle.dump([original, x_test_adv, y_test_adv, y_seg], f)


def load_adversarial_examples(dataset, perturbation='low', batch_size=64, im_size=224, shuffle=False, loader='all'):
    """
    Returns a dataloader with AEs and relevant dat - normalized (it is prepared for prediction via NNs)
    For Cifar 10: (original images, adversarial images, targets)
    For PET dataset: (original images, adversarial images, targets, segmentation masks)
    """

    if dataset == 'CIFAR_10':
        with open(AES_DIR + f'/CIFAR_10/adv_PGD_with_labels.pkl', 'rb') as f:
            [original, adv, target] = pickle.load(f)

        original = torch.Tensor(original)
        adv = torch.Tensor(adv)
        target = torch.LongTensor(target)

        transformation = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if loader == 'original':
            dataloader = DataLoader(TensorDataset(transformation(original), target),
                                    batch_size=batch_size, shuffle=shuffle)
        elif loader == 'adversarial':
            dataloader = DataLoader(TensorDataset(transformation(adv), target),
                                    batch_size=batch_size, shuffle=shuffle)
        elif loader == 'all':
            dataloader = DataLoader(TensorDataset(transformation(original), transformation(adv), target),
                                    batch_size=batch_size, shuffle=shuffle)
        else:
            raise ValueError(f'Loader {loader} not recognized!')

    elif dataset == 'PET':
        with open(AES_DIR + f'/PET/adv_PGD_pet_{perturbation}.pkl', 'rb') as f:
            [original, adv, target, seg_mask] = pickle.load(f)

        original = torch.Tensor(original)
        adv = torch.Tensor(adv)
        target = torch.LongTensor(target)
        seg_mask = torch.Tensor(seg_mask)

        if loader == 'original':
            dataloader = DataLoader(TensorDataset(original, target), batch_size=batch_size,
                                    shuffle=shuffle)
        elif loader == 'adversarial':
            dataloader = DataLoader(TensorDataset(adv, target), batch_size=batch_size,
                                    shuffle=shuffle)
        elif loader == 'all':
            dataloader = DataLoader(TensorDataset(original, adv, target, seg_mask), batch_size=batch_size,
                                    shuffle=shuffle)
        else:
            raise ValueError(f'Loader {loader} not recognized!')

    else:
        raise ValueError(f'Adversarial examples for the dataset {dataset} not found!')

    print(f'Adversarial examples successfully loaded and dataloader created. \nNumber of AEs: {adv.shape[0]} \n')

    return dataloader
