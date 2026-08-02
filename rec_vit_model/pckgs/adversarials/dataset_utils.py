# functions used for preprocessing data, e.g. to make them suitable for ART

from pckgs.data.datasets import load_data
import numpy as np
import torch.nn.functional as F


def load_data_for_art(dataset, batch_size_train=64, batch_size_test=64, transform=None, imsize=32, seg_mask=False):
    """
    function that loads dataset for usage with ART, that is, instead of train_loader, test_loader this returns
    (x_train, y_train), (x_test, y_test) where all four are numpy arrays of type (batch, channels, image sizes)
    """

    train_loader, test_loader, input_shape, classes = load_data(dataset, batch_size_train=batch_size_train,
                                                                batch_size_test=batch_size_test, transform=transform,
                                                                imsize=imsize, get_seg_mask=seg_mask)
    num_classes = len(classes)
    x_train, y_train = None, None
    x_test, y_test = None, None

    for data, label in train_loader:
        if y_train is None:
            y_train = F.one_hot(label, num_classes=num_classes).cpu().detach().numpy()
        else:
            y_train = np.append(y_train, F.one_hot(label, num_classes=num_classes).cpu().detach().numpy(), axis=0)
        if x_train is None:
            x_train = data.cpu().detach().numpy()
        else:
            x_train = np.append(x_train, data.cpu().detach().numpy(), axis=0)

    if seg_mask:
        y_seg = None
        for data_target, data_seg in test_loader:
            label = data_target[1]
            data = data_seg[0]
            segmentation = data_seg[1]

            if y_test is None:
                y_test = F.one_hot(label, num_classes=num_classes).cpu().detach().numpy()
            else:
                y_test = np.append(y_test, F.one_hot(label, num_classes=num_classes).cpu().detach().numpy(), axis=0)

            if x_test is None:
                x_test = data.cpu().detach().numpy()
            else:
                x_test = np.append(x_test, data.cpu().detach().numpy(), axis=0)

            if y_seg is None:
                y_seg = segmentation.cpu().detach().numpy()
            else:
                y_seg = np.append(y_seg, segmentation.cpu().detach().numpy(), axis=0)

        return (x_train, y_train), (x_test, y_test, y_seg), input_shape, classes

    else:
        for data, label in test_loader:
            if y_test is None:
                y_test = F.one_hot(label, num_classes=num_classes).cpu().detach().numpy()
            else:
                y_test = np.append(y_test, F.one_hot(label, num_classes=num_classes).cpu().detach().numpy(), axis=0)
            if x_test is None:
                x_test = data.cpu().detach().numpy()
            else:
                x_test = np.append(x_test, data.cpu().detach().numpy(), axis=0)

        return (x_train, y_train), (x_test, y_test), input_shape, classes


def transform_for_art_from_loader(train_loader, test_loader, num_classes):
    x_train, y_train = None, None
    x_test, y_test = None, None

    for data, label in train_loader:
        if y_train is None:
            y_train = F.one_hot(label, num_classes=num_classes).cpu().detach().numpy()
        else:
            y_train = np.append(y_train, F.one_hot(label, num_classes=num_classes).cpu().detach().numpy(), axis=0)
        if x_train is None:
            x_train = data.cpu().detach().numpy()
        else:
            x_train = np.append(x_train, data.cpu().detach().numpy(), axis=0)
    for data, label in test_loader:
        if y_test is None:
            y_test = F.one_hot(label, num_classes=num_classes).cpu().detach().numpy()
        else:
            y_test = np.append(y_test, F.one_hot(label, num_classes=num_classes).cpu().detach().numpy(), axis=0)
        if x_test is None:
            x_test = data.cpu().detach().numpy()
        else:
            x_test = np.append(x_test, data.cpu().detach().numpy(), axis=0)
    return (x_train, y_train), (x_test, y_test)


def transform_for_art_from_data(data, label, num_classes):
    x_test, y_test = None, None
    if y_test is None:
        y_test = F.one_hot(label, num_classes=num_classes).cpu().detach().numpy()
    else:
        y_test = np.append(y_test, F.one_hot(label, num_classes=num_classes).cpu().detach().numpy(), axis=0)
    if x_test is None:
        x_test = data.cpu().detach().numpy()
    else:
        x_test = np.append(x_test, data.cpu().detach().numpy(), axis=0)
    return x_test, y_test


# Convert dataloader to numpy array, target is not one-hot!
def loader_to_np(loader):
    x, y = None, None
    for data, label in loader:
        if y is None:
            y = label.cpu().detach().numpy()
        else:
            y = np.append(y, label.cpu().detach().numpy(), axis=0)
        if x is None:
            x = data.cpu().detach().numpy()
        else:
            x = np.append(x, data.cpu().detach().numpy(), axis=0)
    return x, y
