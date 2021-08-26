from .mydataset import ImageFolder
from .unaligned_data_loader import UnalignedDataLoader
from collections import Counter
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision.transforms.transforms import *
import random
from torchvision.transforms.functional import _get_perspective_coeffs
from easydl import *
#from data_loader.config import *

def get_loader(source_path, target_path, evaluation_path, transforms,
               batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, test_loader, target_folder_train


def get_loader_class_inc(source_path, target_path, target_labeled_path, evaluation_path, transforms,
                         batch_size=32, return_id=False, balanced=False):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    target_folder_labeled = ImageFolder(os.path.join(target_labeled_path),
                                        transform=transforms[target_labeled_path],
                                        return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    target_labeled_loader = torch.utils.data.DataLoader(
        target_folder_labeled,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, target_labeled_loader, \
           test_loader, target_folder_train


def get_loader_balanced(source_path, target_path, evaluation_path, transforms, batch_size=32):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=True)
    freq = Counter(source_folder.labels)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder.labels]
    sampler = WeightedRandomSampler(source_weights, len(source_folder.labels))

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, test_loader
######################################################################
def perspective(img, startpoints, endpoints, interpolation=Image.BICUBIC):
    """Perform perspective transform of the given PIL Image.

    Args:
        img (PIL Image): Image to be transformed.
        startpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the orignal image
        endpoints: List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image
        interpolation: Default- Image.BICUBIC
    Returns:
        PIL Image:  Perspectively transformed Image.
    """

    coeffs = _get_perspective_coeffs(startpoints, endpoints)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, interpolation, fillcolor=(255, 255, 255))

class MyRandomPerspective(object):
    """Performs Perspective transformation of the given PIL Image randomly with a given probability.

    Args:
        interpolation : Default- Image.BICUBIC

        p (float): probability of the image being perspectively transformed. Default value is 0.5

        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.

    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.

        Returns:
            PIL Image: Random perspectivley transformed image.
        """

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return perspective(img, startpoints, endpoints, self.interpolation)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width : width of the image.
            height : height of the image.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

train_transform = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    CenterCrop(224),
    RandomGrayscale(p=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform2 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[0]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform3 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    FiveCrop(224),
    Lambda(lambda crops: crops[1]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform4 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[2]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform5 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[3]),
    RandomGrayscale(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

a, b, c = 10, 10, 31
c = c - a - b
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]
target_private_classes = [i + a + b for i in range(c)]

source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes
##############################改list_path path_prefix batch_size

source_train_ds = FileListDataset(list_path='txt/source_amazon_opda.txt', path_prefix='',
                                  transform=train_transform, filter=(lambda x: x in source_classes))
source_train_ds2 = FileListDataset(list_path='txt/source_amazon_opda.txt', path_prefix='',
                                   transform=train_transform2, filter=(lambda x: x in source_classes))
'''
source_train_ds3 = FileListDataset(list_path='txt/source_amazon_opda.txt', path_prefix='',
                                   transform=train_transform3, filter=(lambda x: x in source_classes))
source_train_ds4 = FileListDataset(list_path='txt/source_amazon_opda.txt', path_prefix='',
                                   transform=train_transform4, filter=(lambda x: x in source_classes))
source_train_ds5 = FileListDataset(list_path='txt/source_amazon_opda.txt', path_prefix='',
                                   transform=train_transform5, filter=(lambda x: x in source_classes))
'''
classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x: 1.0 / freq[x] if True else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler1 = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=36,
                             sampler=sampler1, num_workers=4, drop_last=True)
source_train_dl2 = DataLoader(dataset=source_train_ds2, batch_size=36,
                              sampler=sampler1, num_workers=4, drop_last=True)
'''
source_train_dl3 = DataLoader(dataset=source_train_ds3, batch_size=36,
                              sampler=sampler1, num_workers=4, drop_last=True)
source_train_dl4 = DataLoader(dataset=source_train_ds4, batch_size=36,
                              sampler=sampler1, num_workers=4, drop_last=True)
source_train_dl5 = DataLoader(dataset=source_train_ds5, batch_size=36,
                              sampler=sampler1, num_workers=4, drop_last=True)
'''
