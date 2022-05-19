import os
import torchvision.transforms as transforms
import pandas as pd
from functools import reduce
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


def download_training_data(dir, force=False):
    if not os.path.isdir(dir) or force:
        os.system('wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O /tmp/VOC.tar')
        os.system('tar -xvf /tmp/VOC.tar')
        os.system(f'mv VOCdevkit/VOC2012 {dir}')


def download_testing_data(dir, force=False):
    if not os.path.isdir(dir) or force:
        os.system('gdown --id 1gVllQTZdxA82byaZ_SjeOyjns0qnugwR -O /tmp/VOC_test.tar')
        os.system('tar -xvf /tmp/VOC_test.tar')
        os.system(f'mv VOCdevkit/VOC2012 {dir}')


pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(pretrained_size, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds)
])

test_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.CenterCrop(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds)
])

normalize = transforms.Normalize(mean=pretrained_means, std=pretrained_stds)

universal_transforms = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.CenterCrop(pretrained_size),
    transforms.ToTensor(),
])


def load_csv(dir, c, suffix):
    return pd.read_csv(f"{dir}/ImageSets/Main/{c}_{suffix}.txt", delimiter=r"\s+", header=None, names=('id', c))


classes = ['bicycle', 'horse', 'cat', 'dog', 'pottedplant', 'aeroplane', 'sofa', 'bus', 'car', 'chair', 'bird', 'boat',
           'cow', 'bottle', 'diningtable', 'person', 'tvmonitor', 'sheep', 'motorbike', 'train']


def get_trainval_dataset(dir):
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='id'), [load_csv(dir, c, 'trainval') for c in classes])
    df['label'] = df.apply(lambda row: row[classes].tolist().index(1), axis=1)
    with open(f"{dir}/ImageSets/Main/train.txt", 'r') as f:
        _ = f.read().split('\n')
        df['train'] = df.apply(lambda row: row.values[0] in _, axis=1)
    train_data_df = df[df['train']][['id', 'label']]
    valid_data_df = df[df['train'] == False][['id', 'label']]
    return train_data_df, valid_data_df


def get_test_dataset(dir):
    df_test = reduce(lambda df1, df2: pd.merge(df1, df2, on='id'), [load_csv(dir, c, 'test') for c in classes])
    df_test['label'] = np.zeros(len(df_test), dtype=int)

    return df_test[['id', 'label']]


class VOCDataset(Dataset):
    def __init__(self, data, dir, transform=None):
        self.data = data
        self.dir = dir
        self.transform = transform
        self.labels = data['label'].tolist()
        self.image_paths = data['id'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = f"{self.dir}/JPEGImages/{self.data.iloc[idx].values[0]}.jpg"
        image = Image.open(image_filepath)

        label = self.data.iloc[idx].values[1]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
