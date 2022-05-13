import os
import gc

import matplotlib.pyplot as plt
import numpy as np

from universal.data_loading import download_training_data, download_testing_data, get_trainval_dataset, \
    get_test_dataset, VOCDataset, test_transforms, train_transforms, classes
from universal.deepfool import deepfool
from universal.plot import plot_images
from universal.prediction_memory import PredictionMemory
from universal.universal_pert import universal_perturbation
from universal.vgg11_model import *
import logging
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader, Subset
import argparse
import torchvision.models as models
import random

if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    parser = argparse.ArgumentParser(description='Compute universal perturbation on VGG-11.')
    parser.add_argument('-s', '--state-dict', type=str, help='File path of a pretrained state dict', default=None)
    parser.add_argument('--train-data', type=str, help='Directory that contains train data', default='.data/VOC2012')
    parser.add_argument('--test-data', type=str, help='Directory that contains test data', default='.data/VOC2012_TEST')
    parser.add_argument('-f', '--force-download', action=argparse.BooleanOptionalAction)
    parser.add_argument('-d', '--device', type=str, help='', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-v', '--visualize', help='', action=argparse.BooleanOptionalAction)
    parser.add_argument('-p', '--perturbation', type=str, help='', default=None)
    parser.add_argument('-o', '--output', type=str, help='', default=None)
    args = parser.parse_args()
    logging.debug(args)

    vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)
    OUTPUT_DIM = 20
    device = args.device
    model = VGG(vgg11_layers, OUTPUT_DIM, device)
    logging.info(f"Device = {device}")

    logging.info(f"Prepare data. Train/validation data: {args.train_data}, Test data: {args.test_data}")
    download_training_data(args.train_data, args.force_download)
    download_testing_data(args.test_data, args.force_download)
    train_data_df, valid_data_df = get_trainval_dataset(args.train_data)
    #test_data_df = get_test_dataset(args.test_data)

    train_data = VOCDataset(train_data_df, args.train_data, train_transforms)
    valid_data = VOCDataset(valid_data_df, args.train_data, test_transforms)
    #test_data = VOCDataset(test_data_df, args.test_data, test_transforms)

    if args.visualize:
        images, labels = zip(*[(image, label) for image, label in
                               [train_data[i] for i in random.sample(range(len(train_data)), 25)]])

        plot_images(images, labels, classes)

    if args.state_dict:
        logging.info(f"State dict {args.state_dict} found. Load pretrained weights.")
        model.load_state_dict(torch.load(args.state_dict))
        # model.eval()
    else:
        logging.info(f"State dict not found.")
        pretrained_model = models.vgg11_bn(pretrained=True)

        IN_FEATURES = pretrained_model.classifier[-1].in_features

        final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
        pretrained_model.classifier[-1] = final_fc

        model.load_state_dict(pretrained_model.state_dict())

    pm = PredictionMemory()

    def classifier(img, use_memory=False):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if use_memory:
            v = pm.get(img.detach().numpy())
            if not v is None:
                return v
        v = model(img)[0]
        if use_memory:
            pm.set(img.detach().numpy(), v)
        return v

    del train_data_df
    del train_data
    gc.collect()

    file_perturbation = os.path.join('.data', 'universal.npy')

    v = np.load(args.perturbation) if args.perturbation else universal_perturbation(torch.stack([valid_data[i][0] for i in random.sample(range(len(valid_data)), 100)]), classifier, num_classes=len(classes))

    if args.output:
        np.save(args.output, v)

    if args.visualize:
        a = v.squeeze().transpose(1, 2, 0)
        print(np.linalg.norm(a))
        perturbation = (a - np.min(a))/np.ptp(a)
        plt.matshow(perturbation)
        plt.show()

        sample =  random.sample(range(len(valid_data)), 25)

        images, labels = zip(*[(image, label) for image, label in
                               [valid_data[i] for i in sample]])
        outputs = model(torch.stack(images).to(device))
        _, predicted = outputs[0].max(1)

        plot_images(images, predicted, classes, true_labels=labels)

        #print(valid_data[0][0])

        images, labels = zip(*[(image + v.squeeze()*5, label) for image, label in
                               [valid_data[i] for i in sample ]])
        outputs = model(torch.stack(images).to(device))
        _, predicted_perturbed = outputs[0].max(1)

        plot_images(images, predicted_perturbed, classes, true_labels=predicted)

