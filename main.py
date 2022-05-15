import os
import gc

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("info.log"),
            logging.StreamHandler()
        ]
    )
    parser = argparse.ArgumentParser(description='Compute universal perturbation on VGG-11.')
    parser.add_argument('-s', '--state-dict', type=str, help='File path of a pretrained state dict', default=None)
    parser.add_argument('--train-data', type=str, help='Directory that contains train data', default='.data/VOC2012')
    parser.add_argument('--test-data', type=str, help='Directory that contains test data', default='.data/VOC2012_TEST')
    parser.add_argument('-f', '--force-download', action='store_true', default=False)
    parser.add_argument('-d', '--device', type=str, help='', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-v', '--visualize', help='', action='store_true', default=False)
    parser.add_argument('-p', '--perturbation', type=str, help='', default=None)
    parser.add_argument('-o', '--output', type=str, help='', default=None)
    parser.add_argument('-b', '--batch-size', type=int, help='', default=25)
    parser.add_argument('--image-per-class', type=int, help='', default=25)
    parser.add_argument('--max-iter', type=int, help='', default=np.inf)
    args = parser.parse_args()
    logging.debug(args)

    vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)
    OUTPUT_DIM = 20
    device = args.device
    model = VGG(vgg11_layers, OUTPUT_DIM, device)
    logging.info(f"Device = {device}")

    logging.info(f"Prepare data. Train/validation data: {args.train_data}, Test data: {args.test_data}")
    os.makedirs('.data/', exist_ok=True)
    download_training_data(args.train_data, args.force_download)
    #download_testing_data(args.test_data, args.force_download)
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
        model.eval()
    else:
        logging.info(f"State dict not found.")
        pretrained_model = models.vgg11_bn(pretrained=True)

        IN_FEATURES = pretrained_model.classifier[-1].in_features

        final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
        pretrained_model.classifier[-1] = final_fc

        model.load_state_dict(pretrained_model.state_dict())
    del train_data
    pm = PredictionMemory()

    def classifier(img):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        v = model(img)[0]
        return v

    gc.collect()

    file_perturbation = os.path.join('.data', 'universal.npy')

    if args.perturbation:
        v = np.load(args.perturbation)
    else:
        logging.info(f'Search {args.image_per_class} image per class '
                     f'(total: {args.image_per_class * len(classes)})')
        imgs = []
        ids = []
        counts = np.zeros(len(classes), dtype=int)
        with tqdm(total=(args.image_per_class * len(classes))) as pbar:
            for i in np.random.permutation(len(valid_data)):
                if len(imgs) < args.image_per_class * len(classes):
                    img, l = valid_data[i]
                    if counts[l] < args.image_per_class:
                        imgs.append(img)
                        ids.append(i)
                        counts[l] = counts[l] + 1
                        pbar.update(1)
                else:
                    print("All image found")
                    break
        logging.info(f'Selected imgs = {ids}')
        logging.info(f'Classes distribution = {counts}')
        logging.info("Starting computation of universal perturbation...")
        v = universal_perturbation(
            torch.stack(imgs),
            classifier,
            num_classes=len(classes),
            xi=2000,
            p=2,
            max_iter_uni=args.max_iter
        )

    if args.output:
        np.save(args.output, v)

    if args.visualize:
        a = v.squeeze().transpose(1, 2, 0)

        logging.info(f"Perturbation vector norm = {np.linalg.norm(abs(a))}")
        perturbation = (a - np.min(a))/np.ptp(a)
        plt.matshow(perturbation)
        plt.show()

        sample = random.sample(range(len(valid_data)), 25)

        images, labels = zip(*[(image, label) for image, label in
                               [valid_data[i] for i in sample]])
        outputs = model(torch.stack(images).to(device))
        _, predicted = outputs[0].max(1)

        plot_images(images, predicted, classes, true_labels=labels)

        images, labels = zip(*[(image + v.squeeze()*3, label) for image, label in
                               [valid_data[i] for i in sample]])
        outputs = model(torch.stack(images).to(device))
        _, predicted_perturbed = outputs[0].max(1)

        plot_images(images, predicted_perturbed, classes, true_labels=predicted)

        norms = np.linspace(0., 100., 25)

        random_v = np.random.rand(1, 3, 224, 224) - 0.5
        random_v = random_v.astype(np.float32)
        v_norm = np.linalg.norm(np.abs(v[0]))
        random_v_norm = np.linalg.norm(np.abs(random_v))

        dataset = torch.stack([valid_data[i][0] for i in range(100)])
        num_images = 100

        first_time = True
        est_labels_orig = np.zeros((num_images))

        num_batches = np.int(np.ceil(np.float(num_images) / np.float(args.batch_size)))

        for norm in tqdm(norms):

            normalized_v = v * (norm / v_norm)
            normalized_random_v = random_v * (norm / random_v_norm)
            #print(norm, np.linalg.norm(np.abs(normalized_v)), np.linalg.norm(np.abs(normalized_random_v)))

            # Perturb the dataset with computed perturbation
            dataset_perturbed = dataset + normalized_v[0]
            dataset_perturbed2 = dataset + normalized_random_v

            est_labels_pert = np.zeros((num_images))
            est_labels_pert2 = np.zeros((num_images))

            # Compute the estimated labels in batches
            for ii in range(0, num_batches):
                m = (ii * args.batch_size)
                M = min((ii + 1) * args.batch_size, num_images)
                if first_time:
                    est_labels_orig[m:M] = np.argmax(classifier(dataset[m:M, :, :, :]).detach().numpy(), axis=1).flatten()
                    first_time = False
                est_labels_pert[m:M] = np.argmax(classifier(dataset_perturbed[m:M, :, :, :]).detach().numpy(), axis=1).flatten()
                est_labels_pert2[m:M] = np.argmax(classifier(dataset_perturbed2[m:M, :, :, :]).detach().numpy(),
                                                 axis=1).flatten()
            fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
            fooling_rate2 = float(np.sum(est_labels_pert2 != est_labels_orig) / float(num_images))
            print(fooling_rate, fooling_rate2)

