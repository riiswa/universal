import os
import gc

import numpy as np
import scipy.io
from tqdm import tqdm

from universal.data_loading import download_training_data, get_trainval_dataset, VOCDataset, test_transforms, \
    train_transforms, classes, universal_transforms, normalize
from universal.experiment1 import experiment1
from universal.experiment2 import experiment2
from universal.universal_pert import universal_perturbation
from universal.vgg11_model import *
import logging
import torch
import argparse

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
    parser.add_argument('-f', '--force-download', action='store_true', default=False)
    parser.add_argument('-d', '--device', type=str, help='', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('-p', '--perturbation', type=str, help='', default=None)
    parser.add_argument('-o', '--output', type=str, help='', default=None)
    parser.add_argument('-b', '--batch-size', type=int, help='', default=25)
    parser.add_argument('--image-per-class', type=int, help='', default=25)
    parser.add_argument('--max-iter', type=int, help='', default=np.inf)
    parser.add_argument('--exp1', action='store_true', default=False)
    parser.add_argument('--exp2', action='store_true', default=False)
    args = parser.parse_args()
    logging.debug(args)

    vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)
    OUTPUT_DIM = 20
    device = args.device
    model = VGG(vgg11_layers, OUTPUT_DIM, device)
    logging.info(f"Device = {device}")

    logging.info(f"Prepare data. Train/validation data: {args.train_data}")
    os.makedirs('.data/', exist_ok=True)
    download_training_data(args.train_data, args.force_download)
    train_data_df, valid_data_df = get_trainval_dataset(args.train_data)

    train_data = VOCDataset(train_data_df, args.train_data, train_transforms)
    valid_data = VOCDataset(valid_data_df, args.train_data, test_transforms)

    if args.state_dict:
        logging.info(f"State dict {args.state_dict} found. Load pretrained weights.")
        model.load_state_dict(torch.load(args.state_dict))
        model.eval()
    else:
        logging.info("State dict not found (--s). "
                     "Please see the universal-voc-dataset.ipynb notebook to train the algorithm.")
        exit()
    del train_data

    model.to(device)

    def classifier(img):
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = normalize(img)
        v = model(img)[0]
        return v

    gc.collect()

    file_perturbation = os.path.join('.data', 'universal.npy')

    valid_data.transform = universal_transforms

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
            torch.stack(imgs).to(device),
            classifier,
            num_classes=len(classes),
            xi=2000/255,
            p=2,
            max_iter_uni=args.max_iter,
            device=device
        )

    if args.output:
        np.save(args.output, v)

    if args.exp1:
        dataset = torch.stack([valid_data[i][0] for i in range(len(valid_data))])
        experiment1(
            np.linspace(0., 10000, 20), [v[0],
                                         np.random.uniform(
                                             low=v.min(),
                                             high=v.max(),
                                             size=(1, 3, 224, 224),
                                         ).astype(np.float32),
                                         dataset.mean(dim=0).unsqueeze(0).numpy(),
                                         np.expand_dims(
                                             scipy.io.loadmat('precomputed/VGG-16.mat')['r'].transpose(2, 0, 1),
                                             0
                                         ),
                                         np.expand_dims(
                                             scipy.io.loadmat('precomputed/VGG-19.mat')['r'].transpose(2, 0, 1),
                                             0
                                         )
                                         ],
            dataset,
            classifier,
            args.batch_size,
            device
        )

    if args.exp2:
        dataset = torch.stack([valid_data[i][0] for i in range(len(valid_data))])
        experiment2(v[0], dataset, classifier, args.batch_size, device)
