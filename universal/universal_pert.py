import logging

import numpy as np
import torch

from universal.deepfool import deepfool
from tqdm import tqdm

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten()))
        # v = v / np.linalg.norm(v.flatten(1)) * xi
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(dataset, f, delta=0.2, max_iter_uni=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10, batch_size=25, device='cpu'):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)

    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).

    :param delta: controls the desired fooling rate (default = 80% fooling rate)

    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

    :param xi: controls the l_p magnitude of the perturbation (default = 10)

    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)

    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)

    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

    :param max_iter_df: maximum number of iterations for deepfool (default = 10)

    :return: the universal perturbation.
    """

    v = np.zeros(dataset[0].unsqueeze(0).shape, dtype=np.float32)

    fooling_rate = 0.0

    best_v = None
    best_fooling_rate = -1

    num_images = np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION

    first_run = True

    est_labels_orig = torch.zeros((num_images)).to(device)

    itr = 0
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        # np.random.shuffle(dataset)

        print ('Starting pass number ', itr)

        # Go through the data set and compute the perturbation increments sequentially
        for k in (pbar := tqdm(range(0, num_images))):
            cur_img = dataset[k:(k+1), :, :, :]
            if f(cur_img).argmax() == f(cur_img+torch.from_numpy(v[0]).to(device)).argmax():
                pbar.set_description(f'>> k = {k}, pass #{itr}, norm ={np.linalg.norm(v)}')

                # Compute adversarial perturbation
                dr,iter,_,_,_ = deepfool(cur_img + torch.from_numpy(v[0]).to(device), f, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df, device=device)
                dr = dr
                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    # Project on l_p ball
                    v = proj_lp(v, xi, p)
            if np.linalg.norm(v) == np.inf:
                raise ValueError('A very specific bad thing happened. v == inf')

        itr = itr + 1

        # Perturb the dataset with computed perturbation
        if device == 'cuda':
            torch.cuda.empty_cache()
        with torch.no_grad():
            dataset_perturbed = dataset + torch.from_numpy(v[0]).to(device)

            est_labels_pert = torch.zeros((num_images)).to(device)

            num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

            # Compute the estimated labels in batches
            for ii in range(0, num_batches):
                m = (ii * batch_size)
                M = min((ii+1)*batch_size, num_images)
                if first_run:
                    est_labels_orig[m:M] = torch.argmax(f(dataset[m:M, :, :, :]), dim=1).flatten()
                est_labels_pert[m:M] = torch.argmax(f(dataset_perturbed[m:M, :, :, :]), dim=1).flatten()
            first_run = False
            # Compute the fooling rate
            fooling_rate = float(torch.sum(est_labels_pert != est_labels_orig) / float(num_images))
            logging.info(f'FOOLING RATE = {fooling_rate}')
            if fooling_rate > best_fooling_rate:
                logging.info(f'Perturbation saved.')
                best_fooling_rate = fooling_rate
                best_v = v
                np.save('.data/_universal.npy', best_v)
            del dataset_perturbed
            del est_labels_pert
    return v
