import numpy as np
import torch


def experiment2(perturbation, dataset, f, batch_size, device):
    with torch.no_grad():
        perturbed_dataset = dataset + torch.from_numpy(perturbation)
        num_images = len(dataset)
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))
        est_labels_orig = np.zeros(num_images)
        est_labels_pert = np.zeros(num_images)
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii + 1) * batch_size, num_images)
            if device == 'cuda':
                torch.cuda.empty_cache()
            est_labels_orig[m:M] = \
                np.argmax(f(dataset[m:M, :, :, :].to(device)).detach().cpu().numpy(), axis=1).flatten()
            est_labels_pert[m:M] = \
                np.argmax(f(perturbed_dataset[m:M, :, :, :].to(device)).detach().cpu().numpy(), axis=1).flatten()
        np.save("exp2.npy", np.dstack((est_labels_orig, est_labels_pert)))
