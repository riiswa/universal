import numpy as np
import torch
from tqdm import tqdm


def experiment1(norms, perturbations, dataset, f, batch_size, device):
    with torch.no_grad():
        perturbations_norm = [np.linalg.norm(p) for p in perturbations]
        print(perturbations_norm)
        num_images = len(dataset)
        num_perts = len(perturbations)
        first_time = True
        est_labels_orig = np.zeros(num_images)

        fooling_rates = [[]] * len(perturbations)

        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

        for norm in (pbar := tqdm(norms)):
            norm = norm / 255
            normalized_perturbations = [perturbations[i] * (norm / perturbations_norm[i]) for i in range(num_perts)]

            perturbed_datasets = [dataset + torch.from_numpy(normalized_perturbations[i]) for i in range(num_perts)]

            est_labels_perts = [np.zeros(num_images) for _ in range(num_perts)]

            # Compute the estimated labels in batches
            for ii in range(0, num_batches):
                if device == 'cuda':
                    torch.cuda.empty_cache()
                m = (ii * batch_size)
                M = min((ii + 1) * batch_size, num_images)
                if first_time:
                    est_labels_orig[m:M] = \
                        np.argmax(f(dataset[m:M, :, :, :].to(device)).detach().cpu().numpy(), axis=1).flatten()
                for i in range(num_perts):
                    est_labels_perts[i][m:M] = \
                        np.argmax(f(perturbed_datasets[i][m:M, :, :, :].to(device)).detach().cpu().numpy(), axis=1) \
                            .flatten()
                first_time = False
            for i in range(num_perts):
                fooling_rates[i].append(float(np.sum(est_labels_perts[i] != est_labels_orig) / float(num_images)))
            pbar.set_description(str([fr[-1] for fr in fooling_rates]))
        np.save("exp1.npy", np.array(fooling_rates))

