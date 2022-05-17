import copy

import numpy as np
import torch


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=10, device='cpu'):
    f_image = net(image).flatten()
    I = f_image.argsort(descending=True)

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.shape
    pert_image = image.clone()
    w = torch.zeros(input_shape).to(device)
    r_tot = torch.zeros(input_shape).to(device)

    loop_i = 0

    x = pert_image[None, :].clone().detach().requires_grad_(True)

    fs = net(x[0])
    fs_list = [fs[0, I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()

        for k in range(1, num_classes):

            # x.zero_grad()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.clone()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data

            pert_k = f_k.abs() / torch.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = ((pert + 1e-4) * w / torch.linalg.norm(w)).to(device)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1 + overshoot) * (torch.from_numpy(r_tot).to(device))

        x = pert_image.clone().detach().requires_grad_(True)
        fs = net(x[0])
        k_i = fs.flatten().argmax()

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

    return r_tot, loop_i, label, k_i, pert_image
