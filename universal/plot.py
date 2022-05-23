import numpy as np
import matplotlib.pyplot as plt


def normalize_image(image):
    """ Normalize image for plotting.

    :param image: Tensor image.
    :return: Normalized image.
    """
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def plot_images(images, labels, classes, true_labels=None, normalize=True):
    """ Plot images.

    :param images: Images to plot.
    :param labels: Labels of images to plot in the title.
    :param classes: Corresponding classes string.
    :param true_labels: True labels of the images. Can be Nine.
    :param normalize: Images should be normalized ?
    :return: The matplotlib figure.
    """

    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(10, 10))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        if not true_labels is None:
            ax.set_title(f"{classes[true_labels[i]]} → {classes[labels[i]]}",
                         color='green' if labels[i] == true_labels[i] else 'red')
        else:
            ax.set_title(classes[labels[i]])
        ax.axis('off')
    return fig
