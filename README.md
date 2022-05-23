# Universal adversarial perturbations

This repository contains the code to calculate a universal adversarial perturbations (UAP) on the VGG-11 architecture and the VOC2012 dataset.  Implemented with Pytorch.

## Requirements

The code require at least Python 3.8 to run. The dependencies can be installed with `pip install -r requirements.txt`.

## Main file/folder list

- `precomputed`: Precomputed UAP, from original repo, except for VGG-11.npy which was calculated from 1000 images with a fooling rate of 55%.
- `main.py`: Main code to run experiments.
- `model_training.ipynb`: Train VGG-11 classifier on VOC2012 data. Inspired from https://github.com/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb
- `visualization.ipynb`: Some visualizations related to computed perturbation.
- `universal/deepfool.py` and `universal/universal_pert.py` are adapted from the original repo for Pytorch.

## Help

```bash
python main.py -h
```

## Perturbation example with pretrained VGG-11

```bash
python main.py -s vg11-voc2012-model.pt -d 'cpu' -p precomputed/VGG-11.npy --exp1
```

## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017
