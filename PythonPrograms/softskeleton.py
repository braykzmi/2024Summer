import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

image = nib.load('/rsrch3/ip/dtfuentes/github/biliaryduct/Processed/1229944/Ven.vessel.2.nii.gz')
I = image.get_fdata()
I = torch.tensor(I, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

kernel_size = 3
stride = 1
padding = kernel_size//2

maxpool_layer = nn.MaxPool3d(kernel_size=kernel_size, stride = stride, padding = padding)

def minpool(image, kernel_size = 3):
    return -maxpool_layer(image)


def maxpool(image, kernel_size = 3):
    return maxpool_layer(image)

def soft_skel_algo(I, k):
    I0 = maxpool(minpool(I))
    S = F.relu(I-I0)

    for _ in range(k):
        I = minpool(I)
        I0 = maxpool(minpool(I))
        S = S + (1-S) * F.relu(I-I0)

    return S

k = 5 #adjust k based on data

S = soft_skel_algo(I, k).squeeze(0).squeeze(0)

S_np = S.numpy()
newfile = nib.Nifti1Image(S_np, image.affine, image.header)
nib.save(newfile, '/rsrch3/ip/dtfuentes/github/biliaryduct/Processed/1229944/softskeleton.nii.gz')