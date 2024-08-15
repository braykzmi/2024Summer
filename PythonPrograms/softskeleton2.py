import torch
import torch.nn.functional as F
import nibabel as nib

# softskeleton algo from clDice github
class SoftSkeletonize(torch.nn.Module):
    def __init__(self, k=40):
        super(SoftSkeletonize, self).__init__()
        self.k = k

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        # erode = minpool
        # dilate = maxpool
        return self.soft_dilate(img)
        """
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.k):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel
        """

    def forward(self, img):
        return self.soft_skel(img)

# adapt to nifti
def softskeletonizenifti(input_file, output_file, k=20):
    nii_img = nib.load(input_file)
    img_data = nii_img.get_fdata()
    img_tensor = torch.tensor(img_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 

    # use softskeletonize from paper
    model = SoftSkeletonize(k=k)
    with torch.no_grad():
        skel = model(img_tensor)

    skel = skel.squeeze(0).squeeze(0)

    # save to nifti
    skel_np = skel.numpy()
    new_nii = nib.Nifti1Image(skel_np, nii_img.affine, nii_img.header)
    nib.save(new_nii, output_file)

# use
input_file = '/rsrch3/ip/dtfuentes/github/biliaryduct/Processed/1229944/Ven.vessel.2.nii.gz'
output_file = '/rsrch3/ip/dtfuentes/github/biliaryduct/Processed/1229944/1maxpool.nii.gz'
softskeletonizenifti(input_file, output_file, k=20)
