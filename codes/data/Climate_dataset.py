import random
import numpy as np
from glob import glob
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F


def upscale(feat, scale_factor: int = 2):
    # resolution decrease
    if scale_factor == 1:
        return feat
    else:
        return F.avg_pool2d(feat, scale_factor)

def normalize(img, entire_mean: list, entire_std: list, inverse: bool = False):
    """Using given mean and std to normalize images.
    If inverse is True, do the inverse process.

    Args:
        images: NCHW or CHW
    Return:
        images
    """
    images = img.clone()
    def normalize_standard(image, mean, std):
        if isinstance(image, torch.Tensor):
            return torch.divide(
                torch.add(image, -torch.tensor(mean)),
                torch.maximum(torch.tensor(std), torch.tensor(1e-5)),
            )
        else:
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            return (image - mean) / max(std, 1e-5)

    def inverse_normalize_standard(image, mean, std):
        if isinstance(image, torch.Tensor):
            return torch.add(
                torch.multiply(
                    image, torch.maximum(torch.tensor(std), torch.tensor(1e-5))
                ),
                torch.tensor(mean),
            )
        else:
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)
            return image * max(std, 1e-5) + mean

    if images.dim() == 3:
        c, _, _ = images.shape
        for j in range(c):
            if inverse:
                images[j] = inverse_normalize_standard(
                    images[j], entire_mean[j], entire_std[j]
                )
            else:
                images[j] = normalize_standard(images[j], entire_mean[j], entire_std[j])
    elif images.dim() == 4:
        n, c, _, _ = images.shape
        for y in range(n):
            for j in range(c):
                if inverse:
                    images[y][j] = inverse_normalize_standard(
                        images[y][j], entire_mean[j], entire_std[j]
                    )
                else:
                    images[y][j] = normalize_standard(
                        images[y][j], entire_mean[j], entire_std[j]
                    )
    return images


def augment(img, hflip=True, rot=True, swap=None):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        img = img.permute(1,2,0) # CHW to HWC
        if hflip:
            # img = img[:, ::-1, :]
            img = img.flip(1)
        if vflip:
            # img = img[::-1, :, :]
            img = img.flip(0)
        if rot90:
            # img = img.transpose(1, 0, 2)
            img = img.permute(1, 0, 2)
        img = img.permute(2,0,1) # HWC to CHW
        return img

    if swap and random.random() < 0.5:
        img.reverse()
    return [_augment(I) for I in img]


class Climate_dataset(data.Dataset):
    def __init__(self, opt):
        """
        Args:
            image_type: 'Solar', 'Wind'
        """
        self.opt = opt
        image_type = self.opt['image_type']
        dataset_device = self.opt['dataset_device']
        if image_type == 'Solar':
            years = ['07', '08', '09', '10', '11', '12', '13']
            if dataset_device == 'HPCC':
                path_train = [
                    '/lustre/scratch/guiyli/Dataset_NSRDB/npyFiles/dni_dhi/20' + i
                    for i in years
                ]
                path_test = '/lustre/scratch/guiyli/Dataset_NSRDB/npyFiles/dni_dhi/2014'
            elif dataset_device == 'PC':
                path_train = None
                path_test = '/home/guiyli/Documents/DataSet/Solar/npyFiles/dni_dhi/2014'
                print('No training data set on current device')
            entire_mean = '392.8659294288083,125.10559238383577'
            entire_std = '351.102247720423,101.6698946847449'
        elif image_type == 'Wind':
            years = ['07', '08', '09', '10', '11', '12', '13']
            if dataset_device == 'HPCC':
                path_train = [
                    '/lustre/scratch/guiyli/Dataset_WIND/npyFiles/20' + i + '/u_v' for i in years
                ]
                path_test = '/lustre/scratch/guiyli/Dataset_WIND/npyFiles/2014/u_v'
            elif dataset_device == 'PC':
                path_train = [
                    '/home/guiyli/Documents/DataSet/Wind/20' + i + '/u_v' for i in years
                ]
                path_test = '/home/guiyli/Documents/DataSet/Wind/2014/u_v'
            entire_mean = '-0.6741845839785552,-1.073033474161022'
            entire_std = '5.720375778518578,4.772050058088903'

        self.entire_mean = [float(i) for i in entire_mean.split(',')]
        self.entire_std = [float(i) for i in entire_std.split(',')]

        self.scale_factor = self.opt["scale"] if self.opt["scale"] else 1
        self.root = path_train if self.opt["phase"] == "train" else path_test

        self.files = []
        if type(self.root) == list:
            for i in self.root:
                self.files += glob(i + '/*.npy')
            assert self.files is not None, 'No data found.'
            self.suffix = 'npy'
        else:
            self.files = glob(self.root + '/*.tif')
            self.suffix = 'tif'
            if not self.files:  # in case images are in npy format
                self.files = glob(self.root + '/*.npy')
                assert self.files is not None, 'No data found.'
                self.suffix = 'npy'

        self.resize2tensor = transforms.Compose(
            [
                transforms.ToTensor(),  # convert from HWC to CHW
                transforms.Resize(
                    (8*self.scale_factor, 8*self.scale_factor),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
            ]
        )

        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if isinstance(index, str):
            f = glob(self.root + '/' + index + '.' + self.suffix)[0]
        else:
            f = self.files[index]

        if self.suffix == 'tif':
            img_GT = Image.open(f).astype(np.float32)
        else:
            img_GT = np.load(f).astype(np.float32)

        img_GT = self.resize2tensor(img_GT) # CHW
        img_GT = normalize(img_GT, self.entire_mean, self.entire_std)

        img_LR = upscale(img_GT, self.scale_factor)

        if self.opt["phase"] == "train":
            img_LR, img_GT = augment(
                [img_LR, img_GT],
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["use_swap"],
            )
        return {"LQ": img_LR, "GT": img_GT}