import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import options as option

from models import create_model
from train_climate import normalize

import sys
sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset


def sliced_wasserstein_cuda(A, B, dir_repeats=4, dirs_per_repeat=128):
    """
    A, B: dreal, dfake(after normalize: -mean/std [0,1])

    Reference:
        https://github.com/tkarras/progressive_growing_of_gans
    """
    assert A.ndim == 2 and A.shape == B.shape                                   # (neighborhood, descriptor_component)
    device = torch.device("cuda")
    results = torch.empty(dir_repeats, device=torch.device("cpu"))
    A = torch.from_numpy(A).to(device) if not isinstance(A, torch.Tensor) else A.to(device)
    B = torch.from_numpy(B).to(device) if not isinstance(B, torch.Tensor) else B.to(device)
    for repeat in range(dir_repeats):
        dirs = torch.randn(A.shape[1], dirs_per_repeat, device=device)          # (descriptor_component, direction)
        dirs = torch.divide(dirs, torch.sqrt(torch.sum(torch.square(dirs), dim=0, keepdim=True)))  # normalize descriptor components for each direction
        projA = torch.matmul(A, dirs)                                           # (neighborhood, direction)
        projB = torch.matmul(B, dirs)
        projA = torch.sort(projA, dim=0)[0]                                     # sort neighborhood projections for each direction
        projB = torch.sort(projB, dim=0)[0]
        dists = torch.abs(projA - projB)                                        # pointwise wasserstein distances
        results[repeat] = torch.mean(dists)                                     # average over neighborhoods and directions
    return torch.mean(results)                                                  # average over repeats


def normalize_standard(image):
    """
    Standard Score Normalization

    (image - mean) / std

    return: data_new, mean, std
    """
    if isinstance(image, torch.Tensor):
        mean = torch.mean(image)
        std = torch.std(image)
        return (
            torch.divide(
                torch.add(image, -mean), torch.maximum(std, torch.tensor(1e-5))
            ),
            mean,
            std,
        )
    else:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        mean = np.mean(image)
        std = np.std(image)
        return (image - mean) / max(std, 1e-5), mean, std


def eval_metrics(opt):
    #### Create test dataset and dataloader
    dataset_opt = opt["datasets"]["val"]
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)

    # load pretrained model by default
    model = create_model(opt)

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=model.device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    # calc metrics
    metrics_mse = torch.empty(len(test_loader), dtype=torch.float64)
    metrics_swd = torch.empty((2, len(test_loader)), dtype=torch.float64)
    metrics_min, metrics_max = 0, 0

    for i, test_data in enumerate(tqdm(test_loader)):

        LQ, GT = test_data["LQ"], test_data["GT"]
        LQ = util.upscale(LQ, scale)
        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, GT)
        model.test(sde, save_states=True)

        visuals = model.get_current_visuals()

        img_fake = normalize(visuals["Output"].squeeze(), test_set.entire_mean, test_set.entire_std, inverse=True)
        img_gt = normalize(GT.squeeze(), test_set.entire_mean, test_set.entire_std, inverse=True)

        metrics_min = metrics_min if img_gt.min() > metrics_min else img_gt.min()
        metrics_max = metrics_max if img_gt.max() < metrics_max else img_gt.max()

        # relative MSE
        metrics_mse[i] = ((img_gt - img_fake)**2).mean() / (img_gt.mean()**2)

        # SWD
        for c in range(2):
            # normalize before calc SWD
            img_gt_n = normalize_standard(img_gt[c])[0]
            img_fake_n = normalize_standard(img_fake[c])[0]
            metrics_swd[c][i] = sliced_wasserstein_cuda(img_gt_n, img_fake_n)


    save_folder = os.path.join(opt["path"]["results_root"], "eval_metrics")
    os.makedirs(save_folder, exist_ok=True)
    np.save(
        os.path.join(save_folder, "error_mse_"+str(scale)+"X.npy"),
        metrics_mse.numpy()
    )
    np.save(
        os.path.join(save_folder, "error_swd_"+str(scale)+"X.npy"),
        torch.mean(metrics_swd, 0).numpy()
    )

    text_file = open(
        os.path.join(save_folder, "mean_metrics_mse_swd_"+str(scale)+"X.txt"),
        "w",
    )

    drange = metrics_max - metrics_min
    text_file.write("\n" + "Data Range: " + str(drange) + "\n")
    text_file.write(str(metrics_min) + ", " + str(metrics_max) + "\n")

    text_file.write("\n" + "MSE/(mean^2) --> mean" + "\n")
    text_file.write(str(torch.mean(metrics_mse).numpy()) + "\n")
    text_file.write("\n" + "MSE/(mean^2) --> median" + "\n")
    text_file.write(str(torch.median(metrics_mse).numpy()) + "\n")

    text_file.write("\n" + "SWD/(mean^2) --> mean" + "\n")
    text_file.write(str(torch.mean(metrics_swd).numpy()) + "\n")
    text_file.write("\n" + "SWD/(mean^2) --> median" + "\n")
    text_file.write(str(torch.median(metrics_swd).numpy()) + "\n")
    print("Validation metrics saved!")
    text_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="options/test/setting_Solar_8X.yml", help="Path to options YMAL file.")
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    eval_metrics(opt)
