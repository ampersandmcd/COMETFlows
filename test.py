import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import wandb
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import datasets
import util
from util import NumpyDataset, VisualCallback
from models import VanillaFlow, TAFlow, CMFlow, SoftFlow, COMETFlow


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--name", default="10j8c0u1", type=str, help="Name of wandb run")
    args = parser.parse_args()

    # wandb logging
    wandb.init(id=args.name, project="comet-flows", resume="must")
    api = wandb.Api()
    run = api.run(f"andrewmcdonald/comet-flows/{args.name}")

    # configure data
    data = None
    if run.config["data"] == "artificial":
        data = datasets.ARTIFICIAL()
    elif run.config["data"] == "bsds300":
        data = datasets.BSDS300()
    elif run.config["data"] == "cifar10":
        data = datasets.CIFAR10()
    elif run.config["data"] == "climdex":
        data = datasets.CLIMDEX()
    elif run.config["data"] == "gas":
        data = datasets.GAS()
    elif run.config["data"] == "hepmass":
        data = datasets.HEPMASS()
    elif run.config["data"] == "miniboone":
        data = datasets.MINIBOONE()
    elif run.config["data"] == "mnist":
        data = datasets.MNIST()
    elif run.config["data"] == "power":
        data = datasets.POWER()

    # configure dataloaders
    test_dataset = NumpyDataset(data.tst.x)
    test_dataloader = DataLoader(test_dataset, batch_size=run.config["batch_size"], num_workers=4)

    # load model from best checkpoint
    model = None
    checkpoint_folder = f"./comet-flows/{args.name}/checkpoints"
    checkpoints = os.listdir(checkpoint_folder)
    best_epoch = 0
    best_checkpoint = None
    for checkpoint in checkpoints:
        epoch = int(checkpoint.split("=")[1].split("-")[0])
        if epoch > best_epoch:
            best_epoch = epoch
            best_checkpoint = f"{checkpoint_folder}/{checkpoint}"

    if run.config["model"] == "vanilla":
        model = VanillaFlow.load_from_checkpoint(best_checkpoint)
    elif run.config["model"][:3] == "taf":
        dof = int(run.config["model"].split("-")[-1])
        model = TAFlow.load_from_checkpoint(best_checkpoint, dof=dof)
    elif run.config["model"][:3] == "cmf":
        tail = float(run.config["model"].split("-")[-1]) / 100
        a, b = tail, 1 - tail
        model = CMFlow.load_from_checkpoint(best_checkpoint, data=data.trn.x, a=a, b=b)
    elif run.config["model"] == "softflow":
        model = SoftFlow.load_from_checkpoint(best_checkpoint)
    elif run.config["model"][:5] == "comet":
        tail = float(run.config["model"].split("-")[-1]) / 100
        a, b = tail, 1 - tail
        model = COMETFlow.load_from_checkpoint(best_checkpoint, data=data.trn.x, a=a, b=b)

    # test
    model.eval()
    nlls, nans, infs = [], [], []
    for batch in test_dataloader:
        x = batch["x"].type(torch.FloatTensor).to(model.device)
        criterion = model.get_criterion()

        # perform forward pass
        if model.conditional_noise:
            std_min, std_max = 1e-3, 5e-3  # set to near-zero noise for validation
            noise = (std_max - std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + std_min
            noisy_x = x + torch.randn_like(x) * noise
            z, delta_logp = model.forward(noisy_x, noise_level=noise)
        else:
            z, delta_logp = model.forward(x)

        # check nans and infs
        nan_idx = torch.all(torch.logical_or(z != z, delta_logp != delta_logp), dim=1)
        inf_idx = torch.all(torch.logical_not(torch.logical_and(torch.isfinite(z), torch.isfinite(delta_logp))), dim=1)
        keep_idx = torch.logical_not(torch.logical_or(nan_idx, inf_idx))
        z, delta_logp = z[keep_idx], delta_logp[keep_idx]

        # compute and save nll for mean computation
        nll = criterion(z, delta_logp)
        nlls.append(nll)

        # save nan/inf count for later logging
        nans.append(sum(nan_idx))
        infs.append(sum(inf_idx))

    # save and log test nll
    test_nll = sum(nlls) / len(test_dataset)
    wandb.log({"test_nll": test_nll})

    # save and log total number of nan/inf
    wandb.log({"test_nan": sum(nans)})
    wandb.log({"test_inf": sum(infs)})

    # save and log image of samples
    run.config["n_test_samples"] = 10_000
    x = model.sample(run.config["n_test_samples"]).detach().cpu().numpy()
    util.pairplot(x, title=None, color=data.color)
    wandb.log({"final_pairplot": wandb.Image(plt)})
    plt.close()
