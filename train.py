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
    parser.add_argument("--name", default="default", type=str, help="Name of wandb run")
    parser.add_argument("--model", default="vanilla", type=str, help="Model to train",
                        choices=[
                            "vanilla",
                            "taf-16", "taf-32", "taf-64",
                            "cmf-10", "cmf-05", "cmf-01",
                            "softflow",
                            "comet-10", "comet-05", "comet-01"
                        ])
    parser.add_argument("--data", default="gas", type=str, help="Dataset to train on",
                        choices=[
                            "artificial",
                            "bsds300",
                            "cifar10",
                            "climdex",
                            "gas",
                            "hepmass",
                            "miniboone",
                            "mnist",
                            "power"
                        ])
    parser.add_argument("--batch_size", default=10_000, type=int, help="Batch size to train with")
    parser.add_argument("--hidden_ds", default=(64, 64, 64), type=tuple, help="Hidden dimensions in coupling NN")
    parser.add_argument("--n_samples", default=1000, type=tuple, help="Number of samples to generate")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--img_epochs", default=10, type=int, help="How often to log images and pairplots")
    args = parser.parse_args()

    # configure data
    data = None
    if args.data == "artificial":
        data = datasets.ARTIFICIAL()
        args.max_epochs = 500   # small dataset
    elif args.data == "bsds300":
        data = datasets.BSDS300()
        args.max_epochs = 100   # large dataset
    elif args.data == "cifar10":
        data = datasets.CIFAR10()
        args.max_epochs = 500   # small dataset
    elif args.data == "climdex":
        data = datasets.CLIMDEX()
        args.max_epochs = 500   # small dataset
    elif args.data == "gas":
        data = datasets.GAS()
        args.max_epochs = 100   # large dataset
    elif args.data == "hepmass":
        data = datasets.HEPMASS()
        args.max_epochs = 100   # large dataset
    elif args.data == "miniboone":
        data = datasets.MINIBOONE()
        args.max_epochs = 500   # small dataset
    elif args.data == "mnist":
        data = datasets.MNIST()
        args.max_epochs = 500   # small dataset
    elif args.data == "power":
        data = datasets.POWER()
        args.max_epochs = 100   # large dataset

    # configure dataloaders
    train_dataset = NumpyDataset(data.trn.x)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_dataset = NumpyDataset(data.val.x)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_dataset = NumpyDataset(data.tst.x)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # configure model
    model = None
    d = data.trn.x.shape[1]
    if args.model == "vanilla":
        model = VanillaFlow(d, args.hidden_ds, args.lr)
    elif args.model[:3] == "taf":
        dof = int(args.model.split("-")[-1])
        model = TAFlow(d, args.hidden_ds, args.lr, dof)
    elif args.model[:3] == "cmf":
        tail = float(args.model.split("-")[-1]) / 100
        a, b = tail, 1 - tail
        model = CMFlow(d, args.hidden_ds, args.lr, data.trn.x, a, b)
    elif args.model == "softflow":
        model = SoftFlow(d, args.hidden_ds, args.lr)
        args.max_epochs = args.max_epochs * 2   # conditional noise takes longer to converge
    elif args.model[:5] == "comet":
        tail = float(args.model.split("-")[-1]) / 100
        a, b = tail, 1 - tail
        model = COMETFlow(d, args.hidden_ds, args.lr, data.trn.x, a, b)
        args.max_epochs = args.max_epochs * 2   # conditional noise takes longer to converge

    # wandb logging
    wandb.init(project="comet-flows")
    if args.name != "default":
        wandb.run.name = args.name
    wandb_logger = pl.loggers.WandbLogger(project="comet-flows")
    wandb_logger.watch(model, log="all", log_freq=10)
    wandb_logger.experiment.config.update(args)

    # trainer configuration
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="v_loss"))
    mins, maxs = np.quantile(data.trn.x, 0.001, axis=0), np.quantile(data.trn.x, 0.999, axis=0)
    trainer.callbacks.append(VisualCallback(n_samples=args.n_samples, color=data.color,
                                            mins=mins, maxs=maxs,
                                            image_size=data.image_size, img_every_n_epochs=args.img_epochs))

    # train
    trainer.fit(model, train_dataloader, val_dataloader)

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
    x = model.sample(args.n_samples).detach().cpu().numpy()
    util.pairplot(x, title=None, color=data.color)
    wandb.log({"final_pairplot": wandb.Image(plt)})
    plt.close()
