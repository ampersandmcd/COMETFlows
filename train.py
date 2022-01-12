import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray as xr
import pytorch_lightning as pl
import pytorch_lightning as pl
import wandb
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import datasets
import util
from util import VisualCallback\

wandb.init(project="comet-flows", entity="andrewmcdonald")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", default="vanilla", type=str, help="Model to train",
                        options=[
                            "vanilla",
                            "ht1-1", "ht1-2", "ht1-4",
                            "ht2-10", "ht2-05", "ht2-01",
                            "td",
                            "comet-10", "comet-05", "comet-01"
                        ])
    parser.add_argument("--data", default="artificial", type=str, help="Dataset to train on",
                        options=[
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
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size to train with")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    args = parser.parse_args()

    # configure data
    data = None
    if args.data == "artificial":
        raise NotImplementedError()
    elif args.data == "bsds300":
        data = datasets.BSDS300()
    elif args.data == "cifar10":
        data = datasets.CIFAR10()
    elif args.data == "climdex":
        raise NotImplementedError()
    elif args.data == "gas":
        data = datasets.GAS()
    elif args.data == "hepmass":
        data = datasets.HEPMASS()
    elif args.data == "miniboone":
        data = datasets.MINIBOONE()
    elif args.data == "mnist":
        data = datasets.MNIST()
    elif args.data == "power":
        data = datasets.POWER()

    # configure dataloaders
    train_dataloader = DataLoader(data.trn.x, batch_size=args.batch_size)
    val_dataloader = DataLoader(data.val.x, batch_size=args.batch_size)

    model = None
    if args.model == "vanilla":
        model = None    # TODO


    # Wandb logging
    wandb_logger = pl.loggers.WandbLogger(project="comet-flows")
    wandb_logger.watch(model, log_freq=500)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="val_loss"))
    trainer.callbacks.append(VisualCallback(n_samples=args.n_samples, color=data.color, image_size=data.image_size))

    trainer.fit(model, train_dataloader, val_dataloader)

