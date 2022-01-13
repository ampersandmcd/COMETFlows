import pytorch_lightning as pl
import torch

import wandb
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import datasets
from util import NumpyDataset, VisualCallback
from models import VanillaFlow, HTSFlow, HTCFlow, TDFlow, COMETFlow


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", default="vanilla", type=str, help="Model to train",
                        choices=[
                            "vanilla",
                            "hts-1", "hts-2", "hts-4",
                            "htc-10", "htc-05", "htc-01",
                            "td",
                            "comet-10", "comet-05", "comet-01"
                        ])
    parser.add_argument("--data", default="artificial", type=str, help="Dataset to train on",
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
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size to train with")
    parser.add_argument("--hidden_ds", default=(64, 64, 64), type=tuple, help="Hidden dimensions in coupling NN")
    parser.add_argument("--n_samples", default=1000, type=tuple, help="Number of samples to generate")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--img_epochs", default=10, type=int, help="How often to log images and pairplots")
    args = parser.parse_args()

    # configure data
    data = None
    if args.data == "artificial":
        data = datasets.ARTIFICIAL()
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
    train_dataset = NumpyDataset(data.trn.x)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4)
    val_dataset = NumpyDataset(data.val.x)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    model = None
    d = data.trn.x.shape[1]
    if args.model == "vanilla":
        model = VanillaFlow(d, args.hidden_ds, args.lr)
    elif args.model[:3] == "hts":
        dof = int(args.model.split("-")[-1])
        model = HTSFlow(d, args.hidden_ds, args.lr, dof)
    elif args.model[:3] == "htc":
        raise NotImplementedError()
    elif args.model[:2] == "td":
        model = TDFlow(d, args.hidden_ds, args.lr)
    elif args.model[:5] == "comet":
        raise NotImplementedError()

    # wandb logging
    wandb_logger = pl.loggers.WandbLogger(project="comet-flows")
    wandb_logger.watch(model, log="all", log_freq=10)
    wandb_logger.experiment.config.update(args)

    # trainer configuration
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ModelCheckpoint(monitor="v_loss"))
    mins, maxs = data.trn.x.min(axis=0), data.trn.x.max(axis=0)
    trainer.callbacks.append(VisualCallback(n_samples=args.n_samples, color=data.color,
                                            mins=mins, maxs=maxs,
                                            image_size=data.image_size, img_every_n_epochs=args.img_epochs))

    trainer.fit(model, train_dataloader, val_dataloader)
