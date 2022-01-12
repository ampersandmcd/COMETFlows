import matplotlib.pyplot as plt

import datasets
import util


def explore_all():

    d = {
        "artificial": None,
        "bsds300": datasets.BSDS300(),
        "cifar10": datasets.CIFAR10(),
        "climdex": None,
        "gas": datasets.GAS(),
        "hepmass": datasets.HEPMASS(),
        "miniboone": datasets.MINIBOONE(),
        "mnist": datasets.MNIST(),
        "power": datasets.POWER()
    }

    for name, data in d.items():
        if data:
            print(f"{name}:\n"
                  f"\td = {data.trn.x.shape[1]}\n"
                  f"\ttrain_n = {data.trn.x.shape[0]}\n"
                  f"\tval_n = {data.val.x.shape[0]}\n"
                  f"\ttst_n = {data.tst.x.shape[0]}")


def explore_power():
    power = datasets.POWER()
    util.pairplot(power.trn.x, "Power Train X")
    plt.show()


if __name__ == "__main__":
    explore_all()
