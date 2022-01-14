import numpy as np
import matplotlib.pyplot as plt

import datasets
import util


def explore_all():

    d = {
        "artificial": datasets.ARTIFICIAL(),
        "bsds300": datasets.BSDS300(),
        "cifar10": datasets.CIFAR10(),
        "climdex": datasets.CLIMDEX(),
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


def explore_artificial():
    artificial = datasets.ARTIFICIAL()
    mins, maxs = np.quantile(artificial.trn.x, 0.001, axis=0), np.quantile(artificial.trn.x, 0.999, axis=0)
    util.pairplot(artificial.trn.x, mins=mins, maxs=maxs, color=artificial.color)
    plt.show()


def explore_bsds300():
    bsds300 = datasets.BSDS300()
    bsds300.trn.x = np.hstack((bsds300.trn.x, bsds300.trn.x[:, [-1]]))  # bsds300, need to add one pixel
    mins, maxs = np.quantile(bsds300.trn.x, 0.001, axis=0), np.quantile(bsds300.trn.x, 0.999, axis=0)
    util.pairplot(bsds300.trn.x, mins=mins, maxs=maxs, color=bsds300.color)
    util.imageplot(bsds300.trn.x[:15], bsds300.image_size, layout=(3, 5))
    plt.show()


def explore_cifar10():
    cifar10 = datasets.CIFAR10()
    mins, maxs = np.quantile(cifar10.trn.x, 0.001, axis=0), np.quantile(cifar10.trn.x, 0.999, axis=0)
    util.pairplot(cifar10.trn.x, mins=mins, maxs=maxs, color=cifar10.color)
    util.imageplot(cifar10.trn.x[:15], cifar10.image_size, layout=(3, 5))
    plt.show()


def explore_climdex():
    climdex = datasets.CLIMDEX()
    mins, maxs = np.quantile(climdex.trn.x, 0.001, axis=0), np.quantile(climdex.trn.x, 0.999, axis=0)
    util.pairplot(climdex.trn.x, mins=mins, maxs=maxs, color=climdex.color)
    plt.show()


def explore_gas():
    gas = datasets.GAS()
    mins, maxs = np.quantile(gas.trn.x, 0.001, axis=0), np.quantile(gas.trn.x, 0.999, axis=0)
    util.pairplot(gas.trn.x, mins=mins, maxs=maxs, color=gas.color)
    plt.show()


def explore_hepmass():
    hepmass = datasets.HEPMASS()
    mins, maxs = np.quantile(hepmass.trn.x, 0.001, axis=0), np.quantile(hepmass.trn.x, 0.999, axis=0)
    util.pairplot(hepmass.trn.x, mins=mins, maxs=maxs, color=hepmass.color)
    plt.show()


def explore_miniboone():
    miniboone = datasets.MINIBOONE()
    mins, maxs = np.quantile(miniboone.trn.x, 0.001, axis=0), np.quantile(miniboone.trn.x, 0.999, axis=0)
    util.pairplot(miniboone.trn.x, mins=mins, maxs=maxs, color=miniboone.color)
    plt.show()


def explore_mnist():
    mnist = datasets.MNIST()
    mins, maxs = np.quantile(mnist.trn.x, 0.001, axis=0), np.quantile(mnist.trn.x, 0.999, axis=0)
    util.pairplot(mnist.trn.x, mins=mins, maxs=maxs, color=mnist.color)
    util.imageplot(mnist.trn.x[:15], mnist.image_size, layout=(3, 5))
    plt.show()


def explore_power():
    power = datasets.POWER()
    mins, maxs = np.quantile(power.trn.x, 0.001, axis=0), np.quantile(power.trn.x, 0.999, axis=0)
    util.pairplot(power.trn.x, mins=mins, maxs=maxs, color=power.color)
    plt.show()


if __name__ == "__main__":
    explore_artificial()
    explore_bsds300()
    explore_cifar10()
    explore_climdex()
    explore_gas()
    explore_hepmass()
    explore_miniboone()
    explore_mnist()
    explore_power()
