import numpy as np
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


def explore_bsds300():
    bsds300 = datasets.BSDS300()
    bsds300.trn.x = np.hstack((bsds300.trn.x, bsds300.trn.x[:, [-1]]))  # bsds300, need to add one pixel
    util.pairplot(bsds300.trn.x, title="BSDS300 Train", color=bsds300.color)
    util.imageplot(bsds300.trn.x[:15], bsds300.image_size, layout=(3, 5))
    plt.show()


def explore_cifar10():
    cifar10 = datasets.CIFAR10()
    util.pairplot(cifar10.trn.x, title="CIFAR10 Train", color=cifar10.color)
    util.imageplot(cifar10.trn.x[:15], cifar10.image_size, layout=(3, 5))
    plt.show()


def explore_gas():
    gas = datasets.GAS()
    util.pairplot(gas.trn.x, title="GAS Train", color=gas.color)
    plt.show()


def explore_hepmass():
    hepmass = datasets.HEPMASS()
    util.pairplot(hepmass.trn.x, title="HEPMASS Train", color=hepmass.color)
    plt.show()


def explore_miniboone():
    miniboone = datasets.MINIBOONE()
    util.pairplot(miniboone.trn.x, title="MINIBOONE Train", color=miniboone.color)
    plt.show()


def explore_mnist():
    mnist = datasets.MNIST()
    util.pairplot(mnist.trn.x, title="MNIST Train", color=mnist.color)
    util.imageplot(mnist.trn.x[:15], mnist.image_size, layout=(3, 5))
    plt.show()


def explore_power():
    power = datasets.POWER()
    util.pairplot(power.trn.x, title="POWER Train", color=power.color)
    plt.show()


if __name__ == "__main__":
    explore_bsds300()
    explore_cifar10()
    explore_gas()
    explore_hepmass()
    explore_miniboone()
    explore_mnist()
    explore_power()
