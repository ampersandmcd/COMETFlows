import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import util


class ARTIFICIAL:

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        trn, val, tst = load_data_normalised()

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]
        self.color = "#ff0000"
        self.image_size = None


def load_data_normalised():

    # generate four subsets of data, then merge
    n = 250_000     # actual n is 4x this number
    p = 0.05        # probability of belonging to tail
    rng = np.random.default_rng(seed=1)

    # # back-project tails of particular dimensions to GPD space
    mu, sigma, xi = 0, 1, 0.1
    gpd = scipy.stats.genpareto(loc=mu, scale=sigma, c=-xi)

    # generate x1, x2 with UTD
    line = rng.random((int(p*n), 1))
    noise = rng.random((int((1-p)*n), 2))
    tail = 1 + gpd.ppf(np.hstack((line, line)))
    x1x2 = np.vstack((tail, noise))
    rng.shuffle(x1x2)
    # plt.scatter(x1x2[:, 0], x1x2[:, 1])
    # plt.show()

    # generate x3, x4 with LTD
    line = rng.random((int(p*n), 1))
    noise = rng.random((int((1-p)*n), 2))
    tail = -gpd.ppf(1-np.hstack((line, line)))
    x3x4 = np.vstack((noise, tail))
    rng.shuffle(x3x4)
    # plt.scatter(x3x4[:, 0], x3x4[:, 1])
    # plt.show()

    # generate x5, x6 with BTD
    lower_line = rng.random((int(p*n), 1))
    noise = rng.random((int((1-2*p)*n), 2))
    upper_line = rng.random((int(p*n), 1))
    lower_tail = -gpd.ppf(1-np.hstack((lower_line, lower_line)))
    upper_tail = 1 + gpd.ppf(np.hstack((upper_line, upper_line)))
    x5x6 = np.vstack((lower_tail, noise, upper_tail))
    rng.shuffle(x5x6)
    # plt.scatter(x5x6[:, 0], x5x6[:, 1])
    # plt.show()

    # generate x5, x6 with FTD
    lower_line = rng.random((int(p*n), 1))
    middle_line = rng.random((int((1-2*p)*n), 1))
    upper_line = rng.random((int(p*n), 1))
    lower_tail = -gpd.ppf(1-np.hstack((lower_line, lower_line)))
    upper_tail = 1 + gpd.ppf(np.hstack((upper_line, upper_line)))
    middle = np.hstack((middle_line, middle_line))
    x7x8 = np.vstack((lower_tail, middle, upper_tail))
    rng.shuffle(x7x8)
    # plt.scatter(x7x8[:, 0], x7x8[:, 1])
    # plt.show()

    # merge dimensions together
    x = np.hstack((x1x2, x3x4, x5x6, x7x8)).astype(np.float32)
    rng.shuffle(x)

    data_train = x[:int(0.8 * n)]
    data_validate = x[int(0.8 * n):int(0.9 * n)]
    data_test = x[int(0.9 * n):]

    return data_train, data_validate, data_test


