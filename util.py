import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
import wandb
from pytorch_lightning import Callback
from torch.utils.data import Dataset


import datasets


def isposint(n):
    """
    Determines whether number n is a positive integer.
    :param n: number
    :return: bool
    """
    return isinstance(n, int) and n > 0


def logistic(x):
    """
    Elementwise logistic sigmoid.
    :param x: numpy array
    :return: numpy array
    """
    return 1.0 / (1.0 + np.exp(-x))


def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))


def imageplot(xs, image_size, layout=(1,1)):
    """
    Displays an array of images, a page at a time. The user can navigate pages with
    left and right arrows, start over by pressing space, or close the figure by esc.
    :param xs: an numpy array with images as rows
    :param image_size: size of the images
    :param layout: layout of images in a page
    :return: none
    """

    num_plots = np.prod(layout)
    num_xs = xs.shape[0]
    idx = [0]

    # create a figure with suplots
    fig, axs = plt.subplots(layout[0], layout[1])

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ax in axs:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    def plot_page():
        """Plots the next page."""

        ii = np.arange(idx[0], idx[0]+num_plots) % num_xs

        for ax, i in zip(axs, ii):
            if len(image_size) > 2:
                img = xs[i].reshape(image_size).transpose(1, 2, 0)
            else:
                img = xs[i].reshape(image_size)
            ax.imshow(img, interpolation='none')
            ax.set_title(str(i))

        fig.canvas.draw()

    def on_key_event(event):
        """Event handler after key press."""

        key = event.key

        if key == 'right':
            # show next page
            idx[0] = (idx[0] + num_plots) % num_xs
            plot_page()

        elif key == 'left':
            # show previous page
            idx[0] = (idx[0] - num_plots) % num_xs
            plot_page()

        elif key == ' ':
            # show first page
            idx[0] = 0
            plot_page()

        elif key == 'escape':
            # close figure
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key_event)
    plot_page()


def isdistribution(p):
    """
    :param p: a vector representing a discrete probability distribution
    :return: True if p is a valid probability distribution
    """
    return np.all(p >= 0.0) and np.isclose(np.sum(p), 1.0)


def discrete_sample(p, n_samples=1):
    """
    Samples from a discrete distribution.
    :param p: a distribution with N elements
    :param n_samples: number of samples
    :return: vector of samples
    """

    # check distribution
    #assert isdistribution(p), 'Probabilities must be non-negative and sum to one.'

    # cumulative distribution
    c = np.cumsum(p[:-1])[np.newaxis, :]

    # get the samples
    rng = np.random.default_rng(seed=1)
    r = rng.rand(n_samples, 1)
    return np.sum((r > c).astype(int), axis=1)


def ess_importance(ws):
    """
    Calculates the effective sample size of a set of weighted independent samples (e.g. as given by importance
    sampling or sequential monte carlo). Takes as input the normalized sample weights.
    """

    ess = 1.0 / np.sum(ws ** 2)
    return ess


def ess_mcmc(xs):
    """
    Calculates the effective sample size of a correlated sequence of samples, e.g. as given by markov chain monte
    carlo.
    """

    n_samples, n_dim = xs.shape

    mean = np.mean(xs, axis=0)
    xms = xs - mean

    acors = np.zeros_like(xms)
    for i in range(n_dim):
        for lag in range(n_samples):
            acor = np.sum(xms[:n_samples-lag, i] * xms[lag:, i]) / (n_samples - lag)
            if acor <= 0.0: break
            acors[lag, i] = acor

    act = 1.0 + 2.0 * np.sum(acors[1:], axis=0) / acors[0]
    ess = n_samples / act

    return np.min(ess)


def probs2contours(probs, levels):
    """
    Takes an array of probabilities and produces an array of contours at specified percentile levels
    :param probs: probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    :param levels: percentile levels. have to be in [0.0, 1.0]
    :return: array of same shape as probs with percentile labels
    """

    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def plot_pdf_marginals(pdf, lims, gt=None, levels=(0.68, 0.95)):
    """
    Plots marginals of a pdf, for each variable and pair of variables.
    """

    if pdf.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(lims[0], lims[1], 200)

        pp = pdf.eval(xx[:, np.newaxis], log=False)
        ax.plot(xx, pp)
        ax.set_xlim(lims)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        fig, ax = plt.subplots(pdf.ndim, pdf.ndim)

        lims = np.asarray(lims)
        lims = np.tile(lims, [pdf.ndim, 1]) if lims.ndim == 1 else lims

        for i in range(pdf.ndim):
            for j in range(pdf.ndim):

                if i == j:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 500)
                    pp = pdf.eval(xx, ii=[i], log=False)
                    ax[i, j].plot(xx, pp)
                    ax[i, j].set_xlim(lims[i])
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if gt is not None: ax[i, j].vlines(gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 200)
                    yy = np.linspace(lims[j ,0], lims[j, 1], 200)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[i, j], log=False)
                    pp = pp.reshape(list(X.shape))
                    ax[i, j].contour(X, Y, probs2contours(pp, levels), levels)
                    ax[i, j].set_xlim(lims[i])
                    ax[i, j].set_ylim(lims[j])
                    if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    plt.show(block=False)

    return fig, ax


def plot_hist_marginals(data, lims=None, gt=None):
    """
    Plots marginal histograms and pairwise scatter plots of a dataset.
    """

    n_bins = int(np.sqrt(data.shape[0]))

    if data.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, density=True)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if lims is not None: ax.set_xlim(lims)
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        n_dim = data.shape[1]
        fig, ax = plt.subplots(n_dim, n_dim)
        ax = np.array([[ax]]) if n_dim == 1 else ax

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(n_dim):

                if i == j:
                    ax[i, j].hist(data[:, i], n_bins)
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if lims is not None: ax[i, j].set_xlim(lims[i])
                    if gt is not None: ax[i, j].vlines(gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    ax[i, j].plot(data[:, i], data[:, j], 'k.', ms=2)
                    if lims is not None:
                        ax[i, j].set_xlim(lims[i])
                        ax[i, j].set_ylim(lims[j])
                    if gt is not None: ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    plt.show(block=False)

    return fig, ax


def save(data, file):
    """
    Saves data to a file.
    """

    f = open(file, 'w')
    pickle.dump(data, f)
    f.close()


def load(file):
    """
    Loads data from file.
    """

    f = open(file, 'r')
    data = pickle.load(f)
    f.close()
    return data


def calc_whitening_transform(xs):
    """
    Calculates the parameters that whiten a dataset.
    """

    assert xs.ndim == 2, 'Data must be a matrix'
    N = xs.shape[0]

    means = np.mean(xs, axis=0)
    ys = xs - means

    cov = np.dot(ys.T, ys) / N
    vars, U = np.linalg.eig(cov)
    istds = np.sqrt(1.0 / vars)

    return means, U, istds


def whiten(xs, params):
    """
    Whitens a given dataset using the whitening transform provided.
    """

    means, U, istds = params

    ys = xs.copy()
    ys -= means
    ys = np.dot(ys, U)
    ys *= istds

    return ys


def copy_model_parms(source_model, target_model):
    """
    Copies the parameters of source_model to target_model.
    """

    for sp, tp in zip(source_model.parms, target_model.parms):
        tp.set_value(sp.get_value())


def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[range(labels.size), labels] = 1

    return y


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)


def load_data(name):
    """
    Loads the dataset. Has to be called before anything else.
    :param name: string, the dataset's name
    """

    assert isinstance(name, str), "Name must be a string"
    datasets.root = "../MAFData/"
    data = data_name = None

    if data_name == name:
        return

    if name == "mnist":
        data = datasets.MNIST(logit=True, dequantize=True)
        data_name = name

    elif name == "bsds300":
        data = datasets.BSDS300()
        data_name = name

    elif name == "cifar10":
        data = datasets.CIFAR10(logit=True, flip=True, dequantize=True)
        data_name = name

    elif name == "power":
        data = datasets.POWER()
        data_name = name

    elif name == "gas":
        data = datasets.GAS()
        data_name = name

    elif name == "hepmass":
        data = datasets.HEPMASS()
        data_name = name

    elif name == "miniboone":
        data = datasets.MINIBOONE()
        data_name = name

    else:
        raise ValueError("Unknown dataset")

    return data, data_name


def jointplot(data, title=None, color="grey"):

    rng = np.random.default_rng(seed=1)
    if data.shape[0] > 1000:
        idx = rng.choice(data.shape[0], size=1000, replace=False)
        data = data[idx]

    sns.jointplot(data=pd.DataFrame(data, columns=["x1", "x2"]),
                  x="x1", y="x2", color=color, s=10, alpha=0.2, height=4)
    if title:
        plt.suptitle(title)
    plt.subplots_adjust(top=0.9)


def pairplot(data, mins=None, maxs=None, title=None, color="grey"):

    rng = np.random.default_rng(seed=1)
    if data.shape[0] > 1000:
        rows = rng.choice(data.shape[0], size=1000, replace=False)
        data = data[rows]
    cols = range(data.shape[1])
    if data.shape[1] > 10:
        cols = range(10)
    data = data[:, cols]

    g = sns.pairplot(data=pd.DataFrame(data, columns=[f"x{col}" for col in list(cols)]),
                    height=2, aspect=1, diag_kind="hist", diag_kws={"color": color},
                    plot_kws={"color": color, "s": 10, "alpha": 0.2})

    if mins and maxs:
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                g.axes[i, j].set_xlim((mins[j], maxs[j]))
                g.axes[i, j].set_ylim((mins[i], maxs[i]))

    if title:
        plt.suptitle(title)
        plt.subplots_adjust(top=0.9)


class NumpyDataset(Dataset):

    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        return {"x": self.array[i]}


class VisualCallback(Callback):

    def __init__(self, n_samples, color, mins, maxs, image_size=None, img_every_n_epochs=1):
        self.n_samples = n_samples
        self.color = color
        self.mins = mins
        self.maxs = maxs
        self.image_size = image_size
        self.img_every_n_epochs = img_every_n_epochs

    def _log_pairplot(self, data):
        try:
            cols = range(data.shape[1])
            if data.shape[1] > 10:
                cols = range(10)
            data = data[:, cols]
            mins, maxs = self.mins[cols], self.maxs[cols]
            data = np.clip(data, mins - 1, maxs + 1)    # avoid blowing up seaborn
            g = sns.pairplot(data=pd.DataFrame(data, columns=[f"x{col}" for col in list(cols)]),
                             height=2, aspect=1, diag_kind="hist", diag_kws={"color": self.color},
                             plot_kws={"color": self.color, "s": 10, "alpha": 0.2})
            for i in range(data.shape[1]):
                for j in range(data.shape[1]):
                    g.axes[i, j].set_xlim((mins[j], maxs[j]))
                    g.axes[i, j].set_ylim((mins[i], maxs[i]))
            wandb.log({"sample_pairplots": wandb.Image(plt)})
            plt.close()
        except Exception as e:
            print(f"_log_pairplot raised {e}")

    def _log_images(self, data):
        try:
            fig, ax = plt.subplots(2, 5)
            ax = ax.ravel()
            if data.shape[1] == 63:
                data = np.hstack((data, data[:, [-1]]))  # bsds300, need to add one pixel
            for i in range(10):
                if len(self.image_size) > 2:
                    img = data[i].reshape(self.image_size).transpose(1, 2, 0)
                else:
                    img = data[i].reshape(self.image_size)
                ax[i].imshow(img, interpolation="none")
                ax[i].set_title(str(i))
                ax[i].axis("off")

            wandb.log({"sample_images": plt})
            plt.close()
        except Exception as e:
            print(f"_log_pairplot raised {e}")

    def on_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.img_every_n_epochs != 0:
            return
        samples = pl_module.sample(self.n_samples).detach().cpu().numpy()
        self._log_pairplot(samples)
        if self.image_size:
            self._log_images(samples)


