import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import scipy.stats
import scipy.optimize
import statsmodels.api as sm

from scipy.stats import genpareto
from interp1d import Interp1d


class GaussianNLL(nn.Module):
    """
    Compute negative log likelihood of sequential flow wrt isotropic latent Gaussian.
    """
    def __init__(self, device):
        """
        Initialize Gaussian NLL nn.module.
        """
        super().__init__()
        self.device = device
        self.log2pi = torch.FloatTensor((np.log(2 * np.pi),)).to(self.device)

    def forward(self, z, delta_logp):
        """
        Compute NLL of z with respect to the multivariate Gaussian, applying penalty term delta_logp.

        :param z: [tensor] Tensor with shape (n, d).
        :param delta_logp: [tensor] Tensor with shape (n,).
        :return: [scalar] Penalized NLL(z) with respect to standard multivariate Gaussian.
        """
        N, D = z.shape[0], z.shape[1]
        log_pz = -0.5 * D * self.log2pi - 0.5 * torch.sum(z**2, dim=1, keepdim=True)    # (n, 1)
        log_px = log_pz - delta_logp                                                    # (n, 1)
        return -torch.sum(log_px)


class StudentNLL(nn.Module):
    """
    Compute negative log likelihood of sequential flow wrt isotropic latent Student T.
    """
    def __init__(self, device, dof=1):
        """
        Initialize Student NLL nn.module.

        :param dof: [float] Degrees of freedom parameter.
        """
        super().__init__()
        self.device = device
        self.nu = torch.FloatTensor((dof,)).to(self.device)
        self.pi = torch.FloatTensor((np.pi,)).to(self.device)
        self.eps = 1e-10

    def forward(self, z, delta_logp):
        """
        Compute NLL of z with respect to multivariate Student T, applying penalty term delta_logp.
        Assume identity covariance, zero mean, nu degrees of freedom.

        :param z: [tensor] Tensor with shape (n_instances, n_features).
        :param delta_logp: [tensor] Tensor with shape (n_instances, n_features).
        :return: [scalar] Penalized NLL(z) with respect to standard multivariate Student T.
        """
        # shoutout to https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/_multivariate.py#L3997
        # compute and return log probability
        N, D = z.shape[0], z.shape[1]
        log_pz = torch.lgamma((self.nu + D) / 2 + self.eps)
        log_pz -= torch.lgamma(self.nu / 2 + self.eps)
        log_pz -= 0.5 * D * torch.log(self.nu * self.pi + self.eps)
        prod = 1 + (1 / (self.nu + self.eps)) * torch.sum(z**2, dim=1, keepdim=True)    # (n, 1)
        log_pz = log_pz - 0.5 * (self.nu + D) * torch.log(prod + self.eps)              # (n, 1)
        log_px = log_pz - delta_logp                                                    # (n, 1)
        return -torch.sum(log_px)


class TorchGPD(nn.Module):
    """
    Implement collection of two 1D GPD distributions to model tails of distribution.
    """
    def __init__(self, data, a, b, alpha, beta):
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.eps = 0

        # fit parameters of lower tail
        lower_data = data[data < self.alpha].detach().cpu().numpy()
        if len(lower_data) > 1:
            self.lower_xi, self.lower_mu, self.lower_sigma = genpareto.fit(self.alpha.detach().cpu().numpy() - lower_data)
            self.lower_xi = np.clip(self.lower_xi, 0, 1)    # clip to well-behaved xi range
        else:
            self.lower_xi, self.lower_mu, self.lower_sigma = None, None, None

        # fit parameters of upper tail
        upper_data = data[data > self.beta].detach().cpu().numpy()
        if len(upper_data) > 1:
            self.upper_xi, self.upper_mu, self.upper_sigma = genpareto.fit(-self.beta.detach().cpu().numpy() + upper_data)
            self.upper_xi = np.clip(self.upper_xi, 0, 1)    # clip to well-behaved xi range
        else:
            self.upper_xi, self.upper_mu, self.upper_sigma = None, None, None

    def split_data(self, x, quantile=False):

        left, right = self.a if quantile else self.alpha, self.b if quantile else self.beta
        lower_idx = x < left + self.eps
        upper_idx = x > right - self.eps
        middle_idx = torch.logical_not(torch.logical_or(lower_idx, upper_idx))
        lower_data = x[lower_idx].detach().cpu().numpy()
        upper_data = x[upper_idx].detach().cpu().numpy()
        middle_data = x[middle_idx].detach().cpu().numpy()
        return (lower_idx, middle_idx, upper_idx), (lower_data, middle_data, upper_data)

    def cdf(self, x):

        (lower_idx, middle_idx, upper_idx), (lower_data, middle_data, upper_data) = self.split_data(x)

        # handle asymmetric tail logic: when a=0 (b=1, resp.), the left (right, resp.) tail will not matter
        # recall that torch.where will properly merge KDE with tails, so we can set points in middle to anything (zero)
        middle_cdf = torch.zeros((middle_data.shape[0]))
        if self.a == 0:
            lower_cdf = torch.zeros((lower_data.shape[0])).type(torch.FloatTensor)
        else:
            lower_cdf = self.a.detach().cpu().numpy() * (1 - genpareto.cdf(self.alpha.detach().cpu().numpy() - lower_data, loc=-self.lower_mu, scale=self.lower_sigma, c=self.lower_xi))
            lower_cdf = torch.from_numpy(lower_cdf).type(torch.FloatTensor)
        if self.b == 1:
            upper_cdf = torch.zeros((upper_data.shape[0])).type(torch.FloatTensor)
        else:
            upper_cdf = self.b.detach().cpu().numpy() + (1 - self.b.detach().cpu().numpy()) * genpareto.cdf(-self.beta.detach().cpu().numpy() + upper_data, loc=self.upper_mu, scale=self.upper_sigma, c=self.upper_xi)
            upper_cdf = torch.from_numpy(upper_cdf).type(torch.FloatTensor)

        cdf = torch.zeros_like(x)
        cdf[lower_idx] = lower_cdf
        cdf[middle_idx] = middle_cdf
        cdf[upper_idx] = upper_cdf
        return cdf

    def log_prob(self, x):

        (lower_idx, middle_idx, upper_idx), (lower_data, middle_data, upper_data) = self.split_data(x)

        # handle asymmetric tail logic: when a=0 (b=1, resp.), the left (right, resp.) tail will not matter
        # recall that torch.where will properly merge KDE with tails, so we can set points in middle to anything (zero)
        middle_log_prob = torch.zeros((middle_data.shape[0]))
        if self.a == 0:
            lower_log_prob = torch.zeros((lower_data.shape[0])).type(torch.FloatTensor)
        else:
            lower_log_prob = np.log(self.a.detach().cpu().numpy()) + genpareto.logpdf(self.alpha.detach().cpu().numpy() - lower_data, loc=-self.lower_mu, scale=self.lower_sigma, c=self.lower_xi)
            lower_log_prob = torch.from_numpy(lower_log_prob).type(torch.FloatTensor)
        if self.b == 1:
            upper_log_prob = torch.zeros((upper_data.shape[0])).type(torch.FloatTensor)
        else:
            upper_log_prob = np.log(1 - self.b.detach().cpu().numpy()) + genpareto.logpdf(-self.beta.detach().cpu().numpy() + upper_data, loc=self.upper_mu, scale=self.upper_sigma, c=self.upper_xi)
            upper_log_prob = torch.from_numpy(upper_log_prob).type(torch.FloatTensor)

        log_prob = torch.zeros_like(x)
        log_prob[lower_idx] = lower_log_prob
        log_prob[middle_idx] = middle_log_prob
        log_prob[upper_idx] = upper_log_prob
        return log_prob

    def icdf(self, u):

        u = u.clip(0, 1)    # compensate for any numerical instability
        (lower_idx, middle_idx, upper_idx), (lower_u, middle_u, upper_u) = self.split_data(u, quantile=True)

        # handle asymmetric tail logic: when a=0 (b=1, resp.), the left (right, resp.) tail will not matter
        # recall that torch.where will properly merge KDE with each tail, so we can set irrelevant points here to zero
        middle_icdf = torch.ones((middle_u.shape[0]))
        if self.a == 0:
            lower_icdf = torch.ones((lower_u.shape[0])).type(torch.FloatTensor)
        else:
            lower_icdf = self.alpha.detach().cpu().numpy() - genpareto.ppf(1 - lower_u / self.a.detach().cpu().numpy(), c=self.lower_xi)
            lower_icdf = torch.from_numpy(lower_icdf).type(torch.FloatTensor)
        if self.b == 1:
            upper_icdf = torch.ones((upper_u.shape[0])).type(torch.FloatTensor)
        else:
            upper_icdf = self.beta.detach().cpu().numpy() + genpareto.ppf(1 - (1 - upper_u) / (1 - self.b.detach().cpu().numpy()), loc=self.upper_mu, scale=self.upper_sigma, c=self.upper_xi)
            upper_icdf = torch.from_numpy(upper_icdf).type(torch.FloatTensor)

        icdf = torch.ones_like(u)
        icdf[lower_idx] = lower_icdf
        icdf[middle_idx] = middle_icdf
        icdf[upper_idx] = upper_icdf
        return icdf


class CopulaLayer(nn.Module):
    """
    Construct invertible layer mapping marginal distributions to uniform[0, 1] by
    decoupled 1D KDE on random sample of size n < N of data points.
    """

    def __init__(self, data, n=1000, a=0.05, b=0.95, debug=False):
        super().__init__()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).type(torch.FloatTensor)
        else:
            data = data
        self.n, self.d = data.shape
        self.n_anchors = min(n, self.n)
        self.register_buffer("a", torch.FloatTensor([a]))      # lower quantile cutoff
        self.register_buffer("b", torch.FloatTensor([b]))      # lower quantile cutoff
        data = data.sort(dim=0).values
        lower_idx = int(self.a * self.n)
        upper_idx = int(self.b * self.n) - 1
        anchor_idxs = torch.randint(low=lower_idx, high=upper_idx, size=(self.n_anchors,))
        self.register_buffer("anchors", data[anchor_idxs])               # sample for KDE
        self.register_buffer("alpha", data[lower_idx])                   # lower data cutoff
        self.register_buffer("beta", data[upper_idx])                    # upper data cutoff

        # define a tail model for each dimension i=1,...,d
        tails = []
        tail_idxs = torch.cat((torch.arange(0, lower_idx), torch.arange(upper_idx, self.n - 1)))
        for i in range(self.d):
            tails.append(TorchGPD(data=data[tail_idxs, i], a=self.a, b=self.b, alpha=self.alpha[[i]], beta=self.beta[[i]]))
        self.tails = nn.ModuleList(tails)

        # echo xi parameters from tail model to command line
        print(f"Lower xi: {[x.lower_xi for x in self.tails]}")
        print(f"Upper xi: {[x.upper_xi for x in self.tails]}")

        # define KDE mixture for central data
        self.mixture = self.construct_mixture()

    def __str__(self):
        lower_tails = "".join([f"\n\t(mu, sigma, xi) = {(t.lower_mu, t.lower_sigma, t.lower_xi)}" for t in self.tails])
        upper_tails = "".join([f"\n\t(mu, sigma, xi) = {(t.upper_mu, t.upper_sigma, t.upper_xi)}" for t in self.tails])
        return "CopulaTransform\n" \
               f"N, D: {self.n, self.d}\n" \
               f"a, b: {self.a, self.b}\n" \
               f"alpha, beta: {self.alpha, self.beta}\n" \
               f"lower_tails:{lower_tails}\n" \
               f"upper_tails:{upper_tails}"

    __repr__ = __str__

    def construct_mixture(self):
        mixture = []
        for i in range(self.d):
            kde = sm.nonparametric.KDEUnivariate(self.anchors[:, i])
            kde.fit()
            mixture.append(kde)
        return mixture

    def forward(self, x, noise_level=None, logpx=None, reverse=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.FloatTensor)
        if isinstance(logpx, np.ndarray):
            logpx = torch.from_numpy(logpx).type(torch.FloatTensor)

        # work on one dimension i=1,...,d at a time (acceptably efficient for low D)
        out, logpdf = [], []
        for i in range(self.d):
            x_i = x[..., i]         # assume features are in last dimension
            if not reverse:
                # given data x, produce quantile vector u; use CDF for transformation

                # compute CDF i wrt each anchor, then rescale with Definition 11 from Wiese and fix tails
                kde_support = torch.from_numpy(self.mixture[i].support)
                kde_cdf = torch.from_numpy(self.mixture[i].cdf)
                cdf_i = Interp1d()(kde_support, kde_cdf, x_i)
                F_a = Interp1d()(kde_support, kde_cdf, self.alpha[[i]])
                F_b = Interp1d()(kde_support, kde_cdf, self.beta[[i]])
                cdf_i = self.a + (self.b - self.a) * (cdf_i - F_a) / (F_b - F_a)
                cdf_i = torch.where(torch.logical_or(x_i < self.alpha[i], x_i > self.beta[i]),
                                    self.tails[i].cdf(x_i),
                                    cdf_i)
                out.append(cdf_i.view(-1, 1))

                # in either direction, change in probability density is given by sum_i log|f_i(x_i)|
                # compute log prob in data space with x_i, then rescale with Definition 11 from Wiese and fix tails
                logpdf_i = torch.from_numpy(self.mixture[i].evaluate(x_i.detach().cpu().numpy())).type(torch.FloatTensor)
                pdf_i = torch.exp(logpdf_i)
                pdf_i = (self.b - self.a) / (F_b - F_a) * pdf_i
                pdf_i = torch.where(torch.logical_or(x_i < self.alpha[i], x_i > self.beta[i]),
                                    torch.exp(self.tails[i].log_prob(x_i)),
                                    pdf_i)
                logpdf.append(-torch.log(pdf_i.view(-1, 1)))

            else:
                # given quantile u, produce data vector x; use inverse CDF for transformation

                # compute inverseCDF i wrt each anchor, then rescale with Definition 11 from Wiese and fix tails
                kde_support = torch.from_numpy(self.mixture[i].support)
                kde_cdf = torch.from_numpy(self.mixture[i].cdf)
                kde_icdf = torch.from_numpy(self.mixture[i].icdf)
                ones_i = torch.linspace(0, 1, len(kde_icdf))
                icdf_i = Interp1d()(ones_i, kde_icdf, x_i)

                # handle null tail (a=0, b=1) logic
                if self.a == 0:
                    F_inv_a = self.alpha[i]
                else:
                    F_inv_a = Interp1d()(ones_i, kde_icdf, self.a)
                if self.b == 1:
                    F_inv_b = self.beta[i]
                else:
                    F_inv_b = Interp1d()(ones_i, kde_icdf, self.b)
                icdf_i = self.alpha[i] + (self.beta[i] - self.alpha[i]) * (icdf_i - F_inv_a) / (F_inv_b - F_inv_a)
                icdf_i = torch.where(torch.logical_or(x_i < self.a, x_i > self.b),
                                     self.tails[i].icdf(x_i),
                                     icdf_i)
                out.append(icdf_i.view(-1, 1))

                # in either direction, change in probability density is given by sum_i log|f_i(x_i)|
                # compute log prob in data space with x_i, then rescale with Definition 11 from Wiese and fix tails
                logpdf_i = torch.from_numpy(self.mixture[i].evaluate(x_i.detach().cpu().numpy())).type(torch.FloatTensor)
                pdf_i = torch.exp(logpdf_i)
                F_a = Interp1d()(kde_support, kde_cdf, self.alpha[[i]])
                F_b = Interp1d()(kde_support, kde_cdf, self.beta[[i]])
                pdf_i = (self.b - self.a) / (F_b - F_a) * pdf_i
                pdf_i = torch.where(torch.logical_or(x_i < self.a, x_i > self.b),
                                    torch.exp(self.tails[i].log_prob(icdf_i)),
                                    pdf_i)
                logpdf.append(torch.log(pdf_i.view(-1, 1)))

        # sum all i log|f_i(x_i)| terms in logpdf list to compute Jacobian term for each point
        # result is shape (x.shape[0],) which comprises change of variable term for each x evaluated
        deltalogp = sum(logpdf)

        # do not modify log probability under model during training, as this layer is not trainable!
        # only modify log probability at inference time
        if not self.training:
            logpx = logpx + deltalogp
        return torch.hstack(out), logpx


class LogitLayer(nn.Module):
    """
    Map data from unit hypercube to Rn in forward direction with logit function.
    Map data from Rn to unit hypercube in reverse direction with sigmoid function.
    """

    def __init__(self, alpha=1e-6):
        nn.Module.__init__(self)
        self.alpha = alpha

    def logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + np.log(1 - 2 * self.alpha)
        return logdetgrad

    def forward(self, x, noise_level=None, logpx=None, reverse=False):
        if reverse:
            x = (torch.sigmoid(x) - self.alpha) / (1 - 2 * self.alpha)
            delta_logp = self.logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=True)
            if logpx is None:
                return x
            return x, logpx + delta_logp
        else:
            s = self.alpha + (1 - 2 * self.alpha) * x
            delta_logp = self.logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=True)
            x = torch.log(s) - torch.log(1 - s)
            if logpx is None:
                return x
            return x, logpx - delta_logp


class CouplingLayer(nn.Module):
    """
    Basic coupling layer.
    Inspired by ffjord.lib.layers.coupling at https://github.com/rtqichen/ffjord.
    """

    def __init__(self, d, hidden_d=64, swap=False, conditional_noise=False):
        super().__init__()
        self.d = d - (d // 2)
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, hidden_d),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_d, hidden_d),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_d, (d - self.d) * 2),
        )
        if conditional_noise:
            self.hyper_gate = nn.Linear(1, (d - self.d) * 2)
            self.hyper_bias = nn.Linear(1, (d - self.d) * 2)

    def forward(self, x, noise_level=None, logpx=None, reverse=False):
        """
        For use with models that DO apply conditional noise.
        """
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_d = self.d
        out_d = x.shape[1] - self.d

        if noise_level is None:
            s_t = self.net_s_t(x[:, :in_d])
        else:
            s_t = self.hyper_gate(noise_level) * self.net_s_t(x[:, :in_d]) + self.hyper_bias(noise_level)

        scale = torch.sigmoid(s_t[:, :out_d] + 2.)
        shift = s_t[:, out_d:]
        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        if self.swap:
            y = torch.cat([y1, x[:, :self.d]], 1)
        else:
            y = torch.cat([x[:, :self.d], y1], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


class BaseFlow(pl.LightningModule):
    """
    Base normalizing flow model.
    Inspired by ffjord.lib.layers.container at https://github.com/rtqichen/ffjord.
    """

    def __init__(self, d, hidden_ds, lr):
        super().__init__()
        self.save_hyperparameters()
        self.d = d
        self.hidden_ds = hidden_ds
        self.lr = lr
        self.conditional_noise = False
        self.layers = []

    def forward(self, x, noise_level=None, logpx=None, reverse=False):
        if logpx is None:
            logpx = torch.zeros(x.shape[0], 1, device=self.device)

        if reverse:
            layers = reversed(self.layers)
        else:
            layers = self.layers

        for layer in layers:
            x, logpx = layer(x, noise_level, logpx, reverse)
        return x, logpx

    def training_step(self, batch, batch_idx):
        self.train()
        x = batch["x"].type(torch.FloatTensor).to(self.device)
        criterion = self.get_criterion()

        # perform forward pass
        if self.conditional_noise:
            std_min, std_max = 1e-3, 1e-1
            noise = (std_max - std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + std_min
            noisy_x = x + torch.randn_like(x) * noise
            z, delta_logp = self.forward(noisy_x, noise_level=noise)
        else:
            z, delta_logp = self.forward(x)

        # check nans and infs
        # nan_idx = torch.all(torch.logical_or(z != z, delta_logp != delta_logp), dim=1)
        # inf_idx = torch.all(torch.logical_not(torch.logical_and(torch.isfinite(z), torch.isfinite(delta_logp))), dim=1)
        # keep_idx = torch.logical_not(torch.logical_or(nan_idx, inf_idx))
        # z, delta_logp = z[keep_idx], delta_logp[keep_idx]

        # compute and log mean nll
        nll = criterion(z, delta_logp) / z.shape[0]
        self.log("t_loss", nll, prog_bar=True)
        # self.log("t_nan", torch.sum(nan_idx), prog_bar=True)
        # self.log("t_inf", torch.sum(inf_idx), prog_bar=True)
        return nll

    def validation_step(self, batch, batch_idx):
        self.eval()
        x = batch["x"].type(torch.FloatTensor).to(self.device)
        criterion = self.get_criterion()

        # perform forward pass
        if self.conditional_noise:
            std_min, std_max = 1e-3, 5e-3       # set to near-zero noise for validation
            noise = (std_max - std_min) * torch.rand_like(x[:, 0]).view(-1, 1) + std_min
            noisy_x = x + torch.randn_like(x) * noise
            z, delta_logp = self.forward(noisy_x, noise_level=noise)
        else:
            z, delta_logp = self.forward(x)

        # check nans and infs
        # nan_idx = torch.all(torch.logical_or(z != z, delta_logp != delta_logp), dim=1)
        # inf_idx = torch.all(torch.logical_not(torch.logical_and(torch.isfinite(z), torch.isfinite(delta_logp))), dim=1)
        # keep_idx = torch.logical_not(torch.logical_or(nan_idx, inf_idx))
        # z, delta_logp = z[keep_idx], delta_logp[keep_idx]

        # compute and log mean nll
        nll = criterion(z, delta_logp) / z.shape[0]
        return {
            "loss": nll,
            # "n_nan": torch.sum(nan_idx).type(torch.FloatTensor),
            # "n_inf": torch.sum(inf_idx).type(torch.FloatTensor)
        }

    def validation_epoch_end(self, outputs):
        nll = torch.stack([o["loss"] for o in outputs]).mean()
        self.log("v_loss", nll, prog_bar=True)
        # n_nan = torch.stack([o["n_nan"] for o in outputs]).mean()
        # self.log("v_nan", n_nan, prog_bar=True)
        # n_inf = torch.stack([o["n_inf"] for o in outputs]).mean()
        # self.log("v_inf", n_inf, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_criterion(self):
        return GaussianNLL(self.device)

    def get_z_samples(self, n_samples):
        return torch.randn((n_samples, self.d), device=self.device)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sample(self, n_samples):
        self.eval()
        z = self.get_z_samples(n_samples)

        # perform reverse pass
        if self.conditional_noise:
            std_min, std_max = 1e-3, 5e-3       # set to near-zero noise for validation
            noise = (std_max - std_min) * torch.rand_like(z[:, 0]).view(-1, 1) + std_min
            noisy_z = z + torch.randn_like(z) * noise
            x, _ = self.forward(noisy_z, noise_level=noise, reverse=True)
        else:
            x, _ = self.forward(z, reverse=True)

        return x


class VanillaFlow(BaseFlow):
    """
    Vanilla coupling layer-based normalizing flow model.
    Inspired by Dinh et al. RealNVP.
    """
    def __init__(self, d, hidden_ds, lr):
        super().__init__(d, hidden_ds, lr)
        for hidden_d in hidden_ds:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True))
        self.layers = nn.ModuleList(self.layers)


class TAFlow(BaseFlow):
    """
    Coupling layer-based normalizing flow model with Student's T latent space for heavy tailed modeling.
    Inspired by Jaini et al. Tail Adaptive Flow.
    """
    def __init__(self, d, hidden_ds, lr, dof):
        super().__init__(d, hidden_ds, lr)
        for hidden_d in hidden_ds:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True))
        self.layers = nn.ModuleList(self.layers)
        self.dof = dof

    def get_criterion(self):
        return StudentNLL(self.device, self.dof)

    def get_z_samples(self, n_samples):
        samples = scipy.stats.multivariate_t(loc=np.zeros(self.d,), shape=np.eye(self.d), df=self.dof).rvs(size=n_samples)
        return torch.from_numpy(samples).type(torch.FloatTensor).to(self.device)


class CMFlow(BaseFlow):
    """
    Coupling layer-based normalizing flow model with copula transform for heavy tail modeling.
    Inspired by Wiese et al. Copula & Marginal Flows.
    """
    def __init__(self, d, hidden_ds, lr, data, a, b):
        super().__init__(d, hidden_ds, lr)
        self.layers.append(CopulaLayer(data, a=a, b=b))    # map to unit hypercube
        self.layers.append(LogitLayer())                                # map to Rn before coupling layers
        for hidden_d in hidden_ds:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True))
        self.layers = nn.ModuleList(self.layers)


class SoftFlow(BaseFlow):
    """
    Coupling layer-based normalizing flow model with conditional noise auxiliary variable for manifold modeling.
    Inspired by Kim et al. SoftFlow.
    """
    def __init__(self, d, hidden_ds, lr):
        super().__init__(d, hidden_ds, lr)
        self.conditional_noise = True
        for hidden_d in hidden_ds:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False, conditional_noise=True))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True, conditional_noise=True))
        self.layers = nn.ModuleList(self.layers)


class COMETFlow(BaseFlow):
    """
    Original coupling layer-based normalizing flow model with copula transform for heavy tail modeling and
    with conditional noise auxiliary variable for manifold modeling.
    """
    def __init__(self, d, hidden_ds, lr, data, a, b):
        super().__init__(d, hidden_ds, lr)
        self.conditional_noise = True
        self.layers.append(CopulaLayer(data, a=a, b=b))    # map to unit hypercube
        self.layers.append(LogitLayer())                                # map to Rn before coupling layers
        for hidden_d in hidden_ds:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False, conditional_noise=True))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True, conditional_noise=True))
        self.layers = nn.ModuleList(self.layers)
