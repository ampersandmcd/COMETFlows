import numpy as np
import torch
import torch.nn as nn


class GaussianNLL(nn.Module):
    """
    Compute negative log likelihood of sequential flow wrt isotropic latent Gaussian.
    """
    def __init__(self):
        """
        Initialize Gaussian NLL nn.module.
        """
        super().__init__()
        self.log2pi = torch.FloatTensor((np.log(2 * np.pi),))

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
    def __init__(self, dof=1):
        """
        Initialize Student NLL nn.module.

        :param dof: [float] Degrees of freedom parameter.
        """
        super().__init__()
        self.nu = torch.FloatTensor((dof,))
        self.pi = torch.FloatTensor((np.pi,))
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

    def forward(self, x, logpx=None, reverse=False):
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

    def forward(self, x, noise_level, logpx=None, reverse=False):
        return self.forward(x, logpx, reverse)


class CopulaLayer(nn.Module):
    pass    # TODO


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
            self.hyper_gate = nn.Linear(1, self.d)
            self.hyper_bias = nn.Linear(1, self.d)

    def forward(self, x, logpx=None, reverse=False):
        """
        For use with models that DO NOT apply conditional noise.
        """
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_d = self.d
        out_d = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_d])
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

    def forward(self, x, noise_level, logpx=None, reverse=False):
        """
        For use with models that DO apply conditional noise.
        """
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_d = self.d
        out_d = x.shape[1] - self.d

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


class BaseFlow(nn.Module):
    """
    Base normalizing flow model.
    Inspired by ffjord.lib.layers.container at https://github.com/rtqichen/ffjord.
    """

    def __init__(self, d, hidden_ds):
        super().__init__()
        self.d = d
        self.hidden_ds = hidden_ds
        self.layers = []

    def forward(self, x, logpx=None, reverse=False):
        """
        For use with models that DO NOT apply conditional noise.
        """
        if reverse:
            layers = reversed(self.layers)
        else:
            layers = self.layers
        for layer in layers:
            x, logpx = layer(x, logpx, reverse)
        return x, logpx

    def forward(self, x, noise_level, logpx=None, reverse=False):
        """
        For use with models that DO apply conditional noise.
        """
        if reverse:
            layers = reversed(self.layers)
        else:
            layers = self.layers
        for layer in layers:
            x, logpx = layer(x, noise_level, logpx, reverse)
        return x, logpx

    def get_criterion(self):
        return None


class VanillaFlow(BaseFlow):
    """
    Vanilla coupling layer-based normalizing flow model.
    Inspired by Dinh et al. RealNVP
    """
    def __init__(self, d, hidden_ds):
        super().__init__(d, hidden_ds)
        for hidden_d in hidden_ds[1:-1]:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True))

    def get_criterion(self):
        return GaussianNLL()


class HTSFlow(VanillaFlow):
    """
    Coupling layer-based normalizing flow model with Student's T latent space for heavy tailed modeling.
    HTS abbreviates Heavy Tail Student.
    Inspired by Jaini et al. Tail Adaptive Flow
    """
    def __init__(self, d, hidden_ds, dof):
        super().__init__(d, hidden_ds)
        self.dof = dof

    def get_criterion(self):
        return StudentNLL(self.dof)


class HTCFlow(VanillaFlow):
    """
    Coupling layer-based normalizing flow model with copula transform for heavy tail modeling.
    HTC abbreviates Heavy Tail Copula.
    Inspired by Wiese et al. Copula & Marginal Flows
    """
    def __init__(self, d, hidden_ds, dof):
        super().__init__(d, hidden_ds)
        self.layers.insert(0, CopulaLayer())    # map to unit hypercube
        self.layers.insert(1, LogitLayer())     # map to Rn before coupling layers


class TDFlow(VanillaFlow):
    """
    Coupling layer-based normalizing flow model with conditional noise auxiliary variable for manifold modeling.
    TD abbreviates Tail Dependence.
    Inspired by Kim et al. SoftFlow
    """
    def __init__(self, d, hidden_ds):
        super().__init__(d, hidden_ds)
        for hidden_d in hidden_ds[1:-1]:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False, conditional_noise=True))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True, conditional_noise=True))


