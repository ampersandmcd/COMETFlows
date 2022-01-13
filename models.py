import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import scipy.stats


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
        x = batch["x"].to(self.device)
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
        nan_idx = torch.all(torch.logical_or(z != z, delta_logp != delta_logp), dim=1)
        inf_idx = torch.all(torch.logical_not(torch.logical_and(torch.isfinite(z), torch.isfinite(delta_logp))), dim=1)
        keep_idx = torch.logical_not(torch.logical_or(nan_idx, inf_idx))
        z, delta_logp = z[keep_idx], delta_logp[keep_idx]

        # compute and log mean nll
        nll = criterion(z, delta_logp) / z.shape[0]
        self.log("t_loss", nll, prog_bar=True)
        self.log("t_nan", torch.sum(nan_idx), prog_bar=True)
        self.log("t_inf", torch.sum(inf_idx), prog_bar=True)
        return nll

    def validation_step(self, batch, batch_idx):
        self.eval()
        x = batch["x"].to(self.device)
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
        nan_idx = torch.all(torch.logical_or(z != z, delta_logp != delta_logp), dim=1)
        inf_idx = torch.all(torch.logical_not(torch.logical_and(torch.isfinite(z), torch.isfinite(delta_logp))), dim=1)
        keep_idx = torch.logical_not(torch.logical_or(nan_idx, inf_idx))
        z, delta_logp = z[keep_idx], delta_logp[keep_idx]

        # compute and log mean nll
        nll = criterion(z, delta_logp) / z.shape[0]
        return {
            "loss": nll,
            "n_nan": torch.sum(nan_idx).type(torch.FloatTensor),
            "n_inf": torch.sum(inf_idx).type(torch.FloatTensor)
        }

    def validation_epoch_end(self, outputs):
        nll = torch.stack([o["loss"] for o in outputs]).mean()
        self.log("v_loss", nll, prog_bar=True)
        n_nan = torch.stack([o["n_nan"] for o in outputs]).mean()
        self.log("v_nan", n_nan, prog_bar=True)
        n_inf = torch.stack([o["n_inf"] for o in outputs]).mean()
        self.log("v_inf", n_inf, prog_bar=True)

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


class HTSFlow(BaseFlow):
    """
    Coupling layer-based normalizing flow model with Student's T latent space for heavy tailed modeling.
    HTS abbreviates Heavy Tail Student.
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


class HTCFlow(BaseFlow):
    """
    Coupling layer-based normalizing flow model with copula transform for heavy tail modeling.
    HTC abbreviates Heavy Tail Copula.
    Inspired by Wiese et al. Copula & Marginal Flows.
    """
    def __init__(self, d, hidden_ds, lr):
        super().__init__(d, hidden_ds, lr)
        self.layers.append(CopulaLayer())    # map to unit hypercube
        self.layers.append(LogitLayer())     # map to Rn before coupling layers
        for hidden_d in hidden_ds:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True))
        self.layers = nn.ModuleList(self.layers)


class TDFlow(BaseFlow):
    """
    Coupling layer-based normalizing flow model with conditional noise auxiliary variable for manifold modeling.
    TD abbreviates Tail Dependence.
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
    def __init__(self, d, hidden_ds, lr):
        super().__init__(d, hidden_ds, lr)
        self.conditional_noise = True
        self.layers.append(CopulaLayer())    # map to unit hypercube
        self.layers.append(LogitLayer())     # map to Rn before coupling layers
        for hidden_d in hidden_ds:
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=False, conditional_noise=True))
            self.layers.append(CouplingLayer(self.d, hidden_d, swap=True, conditional_noise=True))
        self.layers = nn.ModuleList(self.layers)

