# sample.py

import torch
import numpy as np
import pyro
import pyro.distributions as dist

_softplus = torch.nn.functional.softplus

def downsample(mtx, n, seed=0):
    if n is None or n >= mtx.shape[0]:
        return mtx, torch.arange(mtx.shape[0], device="cpu")

    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(mtx.shape[0], generator=g, device="cpu")[:n]
    return mtx[idx], idx


def get_learned_params(trident, data):
    guide_trace = pyro.poutine.trace(trident.guide).get_trace(*data)
    trained_model = pyro.poutine.replay(trident.model, trace=guide_trace)
    trained_trace = pyro.poutine.trace(trained_model).get_trace(*data)
    params = trained_trace.nodes
    return params


def sample(adata, variable, **kwargs):
    if isinstance(variable, str):
        variable = [variable]
    elif not isinstance(variable, list):
        raise ValueError("variable must be either a string or a list.")

    allowed_vars = {"gamma", "eta", "pi", "chi", "X", "L"}

    invalid_vars = set(variable) - allowed_vars
    if invalid_vars:
        raise ValueError(
            f"The following variable values are not allowed: {invalid_vars}. "
            'Allowed values are ("X", "L", "pi", "eta", "gamma", "chi").'
        )

    read_sums = torch.tensor(np.asarray(adata.X.sum(1)).ravel())
    concentration = adata.uns["trident"]["config"]["concentration"]

    sample_funcs = {
        "X": lambda **kw: sample_X(read_sums, **kw),
        "L": lambda **kw: sample_L(concentration, **kw),
        "gamma": lambda **kw: sample_gamma(**kw),
        "eta": lambda **kw: sample_eta(**kw),
        "pi": lambda **kw: sample_pi(**kw),
        "chi": lambda **kw: sample_chi(**kw),
    }

    samples = [sample_funcs[v](**kwargs) for v in variable]
    if len(samples) == 1:
        return samples[0]
    return samples


def sample_gamma(n_samples: int = 100, return_mean: bool = False, **kwargs):
    """
    Sample gamma (topic-to-feature distributions).
    
    Returns: [n_samples, K, T] or [K, T] if return_mean=True
    """
    loc = pyro.param("log_gamma_loc")
    su = pyro.param("log_gamma_su")
    log_gamma = dist.Normal(loc, su).sample((n_samples,))
    log_gamma = log_gamma.mean(0) if return_mean else log_gamma
    gamma = torch.softmax(log_gamma, dim=-1)
    return gamma


def sample_eta(n_samples: int = 100, return_mean: bool = False, 
               start: int = None, end: int = None, **kwargs):
    """
    Sample eta (cell-to-topic distributions).
    
    Args:
        n_samples: Number of posterior samples
        return_mean: If True, return posterior mean instead of samples
        start, end: Optional slice indices for chunked sampling
    
    Returns: [n_samples, R, K] or [R, K] if return_mean=True
    """
    loc = pyro.param("log_eta_loc").detach()
    su = pyro.param("log_eta_su").detach()
    
    if start is not None and end is not None:
        loc = loc[start:end]
        su = su[start:end]
    
    log_eta = dist.Normal(loc,su).sample((n_samples,))
    log_eta = log_eta.mean(0) if return_mean else log_eta
    eta = torch.softmax(log_eta, dim=-1)
    return eta


def sample_pi(n_samples: int = 100, return_mean: bool = False, **kwargs):
    """
    Sample pi (success probabilities for NegativeBinomial).
    
    Returns: [n_samples, T] or [T] if return_mean=True
    """
    loc = pyro.param("pi_loc").detach()
    sd = pyro.param("pi_sd").detach()
    pi = dist.TransformedDistribution(
        dist.Normal(loc, sd), 
        [dist.transforms.SigmoidTransform()]
    ).sample((n_samples,))
    return pi.mean(0) if return_mean else pi


def sample_chi(n_samples: int = 100, return_mean: bool = False, **kwargs):
    """
    Sample chi (topic-specific scaling factors).
    
    Returns: [n_samples, K, 1] or [K, 1] if return_mean=True
    """
    loc = pyro.param("chi_loc").detach()
    scale = pyro.param("chi_scale").detach()
    chi = dist.LogNormal(loc, scale).sample((n_samples,))
    return chi.mean(0) if return_mean else chi


def sample_X(read_sums, n_samples: int = 1000, return_mean: bool = False,
             start: int = None, end: int = None, **kwargs):
    eta = sample_eta(n_samples, return_mean=True, start=start, end=end)  # [R, K]
    gamma = sample_gamma(n_samples, return_mean=True)                     # [K, T]
    pi = sample_pi(n_samples, return_mean=True)                           # [T]

    mc_slice = read_sums[start:end] if start is not None and end is not None else read_sums
    mc_slice = mc_slice.to(device=eta.device, dtype=eta.dtype)
    shape = (eta @ gamma) * mc_slice.unsqueeze(-1)  # [R, T]
    samples = dist.NegativeBinomial(shape, pi).sample()  # [R, T]
    return samples

def sample_L(concentration, n_samples: int = 1000, eps: float = 1e-8, **kwargs):
    gamma = sample_gamma(n_samples, return_mean=True)  # [K, T]
    chi = sample_chi(n_samples, return_mean=True)      # [K, 1]
    pi = sample_pi(n_samples, return_mean=True)        # [T]

    log_phi = (chi.squeeze(-1) @ gamma).squeeze(0) 
    phi = _softplus(log_phi) + eps
    
    samples = dist.Beta(
        concentration1=concentration * (pi * phi),
        concentration0=concentration * ((1.0 - pi) * phi),
    ).sample()
    return samples
