# model.py

import torch, pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.infer.autoguide import AutoDelta, AutoNormal
from pyro import poutine

from typing import Union
import dataclasses
from dataclasses import dataclass, field

_softplus = torch.nn.functional.softplus

@dataclass(unsafe_hash=True)
class TridentConfig:
    reads: str
    library: str
    database: str
    pair: str = None
    pathogen_dict: str = None

    k: int = 35
    l: int = 31
    spaces: int = 3
    
    num_reads: int = None
    num_targets: int = None
    num_latent: int = 5
    max_var: int = 300
    max_acc_per_mm: int = 512

    subsample_size: int=2048
    taxid_filter_quantile: float=0.1
    n_steps: int = 1000
    learning_rate: float = 1e-2
    threads: int=32
    sqlite: bool=True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    verbose: bool = True
    seed: int = 1

    eps: float = 1e-8
    alpha_pi: float = 1.
    beta_pi: float = 1.
    reestimate_threshold: float = 0.5
    concentration: float = 5
    chi_log_shape_model: float = 0.0
    chi_log_scale_model: float = 1.0
    chi_log_shape_guide: float = 0.0
    chi_log_scale_guide: float = 1.0
    pi_guide_scale_mult: float = 1.0

    def to_dict(self):
        d = dataclasses.asdict(self)
        if isinstance(d.get("dtype"), torch.dtype):
            d["dtype"] = str(d["dtype"])
        return d

    def copy(self, **kwargs):
        """Return a copy of this config, optionally with field overrides."""
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def from_dict(cls, config_dict):
        processed = {}
        field_names = {f.name for f in dataclasses.fields(cls)}
        for key, value in config_dict.items():
            if key not in field_names:
                continue
            if key == "dtype" and isinstance(value, str):
                v = value
                if v.startswith("torch."):
                    v = v.split(".", 1)[1]
                if hasattr(torch, v):
                    value = getattr(torch, v)
            processed[key] = value
        return cls(**processed)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

class Trident(pyro.nn.PyroModule):
    def __init__(self, config: TridentConfig):
        super().__init__()
        if isinstance(config, dict):
            config = TridentConfig(**config)
        self.config = config
        scaler = self.config.num_reads# * self.config.num_targets
        self.model = poutine.scale(self._model, scale=1.0/scaler)
        self.guide = poutine.scale(self._guide, scale=1.0/scaler)
        self.losses = []
        self.optimizer = None
        self.register_buffer(
            "posterior_gamma",
            torch.empty(0, device=config.device, dtype=config.dtype),
        )
        self.register_buffer(
            "posterior_pi",
            torch.empty(0, device=config.device, dtype=config.dtype),
        )
        self.register_buffer(
            "posterior_chi",
            torch.empty(0, device=config.device, dtype=config.dtype),
        )

    def cache_posterior(self, gamma=None, pi=None, chi=None):
        if gamma is not None:
            self.posterior_gamma = gamma.detach().to(
                device=self.config.device, dtype=self.config.dtype
            )
        if pi is not None:
            self.posterior_pi = pi.detach().to(
                device=self.config.device, dtype=self.config.dtype
            )
        if chi is not None:
            self.posterior_chi = chi.detach().to(
                device=self.config.device, dtype=self.config.dtype
            )

    def _model(self, X_batch, L, r_idx=None):
        cfg = self.config
        R, T, K = cfg.num_reads, cfg.num_targets, cfg.num_latent
        eps = cfg.eps
        a_pi, b_pi = cfg.alpha_pi, cfg.beta_pi
        device, dtype = cfg.device, cfg.dtype

        if r_idx is None:
            r_idx = torch.arange(X_batch.shape[0], device=device)

        with pyro.plate("latent", K):
            chi = pyro.sample(
                "chi",
                dist.Normal(
                    torch.full((K,1), cfg.chi_log_shape_model, device=device, dtype=dtype),
                    torch.full((K,1), cfg.chi_log_scale_model, device=device, dtype=dtype),
                ).to_event(1),
            )
            log_gamma = pyro.sample(
                "gamma",
                dist.Normal(
                    torch.zeros((K, T), device=device, dtype=dtype),
                    torch.ones((K, T), device=device, dtype=dtype),
                ).to_event(1),
            )
            gamma = torch.softmax(log_gamma, dim=1)
            
        with pyro.plate("target", T, dim=-1):
            pi = pyro.sample(
                "pi",
                dist.Beta(
                    torch.tensor(a_pi, device=device, dtype=dtype),
                    torch.tensor(b_pi, device=device, dtype=dtype),
                ),
            )

            phi = _softplus((chi.squeeze(-1) @ gamma).squeeze(0)) + eps
            with poutine.scale(scale=R):
                pyro.sample(
                    "L",
                    dist.Beta(
                        concentration1=cfg.concentration*(pi * phi),
                        concentration0=cfg.concentration * ((1.0 - pi) * phi),
                    ),
                    obs=torch.clamp(L, 1e-5, 1 - 1e-5),
                )

        with pyro.plate("reads", size=R, subsample=r_idx) as r_idx:
            log_eta = pyro.sample(
                "eta",
                dist.Normal(
                    torch.zeros(len(r_idx), K, device=device, dtype=dtype),
                    torch.ones(len(r_idx), K, device=device, dtype=dtype),
                ).to_event(1),
            )
            eta = torch.softmax(log_eta, dim=1)
            m = pyro.deterministic("m", X_batch.sum(1).unsqueeze(-1))
            with poutine.scale(scale=T):
                pyro.sample(
                    "X",
                    dist.NegativeBinomial(m*(eta@gamma), probs=pi).to_event(1),
                    obs=X_batch,
                )

    def _guide(self, X_batch, L, r_idx=None):
        cfg = self.config
        R, T, K = cfg.num_reads, cfg.num_targets, cfg.num_latent
        device, dtype = cfg.device, cfg.dtype
        a_pi, b_pi = cfg.alpha_pi, cfg.beta_pi

        if r_idx is None:
            r_idx = torch.arange(X_batch.shape[0], device=device)

        with pyro.plate("latent", K):
            chi_loc = pyro.param(
                "chi_loc",
                torch.full((K,1), cfg.chi_log_shape_guide, device=device, dtype=dtype),
            )
            chi_scale = pyro.param(
                "chi_scale",
                torch.full((K,1), cfg.chi_log_scale_guide, device=device, dtype=dtype),
                constraint=constraints.positive,
            )
            pyro.sample("chi", dist.Normal(chi_loc, chi_scale).to_event(1))

            lg_loc = pyro.param(
                "log_gamma_loc", torch.zeros((K, T), device=device, dtype=dtype)
            )
            lg_su = pyro.param(
                "log_gamma_su",
                torch.full((K, T), 1, device=device, dtype=dtype),
                constraint=constraints.positive
            )
            pyro.sample("gamma", dist.Normal(lg_loc, lg_su).to_event(1))

        with pyro.plate("target", T, dim=-1):
            psi1 = torch.special.polygamma(1, torch.tensor([a_pi, b_pi], device=device, dtype=dtype))
            pi_logit_sd = cfg.pi_guide_scale_mult * torch.sqrt(psi1.sum())
            pi_loc = pyro.param("pi_loc", torch.zeros(T, device=device, dtype=dtype))
            pi_sd = pyro.param(
                "pi_sd",
                torch.full((T,), pi_logit_sd.item(), device=device, dtype=dtype),
                constraint=constraints.positive,
            )
            pyro.sample(
                "pi",
                dist.TransformedDistribution(
                    dist.Normal(pi_loc, pi_sd),
                    [dist.transforms.SigmoidTransform()],
                ),
            )

        with pyro.plate("reads", size=R, subsample=r_idx) as r_idx:
            le_loc = pyro.param(
                "log_eta_loc", torch.zeros(R, K, device=device, dtype=dtype)
            )[r_idx]
            le_su = pyro.param(
                "log_eta_su",
                torch.full((R, K), 1, device=device, dtype=dtype),
                constraint=constraints.positive
            )[r_idx]
            pyro.sample("eta", dist.Normal(le_loc, le_su).to_event(1))
