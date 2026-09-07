# ppc.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from pyro import render_model

from trident.tools.sample import sample
from trident.tools.utils import load_model
from trident.tools.data import to_torch
from trident.global_utils import is_notebook

def plate_model(adata, save_path: str=None, guide=False):
    model = load_model(adata)
    idx = np.random.choice(adata.shape[0], min(5, adata.shape[0]), replace=False)

    adata_subset = adata[idx]
    X, L = to_torch(adata_subset)
    
    r_idx = torch.arange(X.shape[0])
    data = (X, L, r_idx)
    fig = render_model(
        model.model,
        model_args=data,
        render_params=True,
        render_distributions=True,
        render_deterministic=True,
        filename=save_path,
    )
    if is_notebook():
        display(fig)
    if guide:
        if save_path is not None: save_path += ".guide"
        fig = render_model(
            model.guide,
            model_args=data,
            render_params=True,
            render_distributions=True,
            render_deterministic=True,
            filename=save_path,
        )
        if is_notebook():
            display(fig)

def plot_ppc(adata, N=1_000, save_path=None):
    rng = np.random.default_rng(adata.uns['trident']['config']['seed'])
    R = adata.shape[0]
    start = rng.integers(0, max(R - N, 0) + 1)
    end = start + min(N, R)
    X, L = to_torch(adata[start:end])
    X_samp, L_samp = sample(adata, ["X", "L"], n_samples=1_000, start=start, end=end)
    idx = rng.choice(X.shape[0], size=min(N, X.shape[0]), replace=False)
    X_obs_sub = torch.log(X[idx] + 1).reshape(-1).detach().cpu().numpy()
    X_pred_sub = torch.log(X_samp[idx] + 1).reshape(-1).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sns.regplot(
        x=X_obs_sub, y=X_pred_sub,
        scatter_kws=dict(s=10, alpha=0.4),
        line_kws=dict(color="red", lw=2),
        ax=axes[0]
    )
    lims = [min(X_obs_sub.min(), X_pred_sub.min()), max(X_obs_sub.max(), X_pred_sub.max())]
    axes[0].plot(lims, lims, "--k", lw=1)
    axes[0].set_title(rf"PPC for $\log(X)$ (N={len(idx)} reads)")
    axes[0].set_xlabel(r"Observed $\log(X+1)$")
    axes[0].set_ylabel(r"Sampled $\log(X+1)$")
    L_obs = torch.clamp(L, 1e-5, 1-1e-5).detach().cpu().numpy()
    L_pred = L_samp.detach().cpu().numpy()
    sns.regplot(
        x=L_obs, y=L_pred, ci=None,
        scatter_kws=dict(s=10, alpha=0.5),
        line_kws=dict(color="red", lw=2),
        ax=axes[1]
    )
    lims = [0, max(L_obs.max(), L_pred.max())]
    axes[1].plot(lims, lims, "--k", lw=1)
    axes[1].set_title("PPC for L")
    axes[1].set_xlabel("Observed L")
    axes[1].set_ylabel("Sampled L")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)
