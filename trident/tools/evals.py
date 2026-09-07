# evals.py

import pyro
import torch
import torch.nn.functional as F
import pyro.distributions as dist
import logging
import pandas as pd
import numpy as np
from functools import reduce
from scipy.sparse import issparse

from trident.global_utils import get_logger, _get_var_label, TRIDENT_GLOBALS
from .utils import load_model, save_model, get_posterior_gamma, get_posterior_pi
from .sample import sample
from .data import to_torch

logger = get_logger(__name__)


def _get_reestimate_threshold(adata, default: float = 0.5) -> float:
    """Read the re-estimation uncertainty threshold from serialized config."""
    cfg = adata.uns.get("trident", {}).get("config", {})
    try:
        return float(cfg.get("reestimate_threshold", default))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid reestimate_threshold=%r in adata.uns['trident']['config']; using %.3f",
            cfg.get("reestimate_threshold"),
            default,
        )
        return default

def _get_ancestor_matrix(adata, taxonomic_data):
    """Build binary [T, T] matrix A where A[i, j] = 1 means j is a descendant of i."""
    var_label = _get_var_label(adata)
    taxid_list = list(adata.var[var_label].astype(int) if var_label == "taxid" else adata.var[var_label])
    taxid_to_col = {t: i for i, t in enumerate(taxid_list)}
    T = len(taxid_list)
    cols_set = set(taxid_list)

    A = np.zeros((T, T), dtype=np.float32)
    for parent_taxid in taxid_list:
        subtree = set(taxonomic_data.get_subtree(parent_taxid))
        children = subtree.intersection(cols_set - {parent_taxid})
        parent_idx = taxid_to_col[parent_taxid]
        for c in children:
            A[parent_idx, taxid_to_col[c]] = 1.0

    return A
        
def _get_read_probabilities(adata, taxonomic_data=None, num_samples=1000, chunk_size=30000):
    model = load_model(adata)
    cfg = model.config
    R, T, K = cfg.num_reads, cfg.num_targets, cfg.num_latent
    device, dtype, eps = cfg.device, cfg.dtype, cfg.eps

    gamma = model.posterior_gamma
    pi = model.posterior_pi
        
    out = np.zeros((R, T), dtype=np.float32)
    for start in range(0, R, chunk_size):
        end = min(start + chunk_size, R)
        with torch.no_grad():
            X_chunk = adata.X[start:end]
            X_chunk = torch.tensor(
                X_chunk.toarray() if hasattr(X_chunk, 'toarray') else X_chunk,
                device=device, dtype=dtype,
            )  # [chunk, T]

            eta = sample(
                adata, "eta",
                start=start, end=end,
                n_samples=num_samples, return_mean=True,
            )  # [chunk, K]
            m = X_chunk.sum(1).unsqueeze(-1)
            s = (eta @ gamma) * m  # [chunk, T]
            posterior_rate = (s + X_chunk) * pi  # [chunk, T]
            probs = torch.softmax(posterior_rate, dim=1)
            # probs = posterior_rate / (posterior_rate.sum(dim=1, keepdim=True) + eps)

            out[start:end] = probs.float().cpu().numpy()

    if taxonomic_data is not None:
        A = _get_ancestor_matrix(adata, taxonomic_data)
        for i in range(T):
            children = np.where(A[i] == 1)[0]
            if len(children) > 0:
                n_children = len(children)
                share = out[:, i:i+1] / n_children  # [R, 1]
                out[:, children] += share
                out[:, i] = 0.0
    
    logger.info("Added `probas` to .layers")
    adata.layers["probas"] = out
    return out

def cumulative_proba(adata, tu, inplace=True):
    """Sum marginal_proba over each taxon's subtree (itself + descendants)."""
    by_taxid = adata.var.groupby("taxid").marginal_proba.sum()
    taxids = by_taxid.index.to_numpy()
    p = by_taxid.to_numpy()

    D = np.array([[bool(tu.is_descendent(tj, ti)) for tj in taxids] for ti in taxids])
    D |= np.eye(len(taxids), dtype=bool)

    cum_map = dict(zip(taxids, D @ p))
    result = adata.var.taxid.map(cum_map).fillna(0.0)

    if inplace:
        adata.var["cumulative_proba"] = result
        logger.info("Added `cumulative_proba` to .var")
    return result


def var_marginal(adata, taxonomic_data=None):
    """Store marginal target probabilities and, when possible, taxon subtree sums."""
    var_label = _get_var_label(adata)

    if "probas" not in adata.layers:
        probas = _get_read_probabilities(adata)
    else:
        probas = adata.layers["probas"]
    p_acc = probas.sum(axis=0)

    p_acc_normalized = p_acc / p_acc.sum()

    indices = p_acc_normalized.argsort()[::-1].copy()
    result = dict(zip(
        adata.var[var_label].iloc[indices].values,
        p_acc_normalized[indices]
    ))
    logger.info("Added `marginal_proba` to .var")
    adata.var["marginal_proba"] = adata.var[var_label].map(result)
    if taxonomic_data is not None and "taxid" in adata.var.columns:
        cumulative_proba(adata, taxonomic_data)


def pi_var(adata, n_samples: int = 1000, logit: bool = False):
    """Compute posterior pi variance and store in adata.var['pi_var']."""
    load_model(adata)
    loc = pyro.param("pi_loc").detach()
    sd = pyro.param("pi_sd").detach()
    if logit:
        adata.var["pi_var"] = (sd ** 2).cpu().numpy()
    else:
        pi_samples = dist.TransformedDistribution(
            dist.Normal(loc, sd),
            [dist.transforms.SigmoidTransform()],
        ).sample((n_samples,))
        adata.var["pi_var"] = pi_samples.var(dim=0, unbiased=False).cpu().numpy()

# def false_positive(adata, taxonomic_data=None):
#     var_label = _get_var_label(adata)
#     cluster_col = f"{var_label}_cluster"
#     cluster_lca_col = f"{var_label}_cluster_lca"

#     if taxonomic_data is None:
#         gammas = [f"gamma_{k}" for k in range(pyro.param("chi_loc").shape[0])]
#         df = adata.var[[cluster_col]+gammas].copy()
#         df["Var Factor"] = df[gammas].values.sum(1)

#         stats = (
#             df.groupby(cluster_col)["Var Factor"]
#               .agg(["mean", "var"])
#               .assign(dist=lambda x: np.sqrt(x["mean"]**2 + x["var"]**2))
#         )

#         closest_cluster = stats["dist"].idxmin()
#         fp_cluster = stats.loc[closest_cluster].name
#         if var_label == "taxid":
#             fp_cluster = int(fp_cluster)
#         adata.var["false_positive"] = adata.var[cluster_col] == fp_cluster
#         logger.info(f"Added: `.var[\"false_positive\"]` == {fp_cluster}")
#     else:
#         rank_map = dict(zip(taxonomic_data.nodes["taxon"], taxonomic_data.nodes["rank_idx"]))
#         adata.var["false_positive"] = adata.var[cluster_lca_col].map(rank_map)
#         logger.info(f"Added: `.var[\"false_positive\"]`")

def cluster_lca(adata, taxonomic_data):
    def lca_reduce(ids):
        valid = [t for t in ids if t not in (0, 1)]
        if not valid:
            return 1
        return reduce(taxonomic_data.get_lca, valid)

    if bool(adata.uns["trident"]["is_acc"]) is False:
        out = (
            adata.var[["taxid", "taxid_cluster"]]
            .groupby("taxid_cluster")["taxid"]
            .apply(lambda s: lca_reduce(s.tolist()))
            .astype("category")
            .rename("taxid_cluster_lca")
        )
        adata.var["taxid_cluster_lca"] = adata.var["taxid_cluster"].map(out)
        logger.info("Added: `.var[taxid_cluster_lca]`")
    else:
        out = (
            adata.var[["taxid", "accession_cluster"]]
            .groupby("accession_cluster")["taxid"]
            .apply(lambda s: lca_reduce(s.tolist()))
            .astype("category")
            .rename("accession_cluster_lca")
        )
        adata.var["accession_cluster_lca"] = adata.var["accession_cluster"].map(out)
        logger.info("Added: `.var[accession_cluster_lca]`")


def _fallback_to_taxid_ancestors(
    adata, taxonomic_data, mean_probs, idx, pi, threshold
):
    """Replace rejected argmax taxa with the first passing candidate ancestor."""
    selected_idx = idx.copy()
    selected_proba = mean_probs.max(axis=1)
    rejected = pi[selected_idx] < threshold
    fallback_rows = []

    taxids = adata.var["taxid"].astype(int).to_numpy()
    taxid_to_idx = {taxid: i for i, taxid in enumerate(taxids)}
    parent_map = taxonomic_data.parent

    for read_idx in np.flatnonzero(rejected):
        current = taxids[selected_idx[read_idx]]
        visited = {current}

        while True:
            parent = parent_map.get(current)
            if parent is None or parent in visited:
                break
            visited.add(parent)

            parent_idx = taxid_to_idx.get(parent)
            if parent_idx is not None and pi[parent_idx] >= threshold:
                selected_idx[read_idx] = parent_idx
                rejected[read_idx] = False
                fallback_rows.append(read_idx)
                break
            current = parent

    if fallback_rows:
        fallback_rows = np.asarray(fallback_rows, dtype=int)
        ancestor_matrix = _get_ancestor_matrix(adata, taxonomic_data).astype(bool)
        selected = selected_idx[fallback_rows]
        subtree_mask = ancestor_matrix[selected]
        subtree_mask[np.arange(len(selected)), selected] = True
        selected_proba[fallback_rows] = (
            mean_probs[fallback_rows] * subtree_mask
        ).sum(axis=1)

    return selected_idx, selected_proba, rejected


def reestimate(adata, taxonomic_data=None, num_samples=1000, chunk_size=int(2**11)):
    """Re-estimate read labels, preferring lower (more specific) taxa on the tree."""
    
    var_label = _get_var_label(adata)
    T = adata.shape[1]
    if "probas" not in adata.layers:
        mean_probs = _get_read_probabilities(adata, num_samples=num_samples, chunk_size=chunk_size)
    else:
        mean_probs = adata.layers["probas"]
    
    pi = adata.var["pi"].values        # shape (T,)
    idx = mean_probs.argmax(axis=1)    # shape (n_reads,)
    selected_proba = mean_probs.max(axis=1)
    reestimate_threshold = _get_reestimate_threshold(adata)

    is_acc = bool(adata.uns["trident"]["is_acc"])
    if not is_acc and taxonomic_data is not None:
        idx, selected_proba, rejected = _fallback_to_taxid_ancestors(
            adata, taxonomic_data, mean_probs, idx, pi, reestimate_threshold
        )
    else:
        rejected = pi[idx] < reestimate_threshold
    warn_mask = pd.Series(rejected, index=adata.obs.index)

    index_to_var = dict(zip(
        range(T),
        adata.var[var_label].astype(int) if var_label == "taxid" else adata.var[var_label]
    ))
    if is_acc and taxonomic_data is not None:
        acc_est = pd.Series(idx, index=adata.obs.index).map(index_to_var).astype(object)
        acc_est[warn_mask] = TRIDENT_GLOBALS["UNCLASSIFIED"]
        adata.obs["accession_reestimation"] = acc_est
        adata.obs["accession_proba"] = pd.Series(selected_proba, index=adata.obs.index)
        logger.info("Added `accession_proba` to .obs")
        logger.info("Added `accession_reestimation` to .obs")

        if taxonomic_data.a2t is None:
            taxonomic_data.load_a2t(list(adata.var.accession.values))

        reest = acc_est.copy()
        not_warned = acc_est != TRIDENT_GLOBALS["UNCLASSIFIED"]
        reest[not_warned] = acc_est[not_warned].map(taxonomic_data.a2t).fillna(TRIDENT_GLOBALS["UNMAPPED"])
        adata.obs["reestimation"] = reest.astype(int)
        sentinel_map = {str(v): k.lower() for k, v in TRIDENT_GLOBALS.items() if k in ("UNCLASSIFIED", "UNMAPPED")}
        adata.obs["accession_reestimation"] = adata.obs["accession_reestimation"].map(lambda x: sentinel_map.get(x, x)).astype(str)
        adata.obs["accession_reestimation"].fillna("unmapped")
        logger.info("Added `reestimation` to .obs")
    else:
        estimation_type = "reestimation" if not is_acc else "accession_reestimation"
        mapped = pd.Series(idx, index=adata.obs.index).map(index_to_var)
        if var_label == "taxid":
            mapped = mapped.fillna(TRIDENT_GLOBALS["UNMAPPED"]).astype(int)
        mapped[warn_mask] = TRIDENT_GLOBALS["UNCLASSIFIED"]
        adata.obs[estimation_type] = mapped.fillna(TRIDENT_GLOBALS["UNMAPPED"])
        logger.info(f"Added `{estimation_type}` to .obs")

        proba_type = "taxid_proba" if not is_acc else "accession_proba"
        adata.obs[proba_type] = pd.Series(selected_proba, index=adata.obs.index)
        logger.info(f"Added `{proba_type}` to .obs")


def get_posteriors(adata, num_samples=1000):
    """Compute posterior summaries using only global params (no read chunking needed)."""
    trident = load_model(adata)
    config = trident.config
    eps = config.eps

    with torch.no_grad():
        gamma = sample(adata, "gamma", n_samples=num_samples, return_mean=True)
        pi = sample(adata, "pi", n_samples=num_samples, return_mean=True)
        chi = sample(adata, "chi", n_samples=num_samples, return_mean=True)

    trident.cache_posterior(gamma=gamma, pi=pi, chi=chi)
    gamma_cols = [c for c in adata.var.columns if c.startswith("gamma_")]
    if gamma_cols:
        adata.var.drop(columns=gamma_cols, inplace=True)
    log_phi = (chi.squeeze(-1) @ gamma).squeeze(0) 
    phi = torch.nn.functional.softplus(log_phi) + 1e-8
    adata.var["phi"] = phi.squeeze().detach().cpu().numpy()
    adata.var["pi"] = pi.squeeze().detach().cpu().numpy()
    save_model(adata, trident, overwrite=True)
    logger.info("Added: `.var[\"pi\"]`, `.var[\"phi\"]`; cached posterior gamma on the model")
