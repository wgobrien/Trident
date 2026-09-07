# post.py

import numpy as np
import pyro
import torch
import pyro.distributions as dist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, to_hex
import seaborn as sns
import logging, re

from pyro.infer import Predictive
from trident.global_utils import get_logger, is_notebook, _get_var_label
from trident.tools.sample import sample_X, sample_L, downsample
from trident.tools.evals import get_posteriors, _get_read_probabilities, pi_var
from trident.tools.utils import load_model, get_posterior_gamma
from trident.tools.data import to_torch

logger = get_logger(__name__)

def _gamma_df(adata):
    model = load_model(adata)
    gamma = get_posterior_gamma(adata, model=model).detach().cpu().numpy().T
    gamma_cols = [f"gamma_{k}" for k in range(gamma.shape[1])]
    return pd.DataFrame(gamma, index=adata.var.index, columns=gamma_cols)

def _hierarchical_cluster(Xn, method="average", metric="correlation",
                          mode="silhouette", kmax=5):
    method_lower = method.lower()
    euclid_only = {"ward", "centroid", "median"}
    metric_eff = "euclidean" if method_lower in euclid_only else metric

    Z = linkage(Xn, method=method_lower, metric=metric_eff)

    if mode == "elbow":
        h = Z[:, 2]
        d = np.diff(h)
        if len(d) == 0:
            t = h[0] if len(h) else 0.0
        else:
            i = int(np.argmax(d))
            t = (h[i] + (h[i + 1] if i + 1 < len(h) else h[i] * 1.001)) / 2.0
        labels = fcluster(Z, t=t, criterion="distance")
    elif mode == "silhouette":
        n = Xn.shape[0]
        ks = range(2, max(3, min(kmax, n - 1)) + 1)
        best_s, labels = -1.0, None
        for k in ks:
            labs = fcluster(Z, t=k, criterion="maxclust")
            n_unique = len(np.unique(labs))
            if n_unique < 2 or n_unique >= n:
                continue
            s = silhouette_score(Xn, labs, metric="euclidean")
            if s > best_s:
                best_s, labels = s, labs
        if labels is None:
            labels = np.ones(n, dtype=int)
    else:
        raise ValueError("mode must be 'elbow' or 'silhouette'")

    return Z, labels

def read_cluster(
    adata,
    method="average",
    metric="correlation",
    mode="silhouette",
    kmax=5,
    standardize=None,
    cmap="viridis",
    n_reads_subsample=1000,
    num_samples=10,
    top_n_var=None,
    save_path=None,
):
    if adata.shape[0] < 5:
        logger.warn("Low number of reads. Skipping read clustering.")
        return
    # Get posterior read probabilities
    if "probas" not in adata.layers:
        probs_rt = _get_read_probabilities(adata, num_samples=num_samples)  # [R, T]
    else:
        probs_rt = adata.layers["probas"]

    n_reads = probs_rt.shape[0]
    if n_reads_subsample is None or n_reads_subsample >= n_reads:
        idx = np.arange(n_reads)
        probs_subsample = probs_rt
    else:
        probs_subsample, idx = downsample(torch.tensor(probs_rt), n_reads_subsample)
        probs_subsample = probs_subsample.numpy()
        idx = idx.numpy()

    col_mask = (probs_subsample > 0).any(axis=0)
    probs_subsample = probs_subsample[:, col_mask]
    probs_full = probs_rt[:, col_mask]

    X_full = pd.DataFrame(
        probs_full,
        index=adata.obs.index,
        columns=np.arange(probs_full.shape[1]),
    )

    if standardize == "zscore_cols":
        mu = X_full.mean(axis=0)
        sigma = X_full.std(axis=0, ddof=0) + 1e-12
        Xn_full = (X_full - mu) / sigma
    elif standardize == "zscore_rows":
        X = X_full.to_numpy(dtype=float)
        row_mu = X.mean(axis=1, keepdims=True)
        row_sigma = X.std(axis=1, ddof=0, keepdims=True) + 1e-12
        Xn_full = pd.DataFrame(
            (X - row_mu) / row_sigma,
            index=X_full.index,
            columns=X_full.columns,
        )
    else:
        Xn_full = X_full

    Xn_sub = Xn_full.iloc[idx, :]

    Z, labels_sub = _hierarchical_cluster(
        Xn_sub.to_numpy(),
        method=method,
        metric=metric,
        mode=mode,
        kmax=kmax,
    )

    clusters_sub = pd.Series(
        labels_sub,
        index=Xn_sub.index,
        name="read_cluster",
    )

    cluster_ids = np.sort(np.unique(labels_sub))
    centers = []
    for cid in cluster_ids:
        centers.append(Xn_sub[clusters_sub == cid].to_numpy().mean(axis=0))
    centers = np.vstack(centers)

    Xn_full_np = Xn_full.to_numpy()  # [N, D]
    C = centers  # [K, D]

    if metric == "correlation":
        X_c = Xn_full_np - Xn_full_np.mean(axis=1, keepdims=True)  # [N, D]
        C_c = C - C.mean(axis=1, keepdims=True)  # [K, D]
        X_norm = np.linalg.norm(X_c, axis=1, keepdims=True) + 1e-12  # [N, 1]
        C_norm = np.linalg.norm(C_c, axis=1, keepdims=True) + 1e-12  # [K, 1]
        corr = (X_c / X_norm) @ (C_c / C_norm).T  # [N, K]
        dist_matrix = 1.0 - corr
    elif metric == "cosine":
        X_norm = np.linalg.norm(Xn_full_np, axis=1, keepdims=True) + 1e-12
        C_norm = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        cos_sim = (Xn_full_np / X_norm) @ (C / C_norm).T  # [N, K]
        dist_matrix = 1.0 - cos_sim
    else:
        X_sq = np.sum(Xn_full_np ** 2, axis=1, keepdims=True)  # [N, 1]
        C_sq = np.sum(C ** 2, axis=1, keepdims=True).T  # [1, K]
        dist_matrix = X_sq + C_sq - 2.0 * (Xn_full_np @ C.T)  # [N, K]

    all_labels = cluster_ids[np.argmin(dist_matrix, axis=1)]

    idx_map = {i: pos for pos, i in enumerate(idx)}
    for i, pos in idx_map.items():
        all_labels[i] = labels_sub[pos]

    clusters_full = pd.Series(
        all_labels,
        index=adata.obs.index,
        name="read_cluster",
    )

    unique_clusters = sorted(clusters_full.unique())
    pal = sns.color_palette("tab20", len(unique_clusters))
    color_map = {c: to_hex(pal[i % len(pal)]) for i, c in enumerate(unique_clusters)}

    adata.obs["read_cluster"] = clusters_full
    adata.obs["read_cluster_color"] = clusters_full.map(color_map).astype("category")

    if "name" in adata.var.columns:
        col_labels = adata.var["name"].values[col_mask]
    elif "accession" in adata.var.columns:
        col_labels = adata.var["accession"].values[col_mask]
    else:
        col_labels = adata.var_names[col_mask]

    Xplot = Xn_sub.copy()
    Xplot.columns = col_labels

    seen = {}
    unique_cols = []
    for c in Xplot.columns:
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique_cols.append(c)
    Xplot.columns = unique_cols

    use_full_linkage = True
    if top_n_var is not None and top_n_var < Xplot.shape[1]:
        raw_sub = pd.DataFrame(
            probs_subsample,
            index=Xn_sub.index,
            columns=Xplot.columns,  # use the deduplicated column labels
        )
        var_totals = raw_sub.sum(axis=0)
        top_vars = var_totals.nlargest(top_n_var).index
        Xplot = Xplot[top_vars]
        use_full_linkage = False  # linkage Z was built on all columns, can't reuse

    row_color_series = pd.Series(
        [color_map[c] for c in clusters_sub.values],
        index=Xn_sub.index,
        name="cluster",
    )

    dominant_col = Xplot.idxmax(axis=1)
    max_prob = Xplot.max(axis=1)
    sort_df = pd.DataFrame({
        'cluster': clusters_sub,
        'dominant': dominant_col,
        'max_prob': max_prob,
    }, index=Xplot.index)
    sort_order = sort_df.sort_values(['cluster', 'dominant', 'max_prob'], ascending=[True, True, False]).index

    Xplot = Xplot.loc[sort_order]
    row_color_series = row_color_series.loc[sort_order]

    cg = sns.clustermap(
        Xplot.T,
        col_linkage=None,
        col_cluster=False,
        row_cluster=True,
        cmap=cmap,
        col_colors=row_color_series,
        figsize=(10, 10),
        dendrogram_ratio=(0.05, 0.05),
        colors_ratio=0.03,
        cbar_pos=None,
    )

    # Add colorbar manually in a safe spot
    cbar_ax = cg.fig.add_axes([-0.03, 0.5 - 0.075, 0.015, 0.15])
    cg.fig.colorbar(cg.ax_heatmap.collections[0], cax=cbar_ax, orientation="vertical")
    cbar_ax.set_title("Probability", fontsize=10)
    cbar_ax.tick_params(labelsize=7)

    tora = "Accession" if adata.uns["trident"]["is_acc"] else "Taxon"
    cg.fig.suptitle(
        f"Read Cluster Heatmap — {tora}-Level Probabilities\n(n={n_reads_subsample:,} sampled reads)",
        fontsize=18, y=1,
    )

    ax = cg.ax_heatmap
    # Row gridlines (between accessions/taxa)
    for i in range(Xplot.T.shape[0]):
        ax.axhline(i, color='black', linewidth=0.5)

    # Vertical lines at cluster boundaries
    sorted_clusters = clusters_sub.loc[sort_order]
    boundaries = np.where(sorted_clusters.values[:-1] != sorted_clusters.values[1:])[0] + 1
    for b in boundaries:
        ax.axvline(b, color='red', linewidth=.5)

    logger.info('Added: `.obs["read_cluster"]` and `.obs["read_cluster_color"]`')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close()

def var_cluster(
    adata,
    method="ward",
    metric="correlation",
    mode="silhouette",
    kmax=5,
    standardize="zscore_cols",
    cmap=None,
    save_path=None
):
    var_label = _get_var_label(adata)
    cluster_col = f"{var_label}_cluster"
    cluster_color_col = f"{var_label}_cluster_color"

    gamma_df = _gamma_df(adata)
    gamma_cols = list(gamma_df.columns)
    name_label = "name" if "name" in adata.var.columns else var_label
    
    df = pd.concat([gamma_df, adata.var[[name_label]]], axis=1)
    X = gamma_df.copy()

    if standardize == "zscore_cols":
        Xn = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
    elif standardize == "zscore_rows":
        Xn = ((X.T - X.mean(axis=1)).T) / (
            X.std(axis=1, ddof=0) + 1e-12
        ).to_numpy()[:, None]
    else:
        Xn = X

    method_lower = method.lower()
    euclid_only = {"ward", "centroid", "median"}
    metric_eff = "euclidean" if method_lower in euclid_only else metric

    Z = linkage(Xn.values, method=method_lower, metric=metric_eff)

    if mode == "elbow":
        h = Z[:, 2]
        d = np.diff(h)
        if len(d) == 0:
            t = h[0] if len(h) else 0.0
        else:
            i = int(np.argmax(d))
            t = (h[i] + (h[i + 1] if i + 1 < len(h) else h[i] * 1.001)) / 2.0
        labels = fcluster(Z, t=t, criterion="distance")
    elif mode == "silhouette":
        n = Xn.shape[0]
        ks = range(2, max(3, min(kmax, n - 1)) + 1)
        best_s, labels = -1.0, None
        for k in ks:
            labs = fcluster(Z, t=k, criterion="maxclust")
            n_unique = len(np.unique(labs))
            if n_unique < 2 or n_unique >= n:
                continue
            s = silhouette_score(Xn, labs, metric="euclidean")
            if s > best_s:
                best_s, labels = s, labs
        if labels is None:
            labels = np.ones(n, dtype=int)
    else:
        raise ValueError("mode must be 'elbow' or 'silhouette'")

    clusters = pd.Series(labels, index=Xn.index, name=cluster_col)

    pal = sns.color_palette("tab20", int(clusters.max()))
    color_map = {
        c: to_hex(pal[(c - 1) % len(pal)])
        for c in range(1, int(clusters.max()) + 1)
    }
    adata.var[cluster_col] = clusters.reindex(adata.var.index)
    adata.var[cluster_color_col] = adata.var[cluster_col].map(color_map).astype("category")

    Xplot = Xn.copy()
    name_series = df[name_label]
    fallback = pd.Series(Xplot.index.astype(str), index=Xplot.index)
    Xplot.index = name_series.fillna(fallback)

    row_colors = [color_map[c] for c in clusters.reindex(Xn.index).to_numpy()]

    Xplot = Xplot.filter(regex=r"^gamma_\d+$")
    def extract_num(c):
        return int(re.search(r"\d+", c).group())
    Xplot.columns = [f"Var Factor {extract_num(c)+1}" for c in Xplot.columns]
    
    cg = sns.clustermap(
        Xplot,
        row_linkage=Z,
        col_cluster=True,
        cmap=cmap,
        row_colors=row_colors,
    )

    logger.info(f'Added: `.var["{cluster_col}"]` and `.var["{cluster_color_col}"]`')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close()

def transition(adata, quantile=.9, n_vars=10, figsize=(12, 10), save_path=None):
    pairs = pd.DataFrame({
        "kraken_label": adata.obs["kraken_label"].values,
        "reestimation": adata.obs["reestimation"].values
    })
    CT = pd.crosstab(
        index=pairs["kraken_label"],
        columns=pairs["reestimation"]
    )
    all_labels = sorted(
        set(pairs["kraken_label"].unique()) | 
        set(pairs["reestimation"].unique()) | 
        {0, 1}
    )
    
    CT = CT.reindex(
        index=all_labels,
        columns=all_labels,
        fill_value=0
    )
    name_map = dict(zip(adata.var["taxid"], adata.var["name"]))
    if adata.uns["trident"]["is_acc"]:
        taxid_name = adata.var["name"].str.extract(r"\((.+?)\)")[0]
        name_map = dict(zip(
            adata.var["taxid"], adata.var["taxid"].astype(str) + " (" + taxid_name + ")"
        ))
        name_map[0] = "Unclassified (0)"
        name_map[1] = "Root (1)"
    
    CT.columns = CT.columns.map(lambda x: name_map.get(x, str(x)))
    CT.index   = CT.index.map(lambda x: name_map.get(x, str(x)))
    
    row_sums = CT.sum(axis=1)
    col_sums = CT.sum(axis=0)
    total_activity = row_sums.add(col_sums, fill_value=0)

    keep = set(total_activity.index)
    if len(keep) > max(n_vars or 0, 3):
        q_set = set()
        n_set = set()
        if quantile is not None:
            threshold = total_activity.quantile(quantile)
            q_set = set(total_activity[total_activity >= threshold].index)
        if n_vars is not None:
            n_set = set(total_activity.nlargest(n_vars).index)
        union = q_set | n_set
        if union:
            keep = union

    active_taxa = total_activity.loc[total_activity.index.isin(keep)].sort_values(ascending=False).index
    CT = CT.reindex(index=active_taxa, columns=active_taxa, fill_value=0)

    if CT.shape[0] == 0:
        logger.warn("Transition matrix detected no significant reestimations.")
        return

    CT_float = CT.replace(0, 1e-6)
    plt.figure(figsize=figsize)
    cmap = sns.light_palette("#69d", as_cmap=True)
    sns.heatmap(
        CT_float,
        annot=CT,
        fmt="d",
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=CT.values.max())
    )
    plt.xticks(rotation=45, ha="right")
    is_acc = " (Accessions)" if adata.uns["trident"]["is_acc"] else ""
    plt.title("Re-estimation Read Movements" + is_acc, fontsize=19)
    plt.xlabel("TRIDENT Re-estimation")
    plt.ylabel("Kraken Estimation")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close()

def var_proba(adata, figsize=(10, 4), top_n=30, save_path=None):
    var_label = _get_var_label(adata)

    name = "name" if "name" in adata.var.columns else var_label
    
    df = adata.var[[name,"marginal_proba"]].sort_values("marginal_proba", ascending=False).head(top_n)

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df,
        x="marginal_proba",
        y=name,
        hue=name,
        dodge=False,
        legend=False,
        palette="viridis"
    )

    ax.set_xlabel("Proba", fontsize=12)
    ax.set_ylabel("")
    title = "Highest accession marginal probabilities" if var_label == "accession" else "Highest taxa marginal probabilities"
    ax.set_title(title, fontsize=13, pad=10)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close()


def pi_discordance(adata, top_n=30, taxa=None, n_samples=1000, figsize=None, save_path=None):
    var_label = _get_var_label(adata)
    name_col = "name" if "name" in adata.var.columns else var_label
    load_model(adata)
    loc = pyro.param("pi_loc").detach()
    sd = pyro.param("pi_sd").detach()
    pi_samples = dist.TransformedDistribution(
        dist.Normal(loc, sd),
        [dist.transforms.SigmoidTransform()],
    ).sample((n_samples,))
    adata.var["pi_var"] = pi_samples.var(dim=0, unbiased=False).cpu().numpy()
    pi_q = torch.quantile(
        pi_samples,
        torch.tensor([0.05, 0.5, 0.95], device=pi_samples.device, dtype=pi_samples.dtype),
        dim=0,
    ).cpu().numpy()
    q_df = pd.DataFrame(
        {"pi_q05": pi_q[0], "pi_q50": pi_q[1], "pi_q95": pi_q[2]},
        index=adata.var.index,
    )
    adata.var["pi_q50"] = q_df["pi_q50"]

    if taxa is not None:
        # Filter to specified taxa in order — match against var index, name, or taxid
        ordered_idx = []
        for t in taxa:
            matches = adata.var.index[
                (adata.var.index == t) |
                (adata.var[name_col] == t) |
                (adata.var.get("taxid", pd.Series(dtype=object)) == t)
            ]
            for m in matches:
                if m not in ordered_idx:
                    ordered_idx.append(m)
        df = adata.var.loc[ordered_idx].copy()
    else:
        ranked_idx = adata.var["pi_var"].to_numpy(dtype=float).argsort()[::-1][:top_n]
        df = adata.var.iloc[ranked_idx].copy().iloc[::-1]

    if len(df) == 0:
        logger.warning("No matching taxa found.")
        return

    df["label"] = df[name_col].fillna(pd.Series(df.index.astype(str), index=df.index))
    df = df.join(q_df, how="left", rsuffix="_dup")
    if figsize is None:
        figsize = (12, max(4, len(df) * 0.35))
    fig, axes = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1.1, 1.4]},
        sharey=True,
    )
    sns.barplot(
        data=df, x="pi_var", y="label",
        hue="pi_var", dodge=False, legend=False,
        palette="mako", ax=axes[0],
    )
    axes[0].set_title("Posterior $\pi$ Variance")
    axes[0].set_xlabel("Var($\pi$)")
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[0].set_ylabel("")
    y = np.arange(df.shape[0])
    lower = df["pi_q50"] - df["pi_q05"]
    upper = df["pi_q95"] - df["pi_q50"]
    axes[1].errorbar(
        df["pi_q50"], y,
        xerr=np.vstack([lower, upper]),
        fmt="o", color="#1f3b4d", ecolor="#7aa6c2",
        elinewidth=2, capsize=3,
    )
    axes[1].scatter(
        df["pi_q50"], y,
        s=50, c=df["pi_var"], cmap="mako",
        edgecolor="black", linewidth=0.3, zorder=3,
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(df["label"])
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_title("Posterior $\pi$ Median and 90% Interval")
    axes[1].set_xlabel("$\pi$")
    axes[1].set_ylabel("")
    title = (
        r"Top accession discordance by posterior $\pi$ variance"
        if var_label == "accession"
        else r"Top taxa discordance by posterior $\pi$ variance"
    )
    fig.suptitle(title, fontsize=14)
    sns.despine(ax=axes[0], left=True, bottom=False)
    sns.despine(ax=axes[1], left=False, bottom=False)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)


def taxid_totals(adata, taxonomic_data, name=None, save_path=None, quantile=.9, n_vars=10):
    names = taxonomic_data.names
    target_taxa = taxonomic_data.target_taxa
    
    labels = adata.obs["kraken_label"].values
    new_labels = adata.obs["reestimation"].values
    a_ids, a_cnt = np.unique(labels, return_counts=True)
    b_ids, b_cnt = np.unique(new_labels, return_counts=True)
    taxids = np.union1d(a_ids, b_ids)
    a_map = dict(zip(a_ids, a_cnt))
    b_map = dict(zip(b_ids, b_cnt))
    A = np.array([a_map.get(t, 0) for t in taxids], dtype=float)
    B = np.array([b_map.get(t, 0) for t in taxids], dtype=float)
    df = pd.DataFrame(
        np.vstack([A, B]).T,
        index=taxids,
        columns=["Kraken", "Trident"], 
    )
    
    taxids_to_plot = [t for t in target_taxa if t in df.index]
    if len(taxids_to_plot) == 0:
        logger.warn("No target taxa found in the data.")
        return
    df = df.loc[taxids_to_plot]
    total = df["Kraken"] + df["Trident"]

    candidates = []
    if quantile is not None:
        threshold = total.quantile(quantile)
        q_mask = total >= threshold
        if q_mask.any():
            candidates.append(q_mask)
    if n_vars is not None:
        top_n = total.nlargest(n_vars).index
        n_mask = df.index.isin(top_n)
        if n_mask.any():
            candidates.append(n_mask)

    if candidates:
        mask = min(candidates, key=lambda m: m.sum())
    else:
        mask = pd.Series(True, index=df.index)

    df = df.loc[mask]
    df.index = [f"{names.get(int(t), str(t))} ({int(t)})" for t in df.index]

    fig_h = max(4, len(df) * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    sns.heatmap(
        df.clip(lower=1e-1),
        annot=True,
        fmt=".0f",
        cmap="coolwarm",
        norm=plt.matplotlib.colors.LogNorm(vmin=1),
        ax=ax,
    )
    is_acc = " (Accessions)" if adata.uns["trident"]["is_acc"] else ""
    title = f"Re-Estimation Totals" + is_acc
    if name is not None: title += f"({name})"
    plt.title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)

def var_factor_density(adata, taxonomic_data=None, save_path=None):
    var_label = _get_var_label(adata)
    cluster_col = f"{var_label}_cluster"
    cluster_color_col = f"{var_label}_cluster_color"
    cluster_lca_col = f"{var_label}_cluster_lca"

    # Find all gamma columns dynamically
    gamma_df = _gamma_df(adata)
    gamma_cols = list(gamma_df.columns)

    # Check which columns exist
    required_cols = [cluster_col, cluster_color_col]
    has_lca = cluster_lca_col in adata.var.columns

    df = pd.concat(
        [
            adata.var[required_cols + ([cluster_lca_col] if has_lca else [])],
            gamma_df,
        ],
        axis=1,
    ).copy()

    df["Var Factor"] = df[gamma_cols].sum(axis=1)

    if has_lca and taxonomic_data is not None:
        df["Cluster | Cluster LCA"] = (
            df[cluster_col].astype(str)
            + " | "
            + df[cluster_lca_col].astype(int).map(taxonomic_data.names)
            + " (" + df[cluster_lca_col].astype(str) + ")"
        )
        hue_col = "Cluster | Cluster LCA"
    else:
        df["Cluster"] = df[cluster_col].astype(str)
        hue_col = "Cluster"

    palette = (
        df.groupby(hue_col)[cluster_color_col]
        .first()
        .to_dict()
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.kdeplot(
        data=df,
        x="Var Factor",
        hue=hue_col,
        palette=palette,
        common_norm=False,
        warn_singular=False,
        ax=ax
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)
