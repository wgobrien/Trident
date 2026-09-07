import numpy as np
import pandas as pd
import pysam
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from trident.global_utils import get_logger, is_notebook

logger = get_logger(__name__)

def coverage(acc_adata, by=None, top_n=10, save_path=None):
    vcs = pd.DataFrame(acc_adata.obs["accession_reestimation"].value_counts())
    
    samfile = pysam.AlignmentFile(acc_adata.uns["trident"]["alignment"], "r")
    references = set(samfile.references) & set(vcs.index)
    ref_lengths = {ref: samfile.get_reference_length(ref) for ref in references}
    
    cov_arrays = {ref: np.zeros(ref_lengths[ref], dtype=int) for ref in references}
    
    for read in samfile:
        if not read.is_unmapped and read.reference_name in references:
            ref = read.reference_name
            for pos in range(read.reference_start, read.reference_end):
                if pos < len(cov_arrays[ref]):
                    cov_arrays[ref][pos] += 1
    
    samfile.close()
    
    stats = []
    for ref in references:
        cov = cov_arrays[ref]
        stats.append({
            'accession': ref,
            'pct_coverage': 100 * np.sum(cov > 0) / len(cov),
            'mean_coverage': np.mean(cov),
            'max_coverage': np.max(cov),
            'length': len(cov)
        })
    
    stats_df = (
        pd.DataFrame(
            stats,
            columns=[
                "accession",
                "pct_coverage",
                "mean_coverage",
                "max_coverage",
                "length",
            ],
        )
        .merge(vcs, left_on='accession', right_index=True, how='left')
        .merge(acc_adata.var, left_on='accession', right_on='accession', how='left')
    )
    
    by = "count" if by is None else by
    to_plot = stats_df.sort_values(by, ascending=False)
    n = min(top_n, len(to_plot))
    if n == 0:
        logger.warning("No accessions found")
        return stats_df
    
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", n)
    
    for i, (_, row) in enumerate(to_plot.head(n).iterrows()):
        ref = row['accession']
        cov = cov_arrays[ref]
        positions = np.arange(len(cov))
        
        axes[i].fill_between(positions, cov, alpha=0.3, color=colors[i])
        axes[i].plot(positions, cov, linewidth=1.5, color=colors[i])
        axes[i].axhline(y=row['mean_coverage'], color='red', linestyle='--',
                        label=f'Mean: {row["mean_coverage"]:.1f}x', linewidth=2)
        
        name = row['name'] if pd.notna(row.get('name')) else ref
        title = f'{name}\nCov: {row["pct_coverage"]:.1f}% | {row["length"]:,} bp | Reads: {row["count"]}'
        if by is not None and by != 'count':
            title += f' | {by}: {row[by]:.4f}'
        axes[i].set_title(title, fontsize=11)
        axes[i].set_xlabel('Position (bp)')
        axes[i].set_ylabel('Coverage')
        axes[i].legend(loc='upper right', fontsize=8)
        sns.despine(ax=axes[i])
    
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)
    return stats_df


def cluster_assembly(
    acc_adata,
    taxonomic_data=None,
    top_n=4,
    best_per_cluster=True,
    figsize_width=10,
    row_height=1.0,
    save_path=None,
    ax=None,
):
    """
    Parse assembled contigs FASTA and plot BLAST results colored by read cluster.

    Parameters
    ----------
    acc_adata : AnnData
        Must have obs columns 'read_cluster' and 'read_cluster_color'
        and uns['trident']['contigs'] pointing to cluster_contigs.fasta
    taxonomic_data : object, optional
        If provided, checks taxonomic_data.a2t for modeled accessions
    top_n : int
        Number of top clusters to plot (by contig count)
    best_per_cluster : bool
        If True, keep only the best BLAST hit per cluster (by bitscore)
    figsize_width : float
        Figure width
    row_height : float
        Height per row
    save_path : str, optional
        If provided, saves figure to this path
    ax : matplotlib Axes, optional
        If provided, plot into this axes

    Returns
    -------
    df : pd.DataFrame
        Parsed and filtered contig dataframe
    """
    assembly_path = acc_adata.uns["trident"]["contigs"]
    records = []
    with open(assembly_path) as f:
        for line in f:
            if line.startswith(">"):
                header = line.strip().lstrip(">")
                fields = header.split()
                contig_name = fields[0]
                cluster_idx = int(contig_name.split("_")[1])

                meta = {}
                for field in fields[1:]:
                    if "=" in field:
                        k, v = field.split("=", 1)
                        meta[k] = v

                records.append({
                    "contig": contig_name,
                    "cluster": cluster_idx,
                    "accession": meta.get("blast_top_acc", ""),
                    "organism": meta.get("blast_top_org", "").replace("_", " "),
                    "pct_id": float(meta.get("blast_pct_id", 0)),
                    "qcov": float(meta.get("blast_qcov", 0)),
                    "length": int(meta.get("len", 0)),
                    "bitscore": float(meta.get("blast_bitscore", 0)),
                })

    df = pd.DataFrame(records)

    top_clusters = df["cluster"].value_counts().head(top_n).index.tolist()
    df = df[df["cluster"].isin(top_clusters)].copy()

    if best_per_cluster:
        df = df.sort_values("bitscore", ascending=False).drop_duplicates("cluster", keep="first")

    cluster_color_map = (
        acc_adata.obs[["read_cluster", "read_cluster_color"]]
        .drop_duplicates()
        .set_index("read_cluster")["read_cluster_color"]
        .to_dict()
    )
    df["color"] = df["cluster"].map(cluster_color_map).fillna("#999999")

    if taxonomic_data is not None:
        modeled_accs = set(
            acc_adata.var.accession.str.replace(r"\.\d{1,2}$", "", regex=True)
        )
        df["modeled"] = df["accession"].isin(modeled_accs) | df["accession"].isin(taxonomic_data.a2t)
    else:
        df["modeled"] = True

    df = df.sort_values("cluster").reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(figsize_width, max(4, len(df) * row_height)))

    y = range(len(df))
    ax.barh(y, df["qcov"], color=df["color"], edgecolor="white", height=0.7)
    ax.barh(y, df["pct_id"], color=df["color"], edgecolor="black",
            linewidth=0.5, height=0.35, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([f"C{c}" for c in df["cluster"]], fontsize=9)
    ax.set_xlabel("Percent (%)")
    ax.set_xlim(0, 105)
    ax.set_title(f"Top Clusters — Best Assembly BLAST Hit", fontsize=11)
    ax.invert_yaxis()

    for i, row in df.iterrows():
        modeled_tag = "" if row["modeled"] else "  [NOT MODELED]"
        org_short = row["organism"][:45]
        line1 = f"{row['accession']} | {org_short}{modeled_tag}"
        line2 = f"qcov={row['qcov']}%  id={row['pct_id']}%  len={row['length']}"

        ax.text(1, i-.11, line1, va="top", ha="left",
                fontsize=7, family="monospace", color="0.")
        ax.text(1, i+.05, line2, va="top", ha="left",
                fontsize=6.5, family="monospace", color="0.")

    ax.annotate("Wide bar = query coverage, narrow bar = % identity",
                xy=(0.99, 0.01), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8, fontstyle="italic", color="gray")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)

    return df
