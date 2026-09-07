from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
import numpy as np
import pandas as pd
import gzip, os

from trident.global_utils import is_notebook

def kmer_venn(acc_adata, taxonomic_data, accessions, log_scale=True, save_path=None):
    read_kmers = None
    coverage_path = acc_adata.uns["trident"]["L_file"]

    coverage_gz = coverage_path + ".gz"
    if not os.path.exists(coverage_path) and os.path.exists(coverage_gz):
        coverage_read_path = coverage_gz
        opener = gzip.open
    else:
        coverage_read_path = coverage_path
        opener = open

    with opener(coverage_read_path, "rt") as f:
        for line in f:
            if line.startswith("#read_kmers"):
                read_kmers = int(line.strip().split("\t")[1])
                break
                
    cvg = pd.read_csv(coverage_read_path, sep="\t", comment="#")
    cvg["name"] = cvg.taxid.map(taxonomic_data.names)
    
    cvg = acc_adata.var.merge(
        cvg[["accession", "num_kmers", "shared_minimizers"]],
        on="accession"
    )[["accession", "taxid", "L", "seq_len", "shared_minimizers", "num_kmers", "marginal_proba"]]
    cvg = cvg.sort_values("marginal_proba", ascending=False)
    cvg["name"] = cvg["taxid"].map(taxonomic_data.names) + " (" + cvg["taxid"].astype(str) + ")"

    n = len(accessions)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, acc in zip(axes, accessions):
        row = cvg[cvg["accession"] == acc].iloc[0]
        acc_kmers = row["num_kmers"]
        shared = row["shared_minimizers"]
        read_only = read_kmers - shared
        acc_only = acc_kmers - shared
        subsets = (
            (np.log(max(read_only, 1)), np.log(max(acc_only, 1)), np.log(max(shared, 1)))
            if log_scale else (read_only, acc_only, shared)
        )
        v = venn2(
            subsets=subsets,
            set_labels=(f"Read kmers\n({read_kmers:,})", f"Accession kmers\n({int(acc_kmers):,})"),
            ax=ax,
        )
        v.get_label_by_id("10").set_text(f"{read_only:,}")
        v.get_label_by_id("01").set_text(f"{int(acc_only):,}")
        v.get_label_by_id("11").set_text(f"{int(shared):,}\n({shared/acc_kmers:.1%} of acc)")
        ax.set_title(f"{row.get('name', acc)}\n{acc}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)

def kmer_tree(adata, taxonomic_data, root, branches, top_n=None, save_path=None, read_cluster=None, row_idx=None, seed=0):
    subtree_taxids = taxonomic_data.get_subtree(root)

    parent_map = dict(zip(
        taxonomic_data.nodes["taxon"],
        taxonomic_data.nodes["parent"])
    )
    parent_map[1] = None

    rng = np.random.default_rng(seed)
    taxids = adata.var["taxid"].values
    root_col_idx = np.where(taxids == root)[0]

    if row_idx is not None:
        # If row_idx given, check if root is non-zero; if not, find one that is
        if root_col_idx.size > 0:
            root_val = adata[row_idx].X.toarray().ravel()[root_col_idx[0]]
            if root_val == 0:
                # Find all rows where root is non-zero
                root_counts = np.asarray(adata.X[:, root_col_idx[0]].toarray()).ravel()
                nonzero_rows = np.where(root_counts > 0)[0]
                if nonzero_rows.size > 0:
                    row_idx = rng.choice(nonzero_rows)

        row_data = adata[row_idx].X.toarray().ravel()
        read_id = adata.obs.iloc[row_idx]["read_id"] if "read_id" in adata.obs.columns else str(row_idx)
        counts = {int(t): c for t, c in zip(taxids, row_data) if c > 0 and t in subtree_taxids}
    else:
        if read_cluster is None:
            # No row_idx or read_cluster: pick any row where root is non-zero
            if root_col_idx.size > 0:
                root_counts = np.asarray(adata.X[:, root_col_idx[0]].toarray()).ravel()
                nonzero_rows = np.where(root_counts > 0)[0]
                if nonzero_rows.size > 0:
                    row_idx = rng.choice(nonzero_rows)
                else:
                    row_idx = rng.integers(adata.shape[0])
            else:
                row_idx = rng.integers(adata.shape[0])

            row_data = adata[row_idx].X.toarray().ravel()
            read_id = adata.obs.iloc[row_idx]["read_id"] if "read_id" in adata.obs.columns else str(row_idx)
            counts = {int(t): c for t, c in zip(taxids, row_data) if c > 0 and t in subtree_taxids}
        else:
            mask = adata.obs["read_cluster"] == read_cluster
            df = pd.DataFrame(
                adata[mask].X.toarray(),
                index=adata.obs.loc[mask, "read_id"].values,
                columns=adata.var["taxid"].values,
            )
            df = df[[c for c in df.columns if c in subtree_taxids]]

            root_col = [c for c in df.columns if int(c) == root]
            if root_col:
                nonzero_mask = df[root_col[0]] > 0
                if nonzero_mask.any():
                    idx = rng.choice(np.where(nonzero_mask.values)[0])
                else:
                    idx = rng.integers(df.shape[0])
            else:
                idx = rng.integers(df.shape[0])

            row = df.iloc[idx]
            read_id = df.index[idx]
            counts = {int(t): c for t, c in row[row > 0].to_dict().items()}

    def is_descendant(tid):
        cur = tid
        while cur is not None and cur != root:
            cur = parent_map.get(cur)
        return cur == root

    # Filter to only taxids that are descendants of any branch taxid
    branch_set = set()
    for b in branches:
        branch_set.update(taxonomic_data.get_subtree(b))

    filtered_counts = {t: c for t, c in counts.items() if t in branch_set}

    if top_n:
        sorted_tids = sorted(filtered_counts, key=filtered_counts.get, reverse=True)[:top_n]
        filtered_counts = {t: filtered_counts[t] for t in sorted_tids}

    # Build node set
    nodes = {}
    for tid, count in filtered_counts.items():
        if is_descendant(tid):
            nodes[tid] = {
                "count": count,
                "name": taxonomic_data.names.get(tid, str(tid)),
                "parent": parent_map.get(tid),
            }

    # Fill intermediate nodes
    for tid in list(nodes.keys()):
        cur = parent_map.get(tid)
        while cur is not None and cur != root and cur not in nodes:
            if not is_descendant(cur):
                break
            nodes[cur] = {
                "count": filtered_counts.get(cur, counts.get(cur, 0)),
                "name": taxonomic_data.names.get(cur, str(cur)),
                "parent": parent_map.get(cur),
            }
            cur = parent_map.get(cur)

    # Add root
    nodes[root] = {
        "count": counts.get(root, 0),
        "name": taxonomic_data.names.get(root, str(root)), "parent": None,
    }

    # Build children map
    children_map = defaultdict(list)
    for tid, info in nodes.items():
        if tid == root:
            continue
        p = info["parent"]
        if p in nodes or p == root:
            children_map[p].append(tid)

    # Layout
    pos = {}
    leaf_x = [0]

    def assign(tid, depth):
        kids = sorted(children_map.get(tid, []), key=lambda t: -nodes.get(t, {}).get("count", 0))
        if not kids:
            pos[tid] = (leaf_x[0], depth)
            leaf_x[0] += 1
        else:
            for k in kids:
                assign(k, depth + 1)
            xs = [pos[k][0] for k in kids]
            pos[tid] = ((min(xs) + max(xs)) / 2, depth)

    assign(root, 0)

    # Every non-root node inherits the root's count (shown via dotted lines from root)
    root_count = nodes[root]["count"]
    inherited = {root: 0}
    for tid in nodes:
        if tid != root:
            inherited[tid] = root_count

    # Plot
    max_count = max((n["count"] for n in nodes.values()), default=1) or 1
    fig, ax = plt.subplots(figsize=(max(12, len(pos) * 1.8), 7))

    arrow_props_solid = dict(arrowstyle="-|>", color="gray", lw=1.5,
                              mutation_scale=15, shrinkA=5, shrinkB=15)
    for tid, info in nodes.items():
        if tid == root:
            continue
        p = info["parent"] if info["parent"] in pos else root
        if p not in pos or tid not in pos:
            continue
        x0, y0 = pos[p]
        x1, y1 = pos[tid]
        ax.annotate("", xy=(x1, -y1), xytext=(x0, -y0),
                     arrowprops=arrow_props_solid, zorder=1)

    root_x, root_y = pos[root]
    all_x = [pos[t][0] for t in pos]
    x_span = max(all_x) - min(all_x) if len(all_x) > 1 else 1
    bend_offset = x_span * 0.06

    arrow_props_dotted = dict(arrowstyle="-|>", color="coral", lw=1.5,
                               ls="--", alpha=0.6, mutation_scale=15,
                               shrinkA=5, shrinkB=15)
    for tid in nodes:
        if tid == root:
            continue
        if tid not in pos:
            continue
        inh = inherited.get(tid, 0)
        if inh <= 0:
            continue
        x1, y1 = pos[tid]
        if x1 >= root_x:
            bx = max(root_x, x1) + bend_offset
        else:
            bx = min(root_x, x1) - bend_offset
        by = -(root_y + y1) / 2
        # Draw bend as two segments: root->bend (no arrow), bend->child (arrow)
        ax.plot([root_x, bx], [-root_y, by],
                color="coral", lw=1.5, ls="--", alpha=0.6, zorder=0)
        ax.annotate("", xy=(x1, -y1), xytext=(bx, by),
                     arrowprops=arrow_props_dotted, zorder=0)
        edge_label = f"+{int(inh)}"
        ax.text(bx, by, edge_label, fontsize=8, ha="center", va="bottom",
                color="coral", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    for tid, (x, y) in pos.items():
        info = nodes[tid]
        count = info["count"]
        name = info["name"]
        size = 200 + 800 * (count / max_count)
        color = "steelblue" if count > 0 else "lightgray"

        ax.scatter(x, -y, s=size, c=color, edgecolors="white", lw=1.5, zorder=2)

        if tid != root:
            inh = min(inherited[tid], count)
            own_count = count - inh
            count_str = f"{int(count)} = {int(own_count)} + {int(inh)}"
        else:
            count_str = f"{int(count)}"
        label = f"{name}\n({tid})\n{count_str}"
        ax.annotate(label, (x, -y), textcoords="offset points",
                    xytext=(0, -18 - np.sqrt(size) / 3),
                    ha="center", va="top", fontsize=7,
                    fontweight="bold" if count > 0 else "normal")

    ax.set_title(
        f"Taxonomic tree under {taxonomic_data.names.get(root, root)} ({root})\n"
        f"Read: {read_id}",
        fontsize=12,
    )
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close(fig)
