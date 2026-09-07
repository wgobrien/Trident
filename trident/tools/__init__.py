# trident.tools.__init__.py

from .train import train_trident, plot_loss
from .utils import (
    load_model,
    save_model,
    to_treport,
    clear_trident,
    checkpoint,
)
from .model import TridentConfig
from .sample import sample_X, sample_L, sample, get_learned_params
from .data import (
    load_adata,
    get_read_pairs,
    accession_counts,
    load_accession_adata,
    kmer_coverage,
    TridentDataset,
    taxutils,
)
from .evals import (
    get_posteriors,
    cluster_lca,
    reestimate,
    _get_var_label,
    var_marginal,
    _get_read_probabilities,
    pi_var,
)
from .align_tl import align, compare_references, cluster_consensus, blast_contigs

__all__ = [
    "taxutils",
    "list_well_ids",
    "load_from_dirs",
    "load_adata",
    "TridentConfig",
    "train_trident",
    "load_model",
    "save_model",
    "to_treport",
    "cluster_lca",
    "get_posteriors",
    "reestimate",
    "sample",
    "plot_loss",
    "kmer_coverage",
    "accession_counts",
    "load_accession_adata",
    "compute_mean_count",
    "TridentDataset",
    "align",
    "compare_references",
    "clear_trident",
    "sort_var_by_pi_variance",
]
