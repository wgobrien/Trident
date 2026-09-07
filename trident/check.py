import argparse
import os
import shutil
    
from .global_utils import RunResourceTracker, get_basename_from_reads, TRIDENT_GLOBALS, get_logger
from .tools.utils import checkpoint
from .tools.model import TridentConfig
from .tools.data import load_adata, load_accession_adata, taxutils
from .tools.evals import _get_read_probabilities, reestimate, var_marginal, get_posteriors
from .tools.align_tl import align, cluster_consensus, blast_contigs
from .plot.align_pl import coverage
from .plot.post import read_cluster

logger = get_logger(__name__)

def _accession_skip_reason(acc_adata):
    """Return a reason when accession validation has no usable data."""
    if acc_adata is None:
        return "accession data could not be loaded"
    if acc_adata.shape[0] == 0:
        return "accession data contains no reads with target hits"
    if acc_adata.shape[1] == 0:
        return "accession data contains no target accessions"
    if "accession" not in acc_adata.var.columns:
        return "accession data has no accession targets"

    accessions = acc_adata.var["accession"].dropna().astype(str).str.strip()
    if accessions.empty or not accessions.ne("").any():
        return "accession data has no accession targets"
    return None

def _read_cluster_skip_reason(acc_adata):
    """Return a reason when validation cannot assemble read clusters."""
    clusters = acc_adata.obs.get("read_cluster")
    if clusters is not None and clusters.notna().any():
        return None
    if acc_adata.shape[0] < 5:
        return "fewer than five accession-fit reads are available for read clustering"
    return "read clustering produced no usable clusters"

def check(args):
    """
    Validate accession-level results by aligning reads, computing coverage,
    and generating consensus contigs with BLAST verification.

    Requires that `trident accession` has been run prior to invocation.
    """
    with RunResourceTracker("check", args, logger=logger) as tracker:
        if shutil.which("megahit") is None:
            raise EnvironmentError("megahit not found. Install with: conda install -c bioconda megahit=1.2.9 -y")
        config = TridentConfig(
            reads=args.reads,
            pair=args.pair,
            library=args.library,
            threads=args.threads,
            database="",
            pathogen_dict=getattr(args, "pathogen_dict", None),
            sqlite=not args.low_memory,
            seed=args.seed,
            **({"device": args.device} if args.device else {}),
        )
        basename = get_basename_from_reads(args.reads)
        adata_path = os.path.join(TRIDENT_GLOBALS["save_folder"], basename, basename + ".trident.h5ad.gz")
        acc_adata_path = os.path.join(TRIDENT_GLOBALS["save_folder"], basename, basename + ".accessions.trident.h5ad.gz")
        if not os.path.exists(adata_path) or not os.path.exists(acc_adata_path):
            raise FileNotFoundError(
                "Please run `trident accession` before `trident check`."
            )
        fig_path = os.path.join(args.output, "figs") if args.save_figs else None
        if args.save_figs:
            os.makedirs(fig_path, exist_ok=True)
        sqlite = not getattr(args, "low_memory", False)
        taxonomic_data = taxutils(config)
        acc_adata = load_accession_adata(config=config, taxonomic_data=taxonomic_data, rebuild=False)
        skip_reason = _accession_skip_reason(acc_adata)
        if skip_reason is not None:
            logger.warning("Skipping check for %s: %s.", basename, skip_reason)
            tracker.mark(skipped=True, skip_reason=skip_reason)
            return None

        get_posteriors(acc_adata, num_samples=1000)
        taxonomic_data.load_a2t(list(acc_adata.var.accession.values), low_memory=not sqlite)
        acc_adata.var["taxid"] = acc_adata.var["accession"].map(taxonomic_data.a2t).astype(str)
        acc_adata.var["name"] = acc_adata.var["accession"] + " ("+ acc_adata.var["taxid"].map(taxonomic_data.names) + ")"
        _get_read_probabilities(acc_adata)
        reestimate(acc_adata)
        var_marginal(acc_adata, taxonomic_data)
        read_cluster(
            acc_adata, n_reads_subsample=1000, top_n_var=30,
        )
        skip_reason = _read_cluster_skip_reason(acc_adata)
        if skip_reason is not None:
            logger.warning("Skipping check for %s: %s.", basename, skip_reason)
            tracker.mark(skipped=True, skip_reason=skip_reason)
            return None

        with tracker.step(
            "alignment",
            reads=int(acc_adata.shape[0]),
            accessions=int(acc_adata.shape[1]),
        ) as step:
            alignment_path = align(acc_adata, output_path=args.output)
            step.mark(alignment_created=alignment_path is not None)
        if alignment_path is None:
            skip_reason = "none of the fitted accession targets were found in the reference library"
            logger.warning("Skipping check for %s: %s.", basename, skip_reason)
            tracker.mark(skipped=True, skip_reason=skip_reason)
            return None

        coverage_stats = coverage(acc_adata, top_n=4, save_path=os.path.join(fig_path, "coverage_by_reads.png") if args.save_figs else None,)
        if coverage_stats.empty:
            skip_reason = "the alignment contains no hits to fitted accession targets"
            logger.warning("Skipping check for %s: %s.", basename, skip_reason)
            tracker.mark(skipped=True, skip_reason=skip_reason)
            return None
        coverage(acc_adata, top_n=4, by="marginal_proba", save_path=os.path.join(fig_path, "coverage_by_proba.png") if args.save_figs else None,)
        with tracker.step(
            "assembly",
            read_clusters=int(acc_adata.obs["read_cluster"].nunique()) if "read_cluster" in acc_adata.obs.columns else None,
        ):
            cluster_consensus(acc_adata, output_path=args.output)
        contigs = blast_contigs(acc_adata)
        logger.info("Saving anndata...")
        checkpoint(acc_adata)
        logger.info("Check complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Trident accession results via alignment, coverage, and BLAST."
    )
    parser.add_argument("--reads", required=True, help="Path to reads file.")
    parser.add_argument("--library", required=True, help="Path to library file.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--pair", default=None, help="Path to paired reads file.")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads for parallel operations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Compute device, e.g. 'cuda' or 'cpu' (auto-detected by default)")
    parser.add_argument("--low-memory", action="store_true", help="Avoid building SQLite db; scans compressed files instead (slower but uses less disk/memory)")
    parser.add_argument("--save-figs", action="store_true", help="Save figures to [output]/figs/")

    args = parser.parse_args()
    if args.save_figs is True:
        os.makedirs(os.path.join(args.output, "figs"), exist_ok=True)
    check(args)
