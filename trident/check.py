import argparse
import os
import shutil
    
from .global_utils import RunResourceTracker, get_basename_from_reads, TRIDENT_GLOBALS, get_logger
from .tools.utils import checkpoint
from .tools.model import TridentConfig
from .tools.data import load_accession_adata
from .tools.align_tl import align, cluster_consensus, blast_contigs
from .plot.align_pl import coverage

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


def _precomputed_validation_skip_reason(acc_adata):
    """Return a reason when saved accession results lack validation inputs."""
    missing = []
    if "marginal_proba" not in acc_adata.var.columns:
        missing.append('.var["marginal_proba"]')
    if "accession_reestimation" not in acc_adata.obs.columns:
        missing.append('.obs["accession_reestimation"]')
    if "read_cluster" not in acc_adata.obs.columns:
        missing.append('.obs["read_cluster"]')

    if missing:
        fields = ", ".join(missing)
        return (
            f"saved accession data is missing precomputed field(s): {fields}; "
            "rerun `trident accession` before `trident check`"
        )

    return _read_cluster_skip_reason(acc_adata)

def check(args):
    """
    Validate accession-level results by aligning reads, computing coverage,
    and generating consensus contigs with BLAST verification.

    Requires the probabilities, reestimations, and read clusters persisted by
    `trident accession`; these model outputs are not recomputed here.
    """
    with RunResourceTracker("check", args, logger=logger) as tracker:
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
        acc_adata = load_accession_adata(config=config, rebuild=False)
        skip_reason = _accession_skip_reason(acc_adata)
        if skip_reason is not None:
            logger.warning("Skipping check for %s: %s.", basename, skip_reason)
            tracker.mark(skipped=True, skip_reason=skip_reason)
            return None

        skip_reason = _precomputed_validation_skip_reason(acc_adata)
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
        existing_contigs = os.path.join(args.output, "cluster_contigs.fasta")
        if os.path.isfile(existing_contigs):
            logger.info("Reusing existing contigs from %s.", existing_contigs)
            acc_adata.uns["trident"]["contigs"] = existing_contigs
        else:
            if shutil.which("megahit") is None:
                raise EnvironmentError("megahit not found. Install with: conda install -c bioconda megahit=1.2.9 -y")
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
