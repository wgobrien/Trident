import argparse
import os

from .tools.model import TridentConfig
from .tools.data import load_adata, taxutils, _match_library_targets
from .tools.train import train_trident
from .global_utils import RunResourceTracker, get_logger, get_sample_folder, get_basename_from_reads
from .tools.evals import get_posteriors, cluster_lca, _get_read_probabilities, reestimate, var_marginal
from .tools.utils import to_treport, write_no_match_taxa_outputs
from .plot.ppc import plot_ppc
from .plot.post import var_cluster, var_factor_density, read_cluster, var_proba, transition, taxid_totals

logger = get_logger(__name__)

def run_trident(args, taxonomic_data=None):
    with RunResourceTracker("taxa", args, logger=logger) as tracker:
        if not os.path.exists(args.library):
            raise FileNotFoundError(f"Library file not found: {args.library}")
        sqlite = not getattr(args, "low_memory", False)
        pathogen_dict = getattr(args, "pathogen_dict", None)
        config = TridentConfig(
            reads=args.reads,
            pair=args.pair,
            library=args.library,
            database=args.database,
            pathogen_dict=pathogen_dict,
            threads=args.threads,
            spaces=args.spaces,
            k=args.k,
            l=args.l,
            num_latent=args.num_latent,
            max_var=args.max_var,
            concentration=args.concentration,
            subsample_size=args.batch_size,
            n_steps=args.steps,
            learning_rate=args.learning_rate,
            taxid_filter_quantile=args.keep_top,
            sqlite=sqlite,
            seed=args.seed,
            **({"device": args.device} if args.device else {}),
        )
        with tracker.step("taxonomy_setup", sqlite=bool(sqlite)) as step:
            created_taxonomic_data = taxonomic_data is None
            if created_taxonomic_data:
                taxonomic_data = taxutils(config)
                matched_before = False
            else:
                matched_before = bool(getattr(taxonomic_data, "_is_matched", False))
                if not matched_before:
                    taxonomic_data.target_taxa = taxonomic_data.sort_taxa(set(taxonomic_data.target_taxa) | {9606})
                    _match_library_targets(taxonomic_data, config)
            step.mark(
                matched_library=not matched_before,
                target_taxa=int(len(getattr(taxonomic_data, "target_taxa", []) or [])),
            )
        with tracker.step("data_build", rebuild=bool(args.rebuild)) as step:
            adata = load_adata(config, taxonomic_data, rebuild=args.rebuild)
            if adata is not None:
                build_stats = adata.uns["trident"].get("build_stats", {})
                step.mark(
                    reads_before_row_filter=build_stats.get("reads_before_row_filter"),
                    reads_after_row_filter=build_stats.get("reads_after_row_filter", int(adata.shape[0])),
                    taxa_after_filter=build_stats.get("taxa_after_filter", int(adata.shape[1])),
                )
        if adata is None:
            basename = get_basename_from_reads(args.reads)
            write_no_match_taxa_outputs(args.output, basename)
            return
        fig_path = os.path.join(args.output, "figs") if args.save_figs else None
        if args.save_figs:
            os.makedirs(fig_path, exist_ok=True)
        with tracker.step(
            "model_fit",
            reads_for_fit=int(adata.shape[0]),
            taxa_for_fit=int(adata.shape[1]),
            n_steps=int(config.n_steps),
        ) as step:
            did_train = train_trident(
                adata,
                config,
                continue_training=args.continue_training,
                save_loss_plot=os.path.join(fig_path, "loss.png") if args.save_figs else None,
                build_outputs=getattr(args, "build_outputs", False),
            )
            step.mark(reused_model=not did_train)
        basename = adata.uns["trident"]["basename"]
        save_path = os.path.join(get_sample_folder(basename), f"{basename}.trident.h5ad.gz")
        with tracker.step(
            "posterior_and_output",
            reads=int(adata.shape[0]),
            taxa=int(adata.shape[1]),
            num_posterior_samples=int(args.num_samples),
        ):
            if args.save_figs:
                plot_ppc(adata, N=1000, save_path=os.path.join(fig_path, "taxid_fit_ppc.png"))
            get_posteriors(adata, num_samples=args.num_samples)
            var_cluster(
                adata,
                method="ward",
                standardize="zscore_cols",
                mode="silhouette",
                save_path=os.path.join(fig_path, "var_cluster.png") if args.save_figs else None,
            )
            cluster_lca(adata, taxonomic_data)
            var_factor_density(
                adata, taxonomic_data,
                save_path=os.path.join(fig_path, "var_factor_density.png") if args.save_figs else None
            )
            probas = _get_read_probabilities(adata)
            reestimate(adata, taxonomic_data)
            var_marginal(adata, taxonomic_data)
            read_cluster(
                adata, n_reads_subsample=1000, top_n_var=30,
                save_path=os.path.join(fig_path, "read_cluster.png") if args.save_figs else None
            )
            var_proba(
                adata, figsize=(8, 6), top_n=30,
                save_path=os.path.join(fig_path, "var_proba.png") if args.save_figs else None
            )
            transition(
                adata, figsize=(16, 12),
                save_path=os.path.join(fig_path, "transition.png") if args.save_figs else None
            )
            basename = adata.uns["trident"]["basename"]
            treport_path = os.path.join(args.output, f"{basename}.treport")
            to_treport(adata, taxonomic_data, save_path=treport_path)
            taxid_totals(
                adata, taxonomic_data,
                save_path=os.path.join(fig_path, "taxid_totals.png") if args.save_figs else None,
            )
            cols_out = ['read_id', 'kraken_label', 'reestimation', 'taxid_proba']
            if 'read_cluster' in adata.obs.columns:
                cols_out.append('read_cluster')
            adata.obs[cols_out].to_csv(
                os.path.join(args.output, f"{basename}.trident"),
                index=False,
                sep="\t"
            )
            adata.write(save_path, compression='gzip')
        return adata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Trident pipeline")

    parser.add_argument("--reads", required=True, help="Path to reads file")
    parser.add_argument("--library", required=True, help="Path to library file")
    parser.add_argument("--database", required=True, help="Path to database")
    parser.add_argument("--output", required=True, help="Output directory")

    parser.add_argument("--pair", default=None, help="Path to paired reads file")

    parser.add_argument("--num-latent", type=int, default=5, help="Number of latent factors in the model")
    parser.add_argument("--concentration", type=float, default=20.0, help="Concentration for beta prior on coverage metric")
    parser.add_argument("--batch-size", type=int, default=int(2**11), help="Subsample size per SVI step")
    parser.add_argument("--steps", type=int, default=1500, help="Number of iterations to run for SVI fit")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="SVI optimizer learning rate")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples for posterior mean")
    parser.add_argument("--keep-top", type=float, default=0.0, help="Filter quantile for removing low-count taxids (0.0 = keep all)")
    parser.add_argument("--spaces", type=int, default=3, help="Spaced seed masking parameter for kmer matching")
    parser.add_argument("--k", type=int, default=35, help="Kmer size for sequence matching")
    parser.add_argument("--l", type=int, default=31, help="Minimizer window length")
    parser.add_argument("--max-var", type=int, default=300, help="Maximum number of taxa/accessions to retain")

    parser.add_argument("--threads", type=int, default=32, help="Number of threads for parallel operations")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Compute device, e.g. 'cuda' or 'cpu' (auto-detected by default)")
    parser.add_argument("--low-memory", action="store_true", help="Avoid building SQLite db; scans compressed files instead (slower but uses less disk/memory)")

    parser.add_argument("--continue-training", action="store_true", help="Pick up where training left off on a previous run")
    parser.add_argument("--build-outputs", action="store_true", help="Rebuild outputs from an existing trained model; train if none exists")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild data for a set of reads")
    parser.add_argument("--save-figs", action="store_true", help="Save figures to [output]/figs/")

    args = parser.parse_args()
    if args.save_figs is True:
        os.makedirs(os.path.join(args.output, "figs"), exist_ok=True)
    run_trident(args)
