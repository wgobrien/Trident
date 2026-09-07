import argparse
import os
import pandas as pd

from .reestimate import run_trident
from .tools.model import TridentConfig
from .tools.data import load_adata, load_accession_adata, taxutils, _match_library_targets
from .tools.train import train_trident
from .tools.evals import get_posteriors, cluster_lca, _get_read_probabilities, reestimate, var_marginal
from .tools.utils import to_treport, write_no_match_accession_outputs, write_no_match_taxa_outputs
from .plot.ppc import plot_ppc
from .plot.post import var_cluster, var_factor_density, read_cluster, var_proba
from .global_utils import RunResourceTracker, get_logger, get_sample_folder, get_basename_from_reads, TRIDENT_GLOBALS

logger = get_logger(__name__)

def estimate_accessions(args, taxonomic_data=None):
    with RunResourceTracker("accession", args, logger=logger) as tracker:
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
        fig_path = os.path.join(args.output, "figs") if args.save_figs else None
        if args.save_figs:
            os.makedirs(fig_path, exist_ok=True)
        with tracker.step("data_build", rebuild=bool(args.rebuild)) as step:
            adata = load_adata(config=config, taxonomic_data=taxonomic_data, rebuild=False)
            rebuilt_taxid_data = False
            reran_taxa_pipeline = False
            source_reads = None
            source_taxa = None
            if adata is None:
                basename = get_basename_from_reads(args.reads)
                write_no_match_taxa_outputs(args.output, basename)
                write_no_match_accession_outputs(args.output, basename)
                return
            # Check that intermediate files exist for the current database/library
            kraken_file = adata.uns["trident"].get("X_file", "")
            coverage_file = adata.uns["trident"].get("L_file", "")
            if not os.path.exists(kraken_file) or not (os.path.exists(coverage_file) or os.path.exists(coverage_file + ".gz")):
                logger.info("Intermediate files not found for current database/library. Rebuilding data...")
                adata = load_adata(config=config, taxonomic_data=taxonomic_data, rebuild=True)
                rebuilt_taxid_data = True
            if adata is None:
                basename = get_basename_from_reads(args.reads)
                write_no_match_taxa_outputs(args.output, basename)
                write_no_match_accession_outputs(args.output, basename)
                return
            if 'run_id' not in adata.uns["trident"]:
                logger.info("Called accession estimation! Fitting taxid reestimation first.")
                adata = run_trident(args, taxonomic_data)
                reran_taxa_pipeline = True
                if adata is None:
                    basename = get_basename_from_reads(args.reads)
                    write_no_match_accession_outputs(args.output, basename)
                    return
            else:
                get_posteriors(adata, num_samples=1000)
                _get_read_probabilities(adata)
                var_cluster(
                    adata,
                    method="ward",
                    standardize="zscore_cols",
                    mode="silhouette",
                    save_path=os.path.join(fig_path, "var_cluster.png") if args.save_figs else None,
                )
                cluster_lca(adata, taxonomic_data)
                var_marginal(adata, taxonomic_data)

            source_reads = int(adata.shape[0])
            source_taxa = int(adata.shape[1])
            config_accs = config.copy()
            acc_adata = load_accession_adata(config=config_accs, taxonomic_data=taxonomic_data, adata=adata, rebuild=args.rebuild)
            if acc_adata is None:
                basename = adata.uns["trident"]["basename"]
                write_no_match_accession_outputs(args.output, basename)
                return
            build_stats = acc_adata.uns["trident"].get("build_stats", {})
            step.mark(
                rebuilt_taxid_data=bool(rebuilt_taxid_data),
                reran_taxa_pipeline=bool(reran_taxa_pipeline),
                source_taxa_reads=source_reads,
                source_taxa_after_filter=source_taxa,
                reads_before_row_filter=build_stats.get("reads_before_row_filter", int(acc_adata.shape[0])),
                reads_after_row_filter=build_stats.get("reads_after_row_filter", int(acc_adata.shape[0])),
                accessions_selected=build_stats.get("accessions_selected"),
                accessions_after_filter=build_stats.get("accessions_after_filter", int(acc_adata.shape[1])),
                taxa_selected_for_accessions=build_stats.get("taxa_selected_for_accessions"),
            )
        del adata
        with tracker.step(
            "model_fit",
            reads_for_fit=int(acc_adata.shape[0]),
            accessions_for_fit=int(acc_adata.shape[1]),
            n_steps=int(config_accs.n_steps),
        ) as step:
            did_train = train_trident(
                acc_adata,
                config_accs,
                save_loss_plot=os.path.join(fig_path, "loss.accession.png") if args.save_figs else None,
                build_outputs=getattr(args, "build_outputs", False),
            )
            step.mark(reused_model=not did_train)
        with tracker.step(
            "posterior_and_output",
            reads=int(acc_adata.shape[0]),
            accessions=int(acc_adata.shape[1]),
        ):
            if args.save_figs:
                plot_ppc(acc_adata, N=1000, save_path=os.path.join(fig_path, "accession_fit_ppc.png"))
            if taxonomic_data.a2t is None:
                taxonomic_data.load_a2t(list(acc_adata.var.accession.values), low_memory=not sqlite)
            acc_adata.var["taxid"] = acc_adata.var["accession"].map(taxonomic_data.a2t).fillna(TRIDENT_GLOBALS["UNMAPPED"]).astype(int)
            if args.save_figs is True:
                acc_adata.var["name"] = acc_adata.var["accession"] + " (" + acc_adata.var["taxid"].map(taxonomic_data.names) + ")"
            get_posteriors(acc_adata, num_samples=10)
            var_cluster(
                acc_adata,
                method="ward",
                standardize="zscore_cols",
                mode="silhouette",
                save_path=os.path.join(fig_path, "var_cluster.accession.png") if args.save_figs else None,
            )
            cluster_lca(acc_adata, taxonomic_data)
            _get_read_probabilities(acc_adata)
            reestimate(acc_adata, taxonomic_data)
            var_marginal(acc_adata, taxonomic_data)
            basename = acc_adata.uns["trident"]["basename"]
            treport_path = os.path.join(args.output, f"{basename}.accession.treport")
            to_treport(acc_adata, taxonomic_data, save_path=treport_path)
            var_factor_density(
                acc_adata, taxonomic_data,
                save_path=os.path.join(fig_path, "var_factor_density.accession.png") if args.save_figs else None,
            )
            read_cluster(
                acc_adata, n_reads_subsample=5000, top_n_var=40,
                save_path=os.path.join(fig_path, "read_cluster.accession.png") if args.save_figs else None,
            )
            var_proba(
                acc_adata, figsize=(10, 12), top_n=50,
                save_path=os.path.join(fig_path, "var_proba.accession.png") if args.save_figs else None,
            )
            prev_treport_path = os.path.join(args.output, f"{basename}.trident")
            if os.path.exists(prev_treport_path):
                prev_treport = pd.read_csv(prev_treport_path, sep="\t")
                to_merge_cols = ['read_id', 'accession_reestimation', 'reestimation', 'accession_proba']
                if 'read_cluster' in acc_adata.obs.columns:
                    to_merge_cols.append('read_cluster')
                to_merge = acc_adata.obs[to_merge_cols]
                rename_map = {"reestimation": "accession(taxid)_reestimation"}
                if 'read_cluster' in to_merge.columns:
                    rename_map["read_cluster"] = "accession_read_cluster"
                to_merge = to_merge.rename(columns=rename_map)
                cols_to_drop = [c for c in to_merge.columns if c in prev_treport.columns and c != "read_id"]
                prev_treport = prev_treport.drop(columns=cols_to_drop)
                prev_treport.merge(to_merge, on="read_id", how="left").to_csv(
                    os.path.join(args.output, f"{basename}.trident"),
                    index=False,
                    sep="\t"
                )
            else:
                acc_adata.obs = acc_adata.obs.rename(columns={"reestimation": "accession(taxid)_reestimation"})
                cols_out = ['read_id', 'kraken_label', 'accession_reestimation', 'accession(taxid)_reestimation', 'accession_proba']
                if 'read_cluster' in acc_adata.obs.columns:
                    cols_out.append('read_cluster')
                acc_adata.obs[cols_out].to_csv(
                    os.path.join(args.output, f"{basename}.accession.trident"),
                    index=False,
                    sep="\t"
                )
        save_path = os.path.join(get_sample_folder(basename), f"{basename}.accessions.trident.h5ad.gz")
        acc_adata.write(save_path, compression='gzip')
        return acc_adata

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
    if args.save_figs:
        os.makedirs(os.path.join(args.output, "figs"), exist_ok=True)
    estimate_accessions(args)
