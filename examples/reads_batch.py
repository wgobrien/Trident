#!/usr/bin/env python3
"""
Trident metagenomic pipeline runner.
Processes paired-end read files against a viral/metagenomic database.
"""
import argparse
import os
import trident
from trident.check import check as check_accession_results
from trident.global_utils import TRIDENT_GLOBALS, get_basename_from_reads

def make_args(r1, r2, sample_output, args):
    """Build a namespace matching what the per-sample commands expect."""
    return argparse.Namespace(
        reads=r1,
        pair=r2,
        library=args.library,
        database=args.database,
        output=sample_output,
        threads=args.threads,
        spaces=args.spaces,
        num_latent=args.num_latent,
        concentration=args.concentration,
        batch_size=args.subsample_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        rebuild=args.rebuild,
        continue_training=args.continue_training,
        build_outputs=args.build_outputs,
        save_figs=args.save_figs,
        k=args.k,
        l=args.l,
        max_var=args.max_var,
        keep_top=args.keep_top,
        seed=args.seed,
        device=args.device,
        low_memory=args.low_memory,
        skip_existing=args.skip_existing,
    )

def main(args):
    os.makedirs(args.output, exist_ok=True)
    read_pairs = trident.tl.get_read_pairs(args.reads_dir)
    if not read_pairs:
        return

    # Validation uses the accession models and cached data produced by an
    # earlier run, so do not initialize taxonomic data or enter either build
    # or fitting path in check mode.
    if getattr(args, "check", False):
        global_save_folder = os.path.expanduser(TRIDENT_GLOBALS["save_folder"])
        for i, rp in enumerate(read_pairs):
            basename = get_basename_from_reads(rp.R1)
            sample_name = os.path.basename(rp.R1).split("_R1")[0]
            acc_adata_path = os.path.join(
                global_save_folder,
                basename,
                f"{basename}.accessions.trident.h5ad.gz",
            )
            if not os.path.exists(acc_adata_path):
                print(
                    f"[Skip Check {i+1}/{len(read_pairs)}] {sample_name} "
                    "(accession data does not exist)"
                )
                continue

            sample_output = os.path.join(args.output, sample_name)
            os.makedirs(sample_output, exist_ok=True)

            sample_args = make_args(rp.R1, rp.R2, sample_output, args)
            print(f"[Check {i+1}/{len(read_pairs)}] {sample_name}")
            check_accession_results(sample_args)
        return

    global_save_folder = os.path.expanduser(TRIDENT_GLOBALS["save_folder"])
    taxonomy_config = trident.tl.TridentConfig(
        reads=read_pairs[0].R1,
        pair=read_pairs[0].R2,
        library=args.library,
        database=args.database,
        sqlite=not args.low_memory,
    )
    taxonomic_data = trident.tl.taxutils(taxonomy_config)

    # build data as a batch, kraken2 db remains memory mapped
    if args.build_data:
        for i, rp in enumerate(read_pairs):
            basename = get_basename_from_reads(rp.R1)
            sample_name = os.path.basename(rp.R1).split("_R1")[0]
            if args.skip_existing:
                adata_path = os.path.join(
                    global_save_folder,
                    basename,
                    f"{basename}.trident.h5ad.gz",
                )
                if os.path.exists(adata_path):
                    print(f"[Skip Build {i+1}/{len(read_pairs)}] {sample_name} (data exists)")
                    continue
            config = trident.tl.TridentConfig(
                reads=rp.R1,
                pair=rp.R2,
                library=args.library,
                database=args.database,
                threads=args.threads,
                spaces=args.spaces,
                num_latent=args.num_latent,
                concentration=args.concentration,
                subsample_size=args.subsample_size,
                n_steps=args.steps,
                learning_rate=args.learning_rate,
            )
            trident.tl.load_adata(config, taxonomic_data)
            print(f"[Build {i+1}/{len(read_pairs)}] {sample_name}")

    # then fit models
    for i, rp in enumerate(read_pairs):
        basename = get_basename_from_reads(rp.R1)
        sample_name = os.path.basename(rp.R1).split("_R1")[0]
        sample_output = os.path.join(args.output, sample_name)
        os.makedirs(sample_output, exist_ok=True)
        if args.save_figs:
            os.makedirs(os.path.join(sample_output, "figs"), exist_ok=True)

        sample_args = make_args(rp.R1, rp.R2, sample_output, args)

        if args.accessions:
            accession_treport_path = os.path.join(sample_output, f"{basename}.accession.treport")
            if args.skip_existing and os.path.exists(accession_treport_path):
                print(f"[Skip Accessions {i+1}/{len(read_pairs)}] {sample_name} (accession treport exists)")
                continue
            print(f"[Accessions {i+1}/{len(read_pairs)}] {sample_name}")
            trident.estimate_accessions(sample_args, taxonomic_data=taxonomic_data)
        else:
            treport_path = os.path.join(sample_output, f"{basename}.treport")
            if args.skip_existing and os.path.exists(treport_path):
                print(f"[Skip Fit {i+1}/{len(read_pairs)}] {sample_name} (treport exists)")
                continue
            print(f"[Fit {i+1}/{len(read_pairs)}] {sample_name}")
            trident.run_trident(sample_args, taxonomic_data=taxonomic_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Trident metagenomic classification pipeline."
    )
    parser.add_argument("--reads-dir", required=True, help="Directory containing paired-end read files")
    parser.add_argument("--library", required=True, help="Path to the reference library FASTA file")
    parser.add_argument(
        "--database",
        help="Path to the Trident database directory (required unless --check is used)",
    )
    parser.add_argument("--output", required=True, help="Output directory for per-sample results")
    parser.add_argument("--build-data", action="store_true", help="Build data in a batch before fitting.")
    parser.add_argument("--accessions", action="store_true", help="Run accession-level estimation after taxid fitting.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing accession model fits; skips data building and fitting.",
    )

    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--spaces", type=int, default=3)
    parser.add_argument("--num-latent", type=int, default=5)
    parser.add_argument("--concentration", type=float, default=20.0)
    parser.add_argument("--subsample-size", type=int, default=int(2**11))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--num-samples", type=int, default=1000)

    parser.add_argument("--k", type=int, default=35)
    parser.add_argument("--l", type=int, default=31)
    parser.add_argument("--max-var", type=int, default=300)
    parser.add_argument("--keep-top", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--low-memory", action="store_true")
    
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument(
        "--build-outputs",
        action="store_true",
        help="Rebuild outputs from existing trained models; train samples with no model.",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--save-figs", action="store_true", help="Save figures to [output]/figs/")
    parser.add_argument(
        "--skip-existing", 
        action="store_true", help="Skip samples whose output already exists."
    )
    args = parser.parse_args()
    if not args.check and not args.database:
        parser.error("--database is required unless --check is used")
    main(args)
