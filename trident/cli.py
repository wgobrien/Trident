import argparse, os, sys

def add_common_args(p):
    p.add_argument("--reads", required=True, help="Path to reads file")
    p.add_argument("--library", required=True, help="Path to library file")
    p.add_argument("--output", required=True, help="Output directory")

def add_later_common(p):
    p.add_argument("--pair", default=None, help="Path to paired reads file")
    p.add_argument("--pathogen-dict", default=None, help="Path to custom pathogen_dict.json (auto-downloaded if not provided)")
    p.add_argument("--threads", type=int, default=32, help="Number of threads for parallel operations")
    p.add_argument("--save-figs", action="store_true", help="Saves figures to OUTPUT/figs/")
    p.add_argument("--low-memory", action="store_true", help="Avoid building SQLite db; scans compressed files instead (slower but uses less disk/memory)")
    p.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    p.add_argument("--device", type=str, default=None, help="Compute device, e.g. 'cuda' or 'cpu' (auto-detected by default)")

def add_model_args(p):
    p.add_argument("--num-latent", type=int, default=5, help="Number of latent factors in the model")
    p.add_argument("--concentration", type=float, default=20.0, help="Concentration for beta prior on coverage metric")
    p.add_argument("--batch-size", type=int, default=int(2**12), help="Subsample size per SVI step")
    p.add_argument("--steps", type=int, default=1500, help="Number of iterations to run for SVI fit")
    p.add_argument("--learning-rate", type=float, default=1e-2, help="SVI optimizer learning rate")
    p.add_argument("--num-samples", type=int, default=1000, help="Number of samples for posterior mean")
    p.add_argument("--keep-top", type=float, default=0.0, help="Filter quantile for removing low-count taxids (0.0 = keep all)")
    p.add_argument("--spaces", type=int, default=3, help="Spaced seed masking parameter for kmer matching")
    p.add_argument("--k", type=int, default=35, help="Kmer size for sequence matching")
    p.add_argument("--l", type=int, default=31, help="Minimizer window length")
    p.add_argument("--max-var", type=int, default=300, help="Maximum number of taxa/accessions to retain")
    p.add_argument("--continue-training", action="store_true", help="Pick up where training left off on a previous run")
    p.add_argument("--build-outputs", action="store_true", help="Rebuild outputs from an existing trained model; train if none exists")
    p.add_argument("--rebuild", action="store_true", help="Rebuild data for a set of reads")


def main():
    known_commands = ["accession", "check", "skill", "autocomplete"]
    
    if len(sys.argv) > 1 and sys.argv[1] in known_commands:
        parser = argparse.ArgumentParser(description="Trident pipeline")
        subparsers = parser.add_subparsers(dest="command")

        acc_parser = subparsers.add_parser("accession", help="Run accession-level estimation")
        add_common_args(acc_parser)
        acc_parser.add_argument("--database", required=True, help="Path to database")
        add_later_common(acc_parser)
        add_model_args(acc_parser)

        check_parser = subparsers.add_parser("check", help="Validate accession results via alignment and BLAST")
        add_common_args(check_parser)
        add_later_common(check_parser)

        skill_parser = subparsers.add_parser("skill", help="Install or show the Trident SKILL.md file")
        skill_parser.add_argument("--dest", default=None, help="Destination directory or markdown file path")
        skill_parser.add_argument(
            "--source",
            default=None,
            help="Custom local source path. If omitted, fetches the latest SKILL.md from GitHub, with the UCLA mirror as fallback.",
        )
        skill_subparsers = skill_parser.add_subparsers(dest="skill_command")
        skill_subparsers.add_parser("show", help="Print the installed SKILL.md")
        skill_subparsers.add_parser("path", help="Print the most recent saved skill path, or resolve --dest")

        autocomplete_parser = subparsers.add_parser("autocomplete", help="Write shell completion setup for trident")
        autocomplete_parser.add_argument("--shell", choices=["bash", "zsh"], default=None)
        autocomplete_parser.add_argument("--dest", default=None, help="Optional output file for the completion script")
        
        args = parser.parse_args()

        if args.command == "accession":
            from .estimate_accessions import estimate_accessions
            estimate_accessions(args)
        elif args.command == "check":
            from .check import check
            check(args)
        elif args.command == "skill":
            from .skill import skill_install_command, skill_path_command, skill_show_command
            if args.skill_command == "show":
                skill_show_command(args)
            elif args.skill_command == "path":
                skill_path_command(args)
            else:
                skill_install_command(args)
        elif args.command == "autocomplete":
            from .skill import skill_autocomplete_command
            skill_autocomplete_command(args)
    else:
        reest_parser = argparse.ArgumentParser(description="Run Trident reestimation pipeline")
        add_common_args(reest_parser)
        reest_parser.add_argument("--database", required=True, help="Path to database")
        add_later_common(reest_parser)
        add_model_args(reest_parser)
        
        if len(sys.argv) == 1:
            reest_parser.print_help()
            sys.exit(0)
        
        args = reest_parser.parse_args()
        from .reestimate import run_trident
        run_trident(args)
