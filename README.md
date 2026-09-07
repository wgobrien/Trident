# Trident

Trident is a Bayesian metagenomic classification tool that reestimates taxonomic and accession-level abundance from sequencing reads. It takes Kraken2-classified reads, builds a latent factor model over kmer hit counts, and produces posterior probability estimates for each read-taxon assignment.

## Installation

```bash
pip install .
```

Conda setup 

```bash
conda env create -f environment.yml
conda activate trident
```

Trident includes C++ executables for kmer coverage and count building. Compile them before first use:

```bash
make
```

## Setup

Trident stores downloaded taxonomy files (names.dmp, nodes.dmp), accession-to-taxid mappings, per-sample data, and trained models in a global save directory. Set the `TRIDENT_GLOBALS` environment variable to point to this directory:

```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export TRIDENT_GLOBALS=/path/to/trident/saves

# Then reload
source ~/.bashrc
# or source ~/.zshrc, etc
```

If `TRIDENT_GLOBALS` is not set, Trident defaults to `./_trident/` in the current working directory.

The first run will download NCBI taxonomy files (~50 MB) and, when accession-level estimation is used, the accession-to-taxid mapping (~3 GB compressed). A SQLite database is built from this mapping for fast lookups on subsequent runs (~25 GB on disk). Use `--low-memory` to skip building the SQLite database and scan the compressed file directly instead.

## Shell Completion

Generate a completion script for your shell and source it from your shell's rc file.

```bash
# bash
trident autocomplete
source ~/.bash_completion.d/trident.sh

# zsh
trident autocomplete --shell zsh
source ~/.zsh_completion.d/trident.zsh
```

## LLM Skill

The repository root includes a `SKILL.md` file describing efficient Trident usage for LLM workflows. By default, `trident skill --dest ...` fetches the latest `SKILL.md` from GitHub. Pass `--source /path/to/SKILL.md` to install from a local file instead.

```bash
# Claude
trident skill --dest ~/.claude/skills/trident

# Codex
trident skill --dest ~/.codex/skills/trident
```

## Usage

Trident has three commands, intended to be run in order:

### 1. Taxid-level reestimation (default)

Classifies reads with Kraken2, propagates kmer counts through the taxonomic tree, fits a latent factor model, and reestimates read-level taxid assignments.

```bash
trident \
    --reads reads_R1.fastq.gz \
    --pair reads_R2.fastq.gz \
    --library /path/to/library.fa \
    --database /path/to/kraken2_db \
    --output results/ \
    --save-figs
```

### 2. Accession-level estimation

Refines taxid-level results down to individual reference accessions. Requires that the default taxid reestimation has been run first (it will run automatically if not).

```bash
trident accession \
    --reads reads_R1.fastq.gz \
    --pair reads_R2.fastq.gz \
    --library /path/to/library.fa \
    --database /path/to/kraken2_db \
    --output results/ \
    --save-figs
```

### 3. Validation

Aligns reads to top accessions, computes coverage, assembles consensus contigs per read cluster, and BLASTs them for independent verification. Requires that `trident accession` has been run first.

```bash
trident check \
    --reads reads_R1.fastq.gz \
    --pair reads_R2.fastq.gz \
    --library /path/to/library.fa \
    --output results/ \
    --save-figs
```

## Configuration

### Required arguments

| Argument | Description |
|---|---|
| `--reads` | Path to reads file (FASTQ, optionally gzipped) |
| `--library` | Path to reference library FASTA |
| `--output` | Output directory for results and figures |
| `--database` | Path to Kraken2 database (not required for `check`) |

### Common options

| Argument | Default | Description |
|---|---|---|
| `--pair` | None | Path to paired-end reads file (R2) |
| `--threads` | 32 | Number of threads for parallel operations |
| `--save-figs` | off | Save diagnostic figures to `OUTPUT/figs/` |
| `--low-memory` | off | Skip SQLite db construction; scan compressed files directly (slower, less disk) |
| `--seed` | 1 | Random seed for reproducibility |
| `--device` | auto | Compute device (`cuda` or `cpu`; auto-detected by default) |

### Model and data options

These apply to the default reestimation and `accession` commands:

| Argument | Default | Description |
|---|---|---|
| `--num-latent` | 5 | Number of latent factors in the model |
| `--concentration` | 20.0 | Concentration parameter for the beta prior on coverage |
| `--batch-size` | 2048 | Number of reads subsampled per SVI optimization step |
| `--steps` | 1500 | Number of SVI optimization iterations |
| `--learning-rate` | 0.01 | SVI optimizer learning rate |
| `--num-samples` | 1000 | Number of posterior samples for mean estimation |
| `--keep-top` | 0.0 | Filter quantile for removing low-count taxids (0.0 keeps all) |
| `--spaces` | 3 | Spaced seed masking parameter for kmer matching |
| `--k` | 35 | Kmer size for sequence matching |
| `--l` | 31 | Minimizer window length |
| `--max-var` | 300 | Maximum number of taxa or accessions to retain in the model |
| `--continue-training` | off | Resume training from a previous run's saved model |
| `--build-outputs` | off | Rebuild reports and figures from the selected saved model, skipping training; train normally if no model exists |
| `--rebuild` | off | Rebuild data from scratch (rerun Kraken, recount kmers) |

## Output

Trident writes results to the `--output` directory:

- `<sample>.treport` -- tab-separated taxid-level abundance report
- `<sample>.trident` -- per-read classification with columns: `read_id`, `kraken_label`, `reestimation`, `read_cluster`, `taxid_proba`
- `<sample>.accession.treport` -- accession-level abundance report (after `trident accession`)
- `<sample>.runtime.json` -- nested wallclock time and max RAM summaries under `taxa|accession|check`
- `figs/` -- diagnostic plots when `--save-figs` is used

Intermediate data (coverage files, anndata objects, trained models) are stored under `TRIDENT_GLOBALS/<sample>/`.

## Python API

Trident can also be used as a Python library, e.g. in Jupyter notebooks:

```python
import trident

# Configure a run
config = trident.tl.TridentConfig(
    reads="reads_R1.fastq.gz",
    pair="reads_R2.fastq.gz",
    library="/path/to/library.fa",
    database="/path/to/kraken2_db",
    threads=32,
)

# Load taxonomy and match its targets to the reference library
taxonomic_data = trident.tl.taxutils(config)

# Load or build the read-taxon count matrix
adata = trident.tl.load_adata(config, taxonomic_data)

# Fit the model
trident.tl.train_trident(adata, config)

# Posterior inference and reestimation
trident.tl.get_posteriors(adata, num_samples=1000)
trident.tl._get_read_probabilities(adata)
trident.tl.reestimate(adata)
trident.tl.var_marginal(adata)

# Plots
trident.pl.plot_ppc(adata)
trident.pl.var_cluster(adata, method="ward", standardize="zscore_cols", mode="silhouette")
trident.pl.var_factor_density(adata, taxonomic_data)
trident.pl.read_cluster(adata, n_reads_subsample=1000, top_n_var=30)
trident.pl.var_proba(adata, figsize=(8, 6), top_n=30)
trident.pl.transition(adata)
```

For accession-level estimation:

```python
# Load accession-level data (requires taxid reestimation first)
acc_adata = trident.tl.load_accession_adata(config, taxonomic_data)

# Train, infer, and reestimate at accession level
trident.tl.train_trident(acc_adata, config)
taxonomic_data.load_a2t(list(acc_adata.var.accession.values), low_memory=not config.sqlite)
acc_adata.var["taxid"] = acc_adata.var["accession"].map(taxonomic_data.a2t)

trident.tl.get_posteriors(acc_adata, num_samples=1000)
trident.tl._get_read_probabilities(acc_adata)
trident.tl.reestimate(acc_adata, taxonomic_data)
trident.tl.var_marginal(acc_adata)
```

For validation:

```python
trident.tl.align(acc_adata)
trident.pl.coverage(acc_adata, top_n=5)
trident.tl.cluster_consensus(acc_adata)
contigs = trident.tl.blast_contigs(acc_adata)
```

## Batch processing

For processing multiple samples, see `examples/reads_batch.py`:

```bash
python examples/reads_batch.py \
    --reads-dir /path/to/fastq_dir/ \
    --library /path/to/library.fa \
    --database /path/to/kraken2_db \
    --output batch_results/ \
    --accessions \
    --save-figs
```

This discovers paired-end read files in the directory and runs the full pipeline for each sample.

## License

MIT
