# data.py

import array
import pandas as pd
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, eye as speye
import os, re, tempfile
import anndata as ad
from collections import defaultdict
from functools import lru_cache
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from collections.abc import Mapping
from taxutils import taxutils as _taxutils

from trident.global_utils import (
    create_trident_uns_key, 
    TRIDENT_GLOBALS, 
    get_sample_folder,
    get_basename_from_reads,
    get_database_name,
    get_logger,
    is_notebook
)
from .run_kraken import KrakenPipeline
from .align_tl import compare_references
from .extract_headers import extract_accessions

logger = get_logger(__name__)

def _coverage_tag(library):
    """Derive a clean tag from a library path for use in coverage filenames."""
    tag = os.path.basename(library)
    for ext in (".gz", ".fasta", ".fa", ".fna"):
        tag = tag.replace(ext, "")
    return tag.replace("-", "")


def _normalize_cache_path(path):
    """Normalize an input path before storing or comparing cache provenance."""
    if path is None or str(path) == "":
        return ""
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def _data_cache_config(config):
    """Return the input paths that determine whether built data can be reused."""
    return {
        key: _normalize_cache_path(getattr(config, key, None))
        for key in ("reads", "pair", "library", "database")
    }


def _cache_mismatch_reason(adata, config, taxid_counts_path, coverage_path):
    """Explain why an existing AnnData cache cannot be reused, if applicable."""
    trident_uns = adata.uns.get("trident", {})
    cached_config = trident_uns.get("data_config")
    if not isinstance(cached_config, Mapping):
        # Model caches created before data_config was introduced still record
        # the source paths in their training config.
        cached_config = trident_uns.get("config")

    if not isinstance(cached_config, Mapping):
        return "cache does not record its source database and library"

    current_config = _data_cache_config(config)
    for key in ("reads", "pair", "library", "database"):
        if key not in cached_config:
            if key in ("library", "database"):
                return f"cache does not record its source {key}"
            continue
        cached_value = _normalize_cache_path(cached_config[key])
        if cached_value != current_config[key]:
            return (
                f"cached {key} is {cached_value or '<none>'}, "
                f"but the current {key} is {current_config[key] or '<none>'}"
            )

    expected_paths = [taxid_counts_path]
    if not (os.path.exists(coverage_path) or os.path.exists(coverage_path + ".gz")):
        expected_paths.append(coverage_path)
    missing_paths = [path for path in expected_paths if not os.path.exists(path)]
    if missing_paths:
        return "required intermediate file(s) are missing: " + ", ".join(missing_paths)

    return None


def _set_data_cache_metadata(adata, config, taxid_counts_path, coverage_path):
    """Attach data provenance and intermediate paths before caching AnnData."""
    create_trident_uns_key(adata)
    adata.uns["trident"]["data_config"] = _data_cache_config(config)
    adata.uns["trident"]["X_file"] = taxid_counts_path
    adata.uns["trident"]["L_file"] = coverage_path
    adata.uns["trident"]["is_acc"] = False


class TridentDataset(Dataset):
    """PyTorch Dataset that lazily densifies rows from a (possibly sparse) AnnData.X."""
    def __init__(self, adata):
        self._X = adata.X
        self._sparse = scipy.sparse.issparse(self._X)

    def __len__(self):
        return self._X.shape[0]

    def __getitem__(self, idx):
        if self._sparse:
            row = np.asarray(self._X[idx].todense()).squeeze(0)
        else:
            row = np.asarray(self._X[idx]).ravel()
        return torch.tensor(row, dtype=torch.float32), idx

def _add_taxids_to_coverage(path, taxonomic_data, sqlite=True):
    header_lines = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
            else:
                break

    df = pd.read_csv(path, sep="\t", comment="#")
    accessions = set(df["accession"].unique())
    taxonomic_data.load_a2t(accessions, low_memory=not sqlite, extend=True)
    df["taxid"] = df["accession"].map(taxonomic_data.a2t)
    df = df.dropna(subset=["taxid"])
    df["taxid"] = df["taxid"].astype(int)

    with open(path, "w") as f:
        for line in header_lines:
            f.write(line)
        df.to_csv(f, sep="\t", index=False)


def kmer_coverage(
    reads,
    library,
    database,
    taxonomic_data,
    pair=None,
    threads=1,
    k=35,
    l=31,
    spaces=0,
    overwrite_kraken=True,
    sqlite=True,
):
    kraken_pipeline = KrakenPipeline(reads=reads, pair=pair, database=database, threads=threads)
    if overwrite_kraken is True:
        kraken_pipeline.run_pipeline()
    else:
        if kraken_pipeline.output_dne() is True:
            kraken_pipeline.run_pipeline()
    kraken_file = kraken_pipeline.kraken_file
    cols = ["status", "read_id", "taxid", "totals", "lca_mapping"]
    basename = os.path.basename(reads).replace(".fastq.gz","")
    target_taxa = set(taxonomic_data.target_taxa)
    if os.path.getsize(kraken_file) == 0:
        logger.warning(f"No reads with hits to target taxids in {kraken_file}. Skip estimation for {basename}.")
        return None
        
    df = pl.scan_csv(
        kraken_file,
        separator="\t",
        has_header=False,
        new_columns=cols
    )
    
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(kraken_file))
    os.close(tmp_fd)
    
    (
        df.filter(
            pl.col("lca_mapping").map_elements(
                lambda x: extract_target_reads(x, target_taxa),
                return_dtype=pl.Boolean
            )
        )
        .sink_csv(
            tmp_path,
            separator="\t",
            include_header=False
        )
    )
    os.replace(tmp_path, kraken_file)

    if os.path.getsize(kraken_file) == 0:
        logger.warning(f"No reads with hits to target taxids found in {kraken_file}. Nothing to estimate.")
        return None

    exe = (
        Path(__file__)
        .resolve()
        .parent
        / "build_data"
        / "coverage"
    )

    if not exe.exists():
        raise FileNotFoundError(f"Executable not found: {exe}")

    basename = get_basename_from_reads(reads)
    coverage_file = str(os.path.join(get_sample_folder(basename), _coverage_tag(library) + ".coverage.tsv"))
    cmd = [
        str(exe),
        str(reads),
        str(library),
        coverage_file,
        "--threads", str(threads),
        "--k", str(k),
        "--l", str(l),
        "--kraken-file", str(kraken_file),
    ]

    if pair is not None:
        cmd += ["--pair", str(pair)]

    if spaces != 0:
        cmd += ["--spaces", str(spaces)]

    if is_notebook:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(cmd, check=True)

    logger.info("Adding taxids to coverage file...")
    _add_taxids_to_coverage(coverage_file, taxonomic_data, sqlite=sqlite)
    subprocess.run(["gzip", "-f", coverage_file], check=True)
    return kraken_file

def _load_counts_binary(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'TCOO':
            raise ValueError(f"Invalid binary format in {path}: {magic!r}")
        header = np.frombuffer(f.read(24), dtype=np.int64)
        n_reads, n_accs, nnz = int(header[0]), int(header[1]), int(header[2])
        if nnz > 0:
            rows = np.frombuffer(f.read(nnz * 4), dtype=np.uint32).copy()
            cols = np.frombuffer(f.read(nnz * 4), dtype=np.uint32).copy()
            data = np.frombuffer(f.read(nnz * 4), dtype=np.uint32).copy()
        else:
            rows = cols = data = np.array([], dtype=np.uint32)

    with open(path + '.row_ids') as f:
        read_ids = [l.rstrip('\n') for l in f]

    with open(path + '.col_ids') as f:
        acc_names = [l.rstrip('\n') for l in f]

    X = coo_matrix((data, (rows, cols)), shape=(n_reads, n_accs)).tocsr()
    return X, read_ids, acc_names


def accession_counts(
    reads,
    library,
    valid_accessions,
    pair=None,
    kraken_file=None,
    threads=1,
    k=35,
    l=31,
    spaces=7,
    max_acc_per_mm=512,
    sqlite=True,
):
    basename = get_basename_from_reads(reads)
    sample_dir = get_sample_folder(basename)
    tmp_fd, accessions_path = tempfile.mkstemp(suffix=".txt", dir=sample_dir)
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            for acc in valid_accessions:
                f.write(f"{acc}\n")

        exe = (
            Path(__file__)
            .resolve()
            .parent
            / "build_data"
            / "build_counts"
        )

        if not exe.exists():
            raise FileNotFoundError(f"Executable not found: {exe}")

        output = os.path.join(
            sample_dir,
            basename + ".accessions.counts.bin"
        )

        cmd = [
            str(exe),
            str(reads),
            str(library),
            str(accessions_path),
            "--output", str(output),
            "--threads", str(threads),
            "--k", str(k),
            "--l", str(l),
            "--spaces", str(spaces),
            "--max-acc-per-mm", str(max_acc_per_mm),
        ]

        if pair is not None:
            cmd += ["--pair", str(pair)]

        if kraken_file is not None:
            if not os.path.exists(kraken_file):
                raise FileNotFoundError(f"Kraken file not found: {kraken_file}")
            cmd += ["--kraken-file", str(kraken_file)]

        logger.info(f"Accession list ({len(valid_accessions)} accessions) written to {accessions_path}")
        if is_notebook:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(cmd, check=True)
    finally:
        os.unlink(accessions_path)

    return _load_counts_binary(output)


def find_elbow(y):
    """Find elbow using max distance from line connecting endpoints."""
    n = len(y)
    x = np.arange(n)
    start, end = np.array([0, y[0]]), np.array([n-1, y[-1]])
    line_vec = end - start
    line_len = np.sqrt(np.sum(line_vec**2))
    line_unit = line_vec / line_len
    
    point_vecs = np.column_stack([x, y]) - start
    proj_lengths = point_vecs @ line_unit
    proj_points = np.outer(proj_lengths, line_unit)
    distances = np.sqrt(np.sum((point_vecs - proj_points)**2, axis=1))
    
    return np.argmax(distances)


def load_accession_adata(config, taxonomic_data=None, adata=None, rebuild:bool=False, cleanup:bool=True):
    basename = os.path.basename(config.reads).replace(".fastq", "").replace(".gz", "")
    acc_pathway = get_sample_folder(basename)
    acc_adata_path = os.path.join(acc_pathway, basename + ".accessions.trident.h5ad.gz")
    if rebuild is False and os.path.exists(acc_adata_path):
        acc_adata = ad.read_h5ad(acc_adata_path)

        basename = get_basename_from_reads(config.reads)
        sample_folder = get_sample_folder(basename)
    
        coverage_filename = _coverage_tag(config.library) + ".coverage.tsv"
        coverage_path = os.path.join(sample_folder, coverage_filename)
        
        is_paired = "" if config.pair is None else ".paired"
        taxid_counts_path = os.path.join(sample_folder, basename + f'.{get_database_name(config.database)}' + is_paired + '.kraken')
        
        acc_adata.uns["trident"]["X_file"] = taxid_counts_path
        acc_adata.uns["trident"]["L_file"] = coverage_path
        build_stats = acc_adata.uns["trident"].get("build_stats", {})
        if not isinstance(build_stats, dict):
            build_stats = {}
        build_stats.setdefault("reads_before_row_filter", int(acc_adata.shape[0]))
        build_stats["reads_after_row_filter"] = int(acc_adata.shape[0])
        build_stats["accessions_after_filter"] = int(acc_adata.shape[1])
        acc_adata.uns["trident"]["build_stats"] = build_stats
        return acc_adata

    adata = adata if adata is not None else load_adata(config, taxonomic_data)

    if "run_id" not in adata.uns["trident"]:
        raise ValueError("Must fit taxid model first.")
    if "marginal_proba" not in adata.var.columns:
        raise ValueError("Must call trident.tl.var_marginal(adata) first.")
        
    sorted_var = adata.var.sort_values("marginal_proba", ascending=False)
    cumsum = sorted_var["marginal_proba"].cumsum().values
    elbow_idx = find_elbow(cumsum)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=np.arange(len(cumsum)), y=cumsum, ax=ax, label="cumsum")
    ax.axvline(elbow_idx, color='r', linestyle='--', label=f"Elbow at idx {elbow_idx}")
    sns.scatterplot(x=[elbow_idx], y=[cumsum[elbow_idx]], color='r', s=100, marker='x', zorder=5, ax=ax)
    ax.set_xlabel("rank")
    ax.set_ylabel("Cumulative taxa marginal probability")
    ax.legend()
    plt.show()

    logger.info(f"Cumulative probability captured: {cumsum[elbow_idx]:.3f}")

    top_taxids = sorted_var.iloc[:elbow_idx+1].taxid.values

    if taxonomic_data is not None:
        top_taxids_set = set(top_taxids)
        
        to_expand = set()
        for tid in top_taxids_set:
            subtree = set(taxonomic_data.get_subtree(tid))
            has_more_specific = bool(subtree.intersection(top_taxids_set) - {tid})
            if not has_more_specific:
                to_expand.add(tid)
        
        expanded_taxids = set(top_taxids_set)
        for tid in to_expand:
            expanded_taxids.update(taxonomic_data.get_subtree(tid))
        
        top_taxids = expanded_taxids
        del expanded_taxids
    
    logger.info(f"Taxa branches:\n{sorted_var.iloc[:elbow_idx+1][['name', 'marginal_proba']].to_string()}")
    top_var = adata.var[adata.var["taxid"].isin(top_taxids)].copy()
    proba = top_var.groupby("taxid")["marginal_proba"].first()
    proba = proba / proba.sum()
    
    basename = adata.uns["trident"]["basename"]
    adata.uns["trident"]["is_acc"] = False

    coverage_path = adata.uns["trident"].get("L_file",
        os.path.join(get_sample_folder(basename), _coverage_tag(config.library) + ".coverage.tsv"))
    kraken_file = adata.uns["trident"].get("X_file", None)

    # Resolve to whichever exists on disk (.tsv or .tsv.gz)
    coverage_gz = coverage_path + ".gz"
    if not os.path.exists(coverage_path) and os.path.exists(coverage_gz):
        coverage_read_path = coverage_gz
    else:
        coverage_read_path = coverage_path

    cvg_df = pd.read_csv(coverage_read_path, sep="\t", comment="#")
    cvg_df = cvg_df[cvg_df["taxid"].isin(set(top_taxids))]
    cvg_df["L"] = cvg_df["shared_minimizers"] / cvg_df["num_kmers"]
    cvg_df["overall_density"] = cvg_df["shared_minimizers"] / cvg_df["seq_len"]

    max_total = 10_000
    min_per_taxid = 50
    max_per_taxid = 300

    filtered = cvg_df.sort_values("overall_density", ascending=False)
    filtered = filtered.groupby("taxid").head(max_per_taxid)

    guaranteed = filtered.groupby("taxid").head(min_per_taxid)
    remaining = filtered.drop(guaranteed.index).head(max_total - len(guaranteed))
    selected = pd.concat([guaranteed, remaining]).drop_duplicates().head(max_total)
    valid_accessions = selected["accession"].values
    del selected, guaranteed, remaining, filtered

    if len(valid_accessions) == 0:
        logger.warning(f"No accessions with coverage found for top taxa. Skip accession estimation for {basename}.")
        return None
    logger.info(f"Scanning {len(valid_accessions)} accessions from {len(top_taxids)} taxids...")

    X_counts, read_ids_list, acc_names_list = accession_counts(
        config.reads,
        config.library,
        valid_accessions=valid_accessions,
        pair=config.pair,
        kraken_file=kraken_file,
        threads=config.threads,
        k=config.k,
        l=config.l,
        spaces=config.spaces,
        max_acc_per_mm=config.max_acc_per_mm,
    )

    # Build lookup dicts from the single cvg_df (already filtered to top_taxids)
    acc_to_seq_len = dict(zip(cvg_df["accession"], cvg_df["seq_len"]))
    acc_to_L = dict(zip(cvg_df["accession"], cvg_df["L"]))
    acc_to_taxid = dict(zip(cvg_df["accession"], cvg_df["taxid"]))
    del cvg_df

    acc_adata = ad.AnnData(X_counts)
    acc_adata.obs["read_id"] = read_ids_list
    acc_adata.var["accession"] = acc_names_list
    acc_adata.var["seq_len"] = acc_adata.var["accession"].map(acc_to_seq_len)
    acc_adata.var["L"] = acc_adata.var["accession"].map(acc_to_L).fillna(0)
    
    create_trident_uns_key(acc_adata)
    acc_adata.uns["trident"]["basename"] = adata.uns["trident"]["basename"]
    acc_adata.uns["trident"]["is_acc"] = True
    acc_adata.uns["trident"]["X_file"] = adata.uns["trident"]["X_file"]
    acc_adata.uns["trident"]["L_file"] = adata.uns["trident"]["L_file"]
    id_to_k = dict(zip(adata.obs["read_id"], adata.obs["kraken_label"]))
    acc_adata.obs["kraken_label"] = acc_adata.obs["read_id"].map(id_to_k)
    del adata

    acc_adata.var["taxid"] = acc_adata.var["accession"].map(acc_to_taxid).fillna(-1).astype(int)
    if taxonomic_data is not None:
        acc_adata.var["name"] = acc_adata.var["taxid"].map(taxonomic_data.names)
    
    acc_adata.var["sum_r(X_r,t)"] = np.asarray(acc_adata.X.sum(0)).flatten().astype(int)
    
    rank_L = acc_adata.var["L"].rank(pct=True)
    rank_sum = acc_adata.var["sum_r(X_r,t)"].rank(pct=True)
    
    acc_adata.var["composite_score"] = (rank_L + rank_sum) / 2

    # Proportional slot allocation
    threshold = acc_adata.var["composite_score"].quantile(.8)
    n_keep = config.max_var #min(config.max_var, (acc_adata.var["composite_score"] >= threshold).sum())

    acc_taxids = set(acc_adata.var["taxid"].unique())
    proba_filtered = proba[proba.index.isin(acc_taxids)]
    proba_filtered = proba_filtered / proba_filtered.sum()

    acc_counts = acc_adata.var["taxid"].value_counts()
    slots = (proba_filtered * config.max_var).apply(np.floor).astype(int).clip(lower=1)
    slots = slots.clip(upper=acc_counts.reindex(slots.index).fillna(0).astype(int))

    remainder = config.max_var - slots.sum()
    if remainder > 0:
        for taxid in proba_filtered.sort_values(ascending=False).index:
            if remainder <= 0:
                break
            available = acc_counts.get(taxid, 0)
            can_add = available - slots[taxid]
            if can_add > 0:
                add = min(can_add, remainder)
                slots[taxid] += add
                remainder -= add

    keep_idx = []
    for taxid, n in slots.items():
        mask = acc_adata.var["taxid"] == taxid
        taxid_accs = acc_adata.var.loc[mask].nlargest(n, "composite_score")
        keep_idx.extend(taxid_accs.index.tolist())
    
    acc_adata._inplace_subset_var(acc_adata.var.index.isin(keep_idx))

    if (acc_adata.X.shape[0] == 0) or (acc_adata.X.shape[1] < 2):
        logger.warning(f"No reads with hits found for top accessions. Skip accession estimation for {basename}.")
        return None

    sim = compare_references(acc_adata, config)
    for _ in range(10):
        row_sums = np.asarray(sim.sum(1)).flatten()
        sim = sim.multiply(1.0 / (row_sums[:, None] + 1e-12))
        col_sums = np.asarray(sim.sum(0)).flatten()
        sim = sim.multiply(1.0 / (col_sums[None, :] + 1e-12))

    acc_adata.X = acc_adata.X @ (speye(acc_adata.shape[1], dtype=np.float64, format="csr") + sim)
    acc_adata.X = acc_adata.X.ceil()
    
    acc_adata = acc_adata[:, np.asarray(acc_adata.X.sum(0)).ravel() > 0]
    acc_adata = acc_adata[np.asarray(acc_adata.X.sum(1)).ravel() > 0, :]
    acc_adata.uns["trident"]["build_stats"] = {
        "reads_before_row_filter": int(len(read_ids_list)),
        "reads_after_row_filter": int(acc_adata.shape[0]),
        "accessions_selected": int(len(valid_accessions)),
        "accessions_after_filter": int(acc_adata.shape[1]),
        "taxa_selected_for_accessions": int(len(top_taxids)),
    }
    
    if cleanup is True:
        base = os.path.join(
            get_sample_folder(acc_adata.uns["trident"]["basename"]),
            acc_adata.uns["trident"]["basename"] + ".accessions.counts.bin"
        )
        for ext in ("", ".row_ids", ".col_ids"):
            try:
                os.remove(base + ext)
            except FileNotFoundError:
                pass
            
    if (acc_adata.X.shape[0] == 0) or (acc_adata.X.shape[1] < 2):
        logger.warning(f"No reads with hits found for top accessions. Skip accession estimation for {basename}.")
        return None
    logger.info("Accession estimation data loading complete.")
    return acc_adata.copy()

def extract_target_reads(lca_mapping, target_taxa):
    for item in lca_mapping.replace(" |:| ", " ").split():
        if ':' not in item:
            continue
        taxid = item.split(':', 1)[0]
        if taxid == "A":
            continue
        tid = int(taxid)
        if tid != 9606 and tid in target_taxa:
            return True
    return False

def extract_taxid_counts(row, target_taxa_set):
    read_taxid_counts = {}
    
    for pair in str(row).split():
        taxid_split = pair.split(":")
        if len(taxid_split) != 2:
            continue
        left, right = taxid_split
        count = int(right)

        if left.startswith("{"):
            taxids_in_reads = left.strip("{}").split(",")
            if not taxids_in_reads:
                continue
            share = count / len(taxids_in_reads)
            for taxid in taxids_in_reads:
                if taxid == "A":
                    continue
                tid = int(taxid)
                read_taxid_counts[tid] = read_taxid_counts.get(tid, 0) + share
        else:
            if left == "A":
                continue
            tid = int(left)
            read_taxid_counts[tid] = read_taxid_counts.get(tid, 0) + count

    overlap = target_taxa_set.intersection(set(read_taxid_counts.keys()) - {9606})
    if not overlap:
        return None

    filtered_counts = {
        tid: c
        for tid, c in read_taxid_counts.items()
        if tid in target_taxa_set or tid == 9606
    }

    return filtered_counts if filtered_counts else None

def extract_accession_ids(fasta_path, batch_size=10_000):
    """Extract accession IDs with version numbers from FASTA headers."""
    fd, accessions_path = tempfile.mkstemp(suffix=".accessions.txt")
    os.close(fd)
    try:
        extract_accessions(fasta_path, accessions_path, batch_size=batch_size)
        accessions = {}
        with open(accessions_path) as handle:
            for line in handle:
                accession = line.strip()
                if accession:
                    accessions.setdefault(accession, None)
        return list(accessions)
    finally:
        try:
            os.remove(accessions_path)
        except FileNotFoundError:
            pass


def _match_target_subtrees(taxonomic_data, library_taxids):
    target_taxa = list(dict.fromkeys(taxonomic_data.target_taxa))
    target_set = set(target_taxa)
    library_taxids = {int(taxid) for taxid in library_taxids}
    matched = target_set.intersection(library_taxids)

    candidates = [taxid for taxid in target_taxa if taxid not in matched]
    if not candidates or not library_taxids:
        return matched

    for taxid in candidates:
        if library_taxids.intersection(taxonomic_data.get_subtree(taxid)):
            matched.add(taxid)

    return matched


def _match_library_targets(taxonomic_data, config):
    """Restrict taxutils target taxa to taxa represented in the reference library."""
    if bool(getattr(taxonomic_data, "_is_matched", False)):
        return taxonomic_data

    logger.info("Matching library to target taxa.")
    acc_ids = extract_accession_ids(config.library)
    taxonomic_data.load_a2t(
        acc_ids,
        low_memory=not getattr(config, "sqlite", True),
    )
    library_taxids = {
        taxid for accession in acc_ids
        if (taxid := taxonomic_data.a2t.get(accession)) is not None
    }

    matched = _match_target_subtrees(taxonomic_data, library_taxids)
    taxonomic_data.target_taxa = taxonomic_data.sort_taxa(matched)
    taxonomic_data._is_matched = True
    logger.info(
        "Matched %d target taxa using %d library accessions mapped to %d taxids.",
        len(matched),
        len(acc_ids),
        len(library_taxids),
    )
    return taxonomic_data


def taxutils(config):
    """Load taxutils and restrict its targets to the configured reference library."""
    taxonomic_data = _taxutils(
        low_memory=not getattr(config, "sqlite", True),
        targets_json=getattr(config, "pathogen_dict", None),
    )
    taxonomic_data.target_taxa = taxonomic_data.sort_taxa(
        set(taxonomic_data.target_taxa) | {9606}
    )
    return _match_library_targets(taxonomic_data, config)

def target_taxid_counts(
    path,
    taxonomic_data,
    chunksize=100_000,
    filter_quantile=0.,
    return_original: bool = True,
    return_stats: bool = False,
):
    parent_map = taxonomic_data.parent
    
    target_taxa = set(taxonomic_data.target_taxa)
    kmer_map_idx = 4

    # Taxa to exclude upfront
    too_high = set(
        taxonomic_data.nodes.loc[
            taxonomic_data.nodes["rank_idx"] < taxonomic_data.get_rank_order().index("F"),
            "taxon",
        ]
    )

    # First pass: collect taxid totals
    taxid_totals = defaultdict(float)
    n_reads = 0
    for chunk in pd.read_csv(path, sep="\t", header=None, chunksize=chunksize):
        kmer_col = chunk.iloc[:, kmer_map_idx].astype(str).str.replace("|:| ", "", regex=False)
        for row in kmer_col:
            counts = extract_taxid_counts(row, target_taxa)
            if counts is not None:
                n_reads += 1
                for tid, c in counts.items():
                    if tid not in too_high:
                        taxid_totals[tid] += c

    if n_reads == 0:
        empty = csr_matrix((0, 0))
        if return_stats:
            return empty, [], pd.Series(dtype=object), pd.Series(dtype=object), {
                "reads_with_target_hits": 0,
                "rows_in_sparse_matrix": 0,
                "taxids_after_filter": 0,
            }
        return empty, [], pd.Series(dtype=object), pd.Series(dtype=object)

    # Filter low-count taxids
    if filter_quantile > 0:
        threshold = np.quantile(list(taxid_totals.values()), filter_quantile)
        keep_taxids = {t for t, total in taxid_totals.items() if total >= threshold}
    else:
        keep_taxids = set(taxid_totals.keys())
    # del taxid_totals
    ancestor_taxids = set()
    for tid in set(keep_taxids):
        cur = parent_map.get(tid)
        while cur is not None and cur not in too_high:
            if cur not in keep_taxids and cur not in ancestor_taxids:
                ancestor_taxids.add(cur)
            cur = parent_map.get(cur)
    keep_taxids.update(ancestor_taxids)
    taxid_list = sorted(keep_taxids)
    taxid_to_col = {t: i for i, t in enumerate(taxid_list)}
    n_cols = len(taxid_list)

    # build sparse matrix directly
    rows = array.array('i')
    cols = array.array('i')
    data = array.array('f')
    read_ids = []
    labels = []
    row_idx = 0

    for chunk in pd.read_csv(path, sep="\t", header=None, chunksize=chunksize):
        kmer_col = chunk.iloc[:, kmer_map_idx].astype(str).str.replace("|:| ", "", regex=False)
        rid_col = chunk.iloc[:, 1]
        label_col = chunk.iloc[:, 2]

        for row, rid, label in zip(kmer_col, rid_col, label_col):
            counts = extract_taxid_counts(row, target_taxa)
            if counts is None:
                continue
            has_entry = False
            for tid, c in counts.items():
                if tid in keep_taxids:
                    rows.append(row_idx)
                    cols.append(taxid_to_col[tid])
                    data.append(c)
                    has_entry = True
            if has_entry:
                read_ids.append(rid)
                labels.append(label)
                row_idx += 1

    n_rows = row_idx
    sparse_matrix = csr_matrix(
        (np.frombuffer(data, dtype=np.float32),
         (np.frombuffer(rows, dtype=np.int32),
          np.frombuffer(cols, dtype=np.int32))),
        shape=(n_rows, n_cols),
    )
    del rows, cols, data

    # Build propagation matrix
    cols_set = set(taxid_list)
    parent_children = defaultdict(list)
    for child_taxid in taxid_list:
        depth = 1
        cur = parent_map.get(child_taxid)
        while cur is not None:
            if cur in cols_set:
                parent_children[cur].append((child_taxid, depth))
            cur = parent_map.get(cur)
            depth += 1
    
    MAX_PER_PARENT = 50
    p_rows, p_cols, p_data = [], [], []
    for par, children in parent_children.items():
        children = sorted(children, key=lambda x: taxid_totals.get(x[0], 0.0), reverse=True)[:MAX_PER_PARENT]
        par_idx = taxid_to_col[par]
        for child, depth in children:
            p_rows.append(taxid_to_col[child])
            p_cols.append(par_idx)
            p_data.append(1.0)
    
    P = csr_matrix(
        (np.array(p_data, dtype=np.float32), (p_rows, p_cols)),
        shape=(n_cols, n_cols),
    )
    del p_rows, p_cols, p_data

    propagated = sparse_matrix @ (speye(n_cols, dtype=np.float32, format="csr") + P.T)
    del sparse_matrix, P
    propagated = csr_matrix(propagated)
    propagated.data = np.ceil(propagated.data).astype(np.int32)

    if return_stats:
        return propagated, taxid_list, pd.Series(read_ids, name="read_id"), pd.Series(labels, name="label"), {
            "reads_with_target_hits": int(n_reads),
            "rows_in_sparse_matrix": int(n_rows),
            "taxids_after_filter": int(len(taxid_list)),
        }
    return propagated, taxid_list, pd.Series(read_ids, name="read_id"), pd.Series(labels, name="label")


def load_adata(
    config, taxonomic_data=None, rebuild=False,
):
    if not os.path.exists(config.library):
        raise FileNotFoundError(f"Library file not found: {config.library}")
    basename = get_basename_from_reads(config.reads)
    sample_folder = get_sample_folder(basename)
    name = basename + ".trident"
    save_path = os.path.join(sample_folder, f"{name}.h5ad.gz")

    coverage_filename = _coverage_tag(config.library) + ".coverage.tsv"
    coverage_path = os.path.join(sample_folder, coverage_filename)
    
    is_paired = "" if config.pair is None else ".paired"
    taxid_counts_path = os.path.join(sample_folder, basename + f'.{get_database_name(config.database)}' + is_paired + '.kraken')
    build_data = rebuild or not os.path.exists(save_path)
    refresh_intermediates = overwrite_cached_kraken = rebuild
    if not build_data:
        counts_ad = ad.read_h5ad(save_path)
        mismatch_reason = _cache_mismatch_reason(
            counts_ad, config, taxid_counts_path, coverage_path
        )
        if mismatch_reason is not None:
            logger.info(
                "Cached data is incompatible with the current run (%s). "
                "Rebuilding data...",
                mismatch_reason,
            )
            build_data = True
            refresh_intermediates = True
            overwrite_cached_kraken = not mismatch_reason.startswith(
                "required intermediate file"
            )

    if build_data:
        if taxonomic_data is None:
            raise ValueError("Must pass taxonomic_data if anndata not yet constructed.")
        
        if (
            refresh_intermediates
            or not os.path.exists(coverage_path)
            or not os.path.exists(taxid_counts_path)
        ):
            logger.info("Counting kmer coverage...")
            result = kmer_coverage(
                config.reads, config.library, config.database,
                taxonomic_data=taxonomic_data, pair=config.pair,
                threads=config.threads, k=config.k, l=config.l,
                spaces=config.spaces, overwrite_kraken=overwrite_cached_kraken,
                sqlite=config.sqlite,
            )
            if result is None:
                return None

        if os.path.getsize(taxid_counts_path) == 0:
            logger.warning(f"No reads with hits to target taxids in {taxid_counts_path}. Skip estimation for {basename}.")
            return None
        
        os.makedirs(sample_folder, exist_ok=True)
        logger.info("Propagating counts through taxonomic tree...")
        
        propagated, taxid_list, read_ids, labels, target_stats = target_taxid_counts(
            taxid_counts_path, taxonomic_data,
            filter_quantile=config.taxid_filter_quantile,
            return_stats=True,
        )
        coverage_gz = coverage_path + ".gz"
        if not os.path.exists(coverage_path) and os.path.exists(coverage_gz):
            coverage_read_path = coverage_gz
        else:
            coverage_read_path = coverage_path

        logger.info("Computing coverage L scores...")
        sep = "\t" if coverage_path.endswith((".txt", ".tsv")) else ","
        df = pd.read_csv(coverage_read_path, comment="#", sep=sep)

        df["kmer_density"] = df["num_kmers"] / df["seq_len"]
        df["L"] = df["shared_minimizers"] / df["num_kmers"]
        threshold = df["kmer_density"].quantile(0.25)
        df = df[df["kmer_density"] > threshold]
        L_map = df.groupby("taxid", sort=False)["L"].max().to_dict()
        parent = taxonomic_data.parent
        for taxid in list(L_map):
            child_val = L_map[taxid]
            node = taxid
            while node in parent and parent[node] != node:
                node = parent[node]
                if L_map.get(node, 0) < child_val:
                    L_map[node] = child_val
                else:
                    break
        del df
        # 'L': (g['L'] * g['kmer_density']).sum() / g['kmer_density'].sum(),
        # coverage_taxids = set(L_map.keys())
        # taxid_overlap = [t for t in taxid_list if t in coverage_taxids]
        # taxid_overlap.extend([t for t in taxid_list if t not in coverage_taxids])
        # taxid_to_col = {t: i for i, t in enumerate(taxid_list)}
        # col_indices = [taxid_to_col[t] for t in taxid_overlap]

        L_arr = np.array([L_map.get(t, 0.0) for t in taxid_list], dtype=np.float64)
        del L_map
        
        col_sums = np.asarray(propagated.sum(axis=0)).ravel()
        nonzero_mask = col_sums > 0
        
        # Apply nonzero filter first
        propagated = propagated[:, nonzero_mask]
        taxid_list = [taxid_list[i] for i in np.where(nonzero_mask)[0]]
        L_arr = L_arr[nonzero_mask]
        col_sums = col_sums[nonzero_mask]
        
        # Recompute composite on nonzero columns, then apply max_var
        rank_L = pd.Series(L_arr).rank(pct=True).values
        rank_sum = pd.Series(col_sums).rank(pct=True).values
        composite = (rank_L + rank_sum) / 2
        
        keep_indices = np.argsort(composite)[-config.max_var:]
        propagated = propagated[:, keep_indices]
        taxid_list = [taxid_list[i] for i in keep_indices]
        L_arr = L_arr[keep_indices]
        col_sums = col_sums[keep_indices]
        composite = composite[keep_indices]
        
        row_sums = np.asarray(propagated.sum(axis=1)).ravel()
        row_mask = row_sums > 0
        propagated_filtered = propagated[row_mask]
        del propagated
        read_ids_filtered = read_ids[row_mask].reset_index(drop=True)
        labels_filtered = labels[row_mask].reset_index(drop=True)
        
        logger.info("Creating anndata...")
        counts_ad = ad.AnnData(X=csr_matrix(propagated_filtered))
        del propagated_filtered

        counts_ad.obs["read_id"] = read_ids_filtered.values
        counts_ad.obs["kraken_label"] = labels_filtered.values
        counts_ad.var["taxid"] = taxid_list
        counts_ad.var["L"] = L_arr
        human_mask = counts_ad.var["taxid"].astype(str) == "9606"
        if human_mask.any():
            counts_ad.var.loc[human_mask, "L"] = 1
            # logger.info(f"Imputed human (9606) L score with median: {median_L:.4f}")
        
        counts_ad.var["sum_r(X_r,t)"] = col_sums.astype(int)
        counts_ad.var["composite_score"] = composite

        counts_ad.var.index = counts_ad.var.index.astype(str)

        create_trident_uns_key(counts_ad)
        counts_ad.uns["trident"]["basename"] = os.path.basename(config.reads).replace(".fastq.gz", "")
        _set_data_cache_metadata(counts_ad, config, taxid_counts_path, coverage_path)
        counts_ad.uns["trident"]["build_stats"] = {
            "reads_before_row_filter": int(target_stats["reads_with_target_hits"]),
            "reads_after_row_filter": int(counts_ad.shape[0]),
            "taxa_after_filter": int(counts_ad.shape[1]),
        }
        counts_ad.var["name"] = counts_ad.var["taxid"].map(taxonomic_data.names) + (
            " (" + counts_ad.var["taxid"].astype(str) + ")"
        )
        counts_ad.write(save_path, compression='gzip')

    _set_data_cache_metadata(counts_ad, config, taxid_counts_path, coverage_path)
    build_stats = counts_ad.uns["trident"].get("build_stats", {})
    if not isinstance(build_stats, dict):
        build_stats = {}
    build_stats.setdefault("reads_before_row_filter", int(counts_ad.shape[0]))
    build_stats["reads_after_row_filter"] = int(counts_ad.shape[0])
    build_stats["taxa_after_filter"] = int(counts_ad.shape[1])
    counts_ad.uns["trident"]["build_stats"] = build_stats

    if (counts_ad.X.shape[0] == 0) or (counts_ad.X.shape[1] < 2):
        logger.warning(f"No read hits found for target taxa in library. Skip estimation for {basename}.")
        return None
    logger.info("Taxonomic reestimation data loading complete.")
    return counts_ad

ReadPair = namedtuple("ReadPair", ["id", "R1", "R2"])

def get_read_pairs(dirpath):
    files = [os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith(".fastq.gz")]
    pairs = {}
    for f in files:
        m = re.search(r"([A-Za-z]\d{1,2})", os.path.basename(f))
        if not m:
            continue
        sample_id = m.group(1)
        if sample_id not in pairs:
            pairs[sample_id] = {"R1": None, "R2": None}
        if "R1" in os.path.basename(f):
            pairs[sample_id]["R1"] = f
        elif "R2" in os.path.basename(f):
            pairs[sample_id]["R2"] = f
    
    return [ReadPair(id=sid, R1=p["R1"], R2=p["R2"]) for sid, p in sorted(pairs.items())]

def _find_file_by_well(dir_path, well_id, exts):
    f = None
    files = [f for f in os.listdir(dir_path) if f.endswith(exts)]
    for f in files:
        if well_id in f:
            return os.path.join(dir_path, f)
    m = re.search(r"([A-Za-z]\d{2})", well_id)
    code = m.group(1) if m else None
    if code:
        for f in files:
            if code in f:
                return os.path.join(dir_path, f)
    return f

def to_torch(adata):
    X_raw = adata.X
    if scipy.sparse.issparse(X_raw):
        X_raw = X_raw.toarray()
    X = torch.tensor(np.asarray(X_raw), dtype=torch.float32)
    L = torch.tensor(adata.var.loc[:, "L"].values, dtype=torch.float32)
    return X, L
