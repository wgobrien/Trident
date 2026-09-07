# utils.py

import torch
import pyro
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os, gc, json
import shutil
from collections import defaultdict
import dataclasses
from dataclasses import dataclass, field

from trident.global_utils import TRIDENT_GLOBALS, get_logger, get_sample_folder
from .model import TridentConfig, Trident
from .data import load_adata

logger = get_logger(__name__)

def checkpoint(adata):
    basename = adata.uns["trident"]["basename"]
    save_name = f"{basename}.trident.h5ad.gz"
    if bool(adata.uns["trident"]["is_acc"]) is True:
        save_name = f"{basename}.accessions.trident.h5ad.gz"
    save_path = os.path.join(
        get_sample_folder(basename), save_name
    )
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].fillna("").astype(str)
    for col in adata.var.columns:
        if adata.var[col].dtype == object:
            adata.var[col] = adata.var[col].fillna("").astype(str)
    adata.write(save_path, compression='gzip')

def _is_accession_adata(adata):
    return bool(adata.uns.get("trident", {}).get("is_acc", False))

def _artifact_basename(adata):
    basename = adata.uns["trident"]["basename"]
    return f"{basename}.accessions" if _is_accession_adata(adata) else basename

def clear_trident():
    save_folder = os.path.expanduser(TRIDENT_GLOBALS["save_folder"])

    if not os.path.isdir(save_folder):
        logger.info(f"Save folder does not exist at {save_folder}; nothing to clear.")
        return []

    removed_dirs = []
    for entry in os.scandir(save_folder):
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
            removed_dirs.append(entry.path)

    logger.info(f"Removed {len(removed_dirs)} directories under {save_folder}.")
    return removed_dirs

def save_model(adata, model, overwrite=False, simulation=False):
    run_id_key = "run_id_sim" if simulation else "run_id"
    run_id_key_hist = "run_id_sim_history" if simulation else "run_id_history"
    basename = adata.uns["trident"]["basename"]
    artifact_basename = _artifact_basename(adata)
    
    if run_id_key_hist not in adata.uns["trident"]:
        adata.uns["trident"][run_id_key_hist] = []
    else:
        run_hist = adata.uns["trident"][run_id_key_hist]
        if isinstance(run_hist, np.ndarray):
            adata.uns["trident"][run_id_key_hist] = list(run_hist)
    
    sim = "" if not simulation else "simulation "
    
    if overwrite and adata.uns["trident"][run_id_key_hist]:
        adata.uns["trident"][run_id_key] = adata.uns["trident"][run_id_key_hist][-1]
        logging.info(f"Overwriting existing trident {sim}model with run_id {adata.uns['trident'][run_id_key]}.")
    else:
        adata.uns["trident"][run_id_key] = artifact_basename + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        adata.uns["trident"][run_id_key_hist].append(adata.uns["trident"][run_id_key])
        logging.info(f"Saving trident {sim}model with run_id {adata.uns['trident'][run_id_key]}.")

    model_run_id = adata.uns["trident"][run_id_key]
    sample_folder = get_sample_folder(basename)
    full_path = os.path.join(sample_folder, model_run_id)
    os.makedirs(full_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(full_path, "trident_model.pt"))
    torch.save(model.losses, os.path.join(full_path, "losses.pt"))
    
    param_store = pyro.get_param_store()
    param_dict = {name: param_store[name].detach().cpu() for name in param_store.keys()}
    torch.save(param_dict, os.path.join(full_path, "trident_model_param_store.pt"))
    if model.optimizer is not None:
        torch.save(model.optimizer, os.path.join(full_path, "optimizer_state.pt"))
    cfg_dict = model.config.to_dict()
    config_path = os.path.join(full_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
        
    adata.uns["trident"]["config"] = cfg_dict
    adata.uns["trident"]["config"] = model.config.to_dict()
    save_path = os.path.join(get_sample_folder(basename), f"{artifact_basename}.trident.h5ad.gz")
    adata.write(save_path, compression='gzip')

def _to_model_tensor(model, value):
    return value.detach().to(device=model.config.device, dtype=model.config.dtype)

def get_posterior_gamma(adata, model=None):
    if model is None:
        model = load_model(adata)

    gamma = getattr(model, "posterior_gamma", None)
    if isinstance(gamma, torch.Tensor) and gamma.numel() > 0:
        return _to_model_tensor(model, gamma)

    raise ValueError(
        "Posterior gamma not found on the model. Run `trident.tl.get_posteriors(...)` first."
    )

def get_posterior_pi(adata, model=None):
    if model is None:
        model = load_model(adata)

    pi = getattr(model, "posterior_pi", None)
    if isinstance(pi, torch.Tensor) and pi.numel() > 0:
        return _to_model_tensor(model, pi)

    if "pi" in adata.var:
        pi = torch.tensor(adata.var["pi"].values)
        model.cache_posterior(pi=pi)
        return _to_model_tensor(model, model.posterior_pi)

    raise ValueError(
        "Posterior pi not found on the model or in `adata.var`. Run `trident.tl.get_posteriors(...)` first."
    )

def load_model(adata, name=None, model_config=None, simulation=False):
    
    run_id_key = "run_id_sim" if simulation else "run_id"
    run_id_key_hist = "run_id_sim_history" if simulation else "run_id_history"

    if "trident" not in adata.uns:
        sim = "" if not simulation else "simulation "
        raise ValueError(f"No trident {sim}metadata found in adata.uns.")

    trident_uns = adata.uns["trident"]

    # Resolve which run_id to load
    if name is None:
        if run_id_key not in trident_uns:
            sim = "" if not simulation else "simulation "
            raise ValueError(
                f"No trident {sim}model has been saved for this AnnData object and no name was provided."
            )
        model_run_id = trident_uns[run_id_key]
    else:
        # use history to validate the requested run name
        run_hist = trident_uns.get(run_id_key_hist, [])
        if isinstance(run_hist, np.ndarray):
            run_hist = list(run_hist)
        if name not in run_hist:
            raise ValueError(
                f"Requested run_id '{name}' not found in {run_id_key_hist}: {run_hist}"
            )
        model_run_id = name

    basename = adata.uns["trident"]["basename"]
    sample_folder = get_sample_folder(basename)
    full_path = os.path.join(sample_folder, model_run_id)

    if model_config is None:
        config_path = os.path.join(full_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
            model_config = TridentConfig.from_dict(cfg_dict)
        else:
            if "config" not in trident_uns:
                raise ValueError(
                    f"No config.json in '{full_path}' and no 'config' in adata.uns['trident']."
                )
            cfg_dict = trident_uns["config"]
            model_config = TridentConfig.from_dict(cfg_dict)

    model=None
    model = Trident(model_config)

    state_dict = torch.load(
        os.path.join(full_path, "trident_model.pt"),
        map_location=model_config.device,
    )
    posterior_gamma = state_dict.pop("posterior_gamma", None)
    posterior_pi = state_dict.pop("posterior_pi", None)
    posterior_chi = state_dict.pop("posterior_chi", None)
    model.load_state_dict(state_dict, strict=False)
    model.cache_posterior(
        gamma=posterior_gamma,
        pi=posterior_pi,
        chi=posterior_chi,
    )

    pyro.clear_param_store()
    gc.collect()
    if isinstance(model_config.device, str) and model_config.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    param_store = pyro.get_param_store()
    param_dict = torch.load(
        os.path.join(full_path, "trident_model_param_store.pt"),
        map_location=model_config.device,
    )
    for param_name, param in param_dict.items():
        if param_name in param_store:
            param_store[param_name] = param.to(model.config.device)
        else:
            pyro.param(param_name, param.to(model.config.device))
    opt_path = os.path.join(full_path, "optimizer_state.pt")
    if os.path.exists(opt_path):
        model.optimizer = torch.load(opt_path, map_location=model_config.device)
    
    losses_path = os.path.join(full_path, "losses.pt")
    if os.path.exists(losses_path):
        model.losses = torch.load(losses_path, map_location="cpu")
        
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(model.config.threads)
    
    logging.info(f"Loaded model {model_run_id}.")
    return model

class TReportEntry:
    def __init__(self, taxid, taxid_proba, unique_count, rank_code, name, depth):
        self.taxid = taxid
        self.taxid_proba = taxid_proba
        self.unique_count = unique_count
        self.cumulative_count = unique_count
        self.pct = 0
        self.rank_code = rank_code
        self.name = name
        self.depth = depth

    def update_cumulative(self, value):
        self.cumulative_count += value

    def set_pct(self, total):
        self.pct = (self.cumulative_count / total) * 100 if total else 0


def treport_to_df(treport):
    return pd.DataFrame.from_records([
        {
            "taxid": e.taxid,
            "taxid_proba": e.taxid_proba,
            "name": e.name,
            "rank_code": e.rank_code,
            "unique_count": e.unique_count,
            "cumulative_count": e.cumulative_count,
            "pct": e.pct,
            "depth": e.depth,
        }
        for e in treport.values()
    ])

def proba_in_reads(taxid, adata, taxonomic_data):
    try:
        tree = taxonomic_data.get_subtree(taxid)
    except (KeyError, TypeError, ValueError):
        tree = [taxid]
    return float(adata.var[adata.var["taxid"].isin(set(tree))]["marginal_proba"].sum())

def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def write_no_match_treport(save_path, name="No match"):
    _ensure_parent_dir(save_path)
    pd.DataFrame(
        [[0.0, 0, 0, "U", 0.0, 0, name]],
        columns=["pct", "cumulative_count", "unique_count", "rank_code", "taxid_proba", "taxid", "name"],
    ).to_csv(save_path, sep="\t", header=None, index=False)

def write_empty_trident(save_path, columns):
    _ensure_parent_dir(save_path)
    pd.DataFrame(columns=columns).to_csv(save_path, sep="\t", index=False)

def write_no_match_taxa_outputs(output_dir, basename):
    treport_path = os.path.join(output_dir, f"{basename}.treport")
    trident_path = os.path.join(output_dir, f"{basename}.trident")
    write_no_match_treport(treport_path)
    write_empty_trident(trident_path, ["read_id", "kraken_label", "reestimation", "taxid_proba"])
    logger.warning("No taxid matches found for %s. Wrote no-match outputs to %s.", basename, output_dir)
    return {"treport": treport_path, "trident": trident_path}

def write_no_match_accession_outputs(output_dir, basename):
    treport_path = os.path.join(output_dir, f"{basename}.accession.treport")
    trident_path = os.path.join(output_dir, f"{basename}.trident")
    write_no_match_treport(treport_path)

    if os.path.exists(trident_path):
        try:
            prev = pd.read_csv(trident_path, sep="\t")
        except pd.errors.EmptyDataError:
            prev = pd.DataFrame(columns=["read_id", "kraken_label", "reestimation", "taxid_proba"])
        prev["accession_reestimation"] = "no_match"
        prev["accession(taxid)_reestimation"] = 0
        prev["accession_proba"] = 0.0
        prev.to_csv(trident_path, sep="\t", index=False)
    else:
        trident_path = os.path.join(output_dir, f"{basename}.accession.trident")
        write_empty_trident(
            trident_path,
            [
                "read_id",
                "kraken_label",
                "accession_reestimation",
                "accession(taxid)_reestimation",
                "accession_proba",
            ],
        )

    logger.warning("No accession matches found for %s. Wrote no-match outputs to %s.", basename, output_dir)
    return {"treport": treport_path, "trident": trident_path}

def to_treport(adata, taxonomic_data, save_path=None):
    counts = pd.read_csv(adata.uns["trident"]["X_file"], sep="\t", header=None)
    counts.columns = counts.columns.map({0: "status", 1: "read_id", 2: "kraken_label"})
    counts = counts.merge(
        adata.obs[["read_id", "reestimation"]],
        on="read_id", 
        how="left"
    )
    counts.fillna({"reestimation": counts["kraken_label"]}, inplace=True)
    counts["reestimation"] = counts["reestimation"].astype(int)
    counts_df = counts.value_counts("reestimation").reset_index()
    counts_df.columns = counts_df.columns.map({"reestimation": "taxid", "count": "unique_count"})
    counts_map = dict(zip(counts_df["taxid"], counts_df["unique_count"]))
    
    parent = taxonomic_data.parent
    rank_map = dict(zip(taxonomic_data.nodes["taxon"], taxonomic_data.nodes["rank_code"]))
    rank_map[0] = "NA"

    treport = {}

    def init_entry(taxid, taxid_proba, depth):
        unique_count = counts_map.get(taxid, 0)
        return TReportEntry(
            taxid=taxid,
            taxid_proba=taxid_proba,
            unique_count=unique_count,
            rank_code=rank_map.get(taxid, "U"),
            name=taxonomic_data.names.get(taxid, "Unclassified"),
            depth=depth,
        )

    for _, row in counts_df.iterrows():
        taxid = row["taxid"].item()
        leaf_unique = counts_map.get(taxid, 0)

        chain = []
        cur = taxid

        while True:
            chain.append(cur)
            parent_id = parent.get(cur)
            if parent_id is None:
                break
            cur = parent_id

        max_depth = len(chain) - 1

        for idx, node in enumerate(chain):
            depth = max_depth - idx  # top=0, leaf=max
            node_taxid_proba = proba_in_reads(node, adata, taxonomic_data)

            if node not in treport:
                treport[node] = init_entry(node, node_taxid_proba, depth)
            else:
                if depth > treport[node].depth:
                    treport[node].depth = depth

            if node != taxid:
                treport[node].update_cumulative(leaf_unique)

    treport = treport_to_df(treport)
    treport["pct"] = round(100 * treport["cumulative_count"] / treport["unique_count"].sum(), 2)
    treport["name"] = (treport["depth"] * 2).apply(lambda d: " " * d) + treport["name"]
    treport = treport.sort_values(["pct", "depth"], ascending=[False, True])
    treport = treport[["pct", "cumulative_count", "unique_count", "rank_code", "taxid_proba", "taxid", "depth", "name"]]
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        treport.drop("depth", axis=1).to_csv(save_path, sep="\t", header=None, index=False)
    return treport
