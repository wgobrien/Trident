---
name: trident-codebase
description: Use when working in the Trident repository or writing scripts that call Trident repeatedly across many samples. Follow this to avoid breaking the global cache model, to reuse taxonomic_data efficiently, and to preserve the expected load_adata/load_accession_adata workflow.
---

# Trident Codebase

Read this before changing pipeline scripts or writing batch/sweep code around Trident.

## Non-Negotiables

- Treat `TRIDENT_GLOBALS` as true global process state.
- Set `TRIDENT_GLOBALS` once, outside the script, before importing `trident`.
- Do not rewrite `TRIDENT_GLOBALS` per sample, per config, or per helper function.
- Intermediate data and saved models live under `TRIDENT_GLOBALS/<sample>/`.
- Reports such as `.trident`, `.treport`, `.accession.treport`, summaries, and figures belong in an explicit output directory, not under a rewritten `TRIDENT_GLOBALS`.
- In scripts and docs, prefer `python3` over bare `python` unless the environment is explicitly controlled.
- If the user asks for "script only" or says not to run it, do not execute the pipeline. Syntax-only checks such as `python3 -m py_compile script.py` are acceptable only if they do not trigger workflow side effects.

## Architecture And Function Discovery

- `trident/__init__.py` exposes the high-level API: `trident.run_trident`, `trident.estimate_accessions`, `trident.tl` for tools, and `trident.pl` for plotting.
- `trident/cli.py` maps CLI commands to Python entry points. Read this first when translating `trident ...`, `trident accession ...`, or `trident check ...` into script code.
- `trident/reestimate.py` is the taxid-level pipeline entry point behind bare `trident`.
- `trident/estimate_accessions.py` is the accession-level pipeline entry point behind `trident accession`.
- `trident/check.py` is the validation pipeline behind `trident check`; use it to identify local alignment, coverage, clustering, assembly, and BLAST steps.
- `trident/tools/__init__.py` is the best quick index of commonly intended tool functions exposed as `trident.tl.*`.
- Use `trident/tools/data.py` for AnnData construction/loading, read-pair discovery, accession counts, and cache-aware loaders.
- Use `trident/tools/model.py`, `trident/tools/train.py`, and `trident/tools/evals.py` for config, model fitting, posteriors, read probabilities, clustering/LCA, and reestimation.
- Use `trident.tl.taxutils(config)` to load taxonomy and apply Trident's library-specific target filtering. Use the returned object directly for accession-to-taxid mapping, LCA, parent trees, target taxa, and taxid/accession queries.
- Use `trident/tools/align_tl.py` and `trident/plot/align_pl.py` for validation outputs: read alignment, coverage tables/plots, consensus contigs, and BLAST annotation.
- When unsure where a function lives, inspect `trident/tools/__init__.py`, then use `rg -n "def function_name|class ClassName" trident`.

## Standard Multi-Sample Pattern

- Create one `taxonomic_data = trident.tl.taxutils(config)` per compatible library/database run and reuse it across all samples.
- Pass that same object into `trident.run_trident(...)`, `trident.estimate_accessions(...)`, `load_adata(...)`, and `load_accession_adata(...)`.
- Do not call `trident.tl.taxutils(config)` inside a per-sample loop unless a fresh taxonomy object is intentional.
- Reuse one taxonomy object only while samples share the same reference library assumptions. `trident.tl.taxutils(config)` returns an object whose `target_taxa` are already matched and whose `_is_matched` flag is set.
- If switching libraries in the same process, build a new taxonomy object and match the new library separately.
- Prefer the standard pipeline entry points unless there is a concrete reason to reproduce lower-level steps.

Minimal cache-aware loading pattern:

```python
import trident

taxonomic_data = trident.tl.taxutils(configs[0])

for config in configs:
    adata = trident.tl.load_adata(config, taxonomic_data, rebuild=False)
    acc_adata = trident.tl.load_accession_adata(config, taxonomic_data, adata=adata, rebuild=False)
```

## Cache And Output Model

- `load_adata()` and `load_accession_adata()` already know where to find per-sample cached data and trained models through `TRIDENT_GLOBALS`.
- The sample cache key is the read basename. Keep reads/library/database/config consistent if you expect to reuse cached data.
- `load_adata()` requires `taxonomic_data` when data has not been built yet. Only cached reloads can avoid rebuilding.
- Use `rebuild=False` unless intentionally rerunning Kraken and rebuilding counts.
- Use `low_memory=False` unless disk constraints require otherwise; SQLite-backed accession lookup is the normal fast path.

## Previous Run Selection By Config

- Saved model runs live under `TRIDENT_GLOBALS/<basename>/<run_id>/`, where `<basename>` is the read basename from `get_basename_from_reads(config.reads)`.
- Taxid model run IDs are named like `<basename>-YYYYMMDD-HHMMSS`; accession model run IDs are named like `<basename>.accessions-YYYYMMDD-HHMMSS`.
- Each saved run directory should contain `config.json`, `trident_model.pt`, `trident_model_param_store.pt`, and usually `losses.pt` / `optimizer_state.pt`.
- To reuse a previous model with particular hyperparameters, do not retrain and do not rebuild AnnData. Load cached `adata` / `acc_adata`, scan matching run directories, read each `config.json`, choose the desired run, set `adata.uns["trident"]["run_id"]`, then call `trident.tools.utils.load_model(adata)`.
- `alpha_pi`, `beta_pi`, `num_latent`, `concentration`, and similar values are model/training parameters stored in `config.json`; they are not data-building parameters. Changing only those should reuse cached AnnData/counts.
- For one-at-a-time threshold replay, pick the saved run whose `config.json` matches the requested prior (for example `alpha_pi == 1.0` and `beta_pi == 1.0`) and then choose the highest available `num_latent` if that is the intended source model.
- Validate library/database compatibility when scanning prior runs. Compare `config.json["library"]` and `config.json["database"]` to the active paths when those fields are present.
- `load_model(adata, name=...)` validates `name` against `adata.uns["trident"]["run_id_history"]`. If selecting by directory scan instead, either pass through that history correctly or set `adata.uns["trident"]["run_id"]` and ensure the selected run id appears in `run_id_history` before loading.
- Be careful with helpers that save back into the selected run. `get_posteriors(adata)` calls `save_model(..., overwrite=True)` and can rewrite the run selected via `run_id_history`. If the saved model already contains cached posterior tensors, prefer `load_model(adata)`, copy `model.posterior_pi` into `adata.var["pi"]`, and call `_get_read_probabilities(adata)` before `reestimate(...)`.
- For accession models, load cached accession AnnData with `load_accession_adata(config.copy(), taxonomic_data, adata=adata, rebuild=False)`, scan for `<basename>.accessions-*` runs, select the matching `config.json`, load that run id, and only ensure `acc_adata.var["taxid"]` is mapped before accession reestimation.

Minimal previous-run selection pattern:

```python
import json
from pathlib import Path

import trident
from trident.global_utils import get_sample_folder
from trident.tools.utils import load_model


def matching_runs(adata, *, alpha_pi, beta_pi, accession=False):
    basename = adata.uns["trident"]["basename"]
    prefix = f"{basename}.accessions" if accession else basename
    sample_folder = Path(get_sample_folder(basename))

    matches = []
    for run_dir in sample_folder.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(f"{prefix}-"):
            continue
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        cfg = json.loads(config_path.read_text())
        if float(cfg.get("alpha_pi", "nan")) != alpha_pi:
            continue
        if float(cfg.get("beta_pi", "nan")) != beta_pi:
            continue
        matches.append((int(cfg["num_latent"]), run_dir.name, cfg))
    return sorted(matches)


adata = trident.tl.load_adata(config, taxonomic_data, rebuild=False)
num_latent, run_id, cfg = matching_runs(adata, alpha_pi=1.0, beta_pi=1.0)[-1]
adata.uns["trident"]["run_id"] = run_id
history = list(adata.uns["trident"].get("run_id_history", []))
adata.uns["trident"]["run_id_history"] = [r for r in history if r != run_id] + [run_id]
adata.uns["trident"]["config"] = cfg

model = load_model(adata)
adata.var["pi"] = model.posterior_pi.squeeze().detach().cpu().numpy()
trident.tl._get_read_probabilities(adata)
trident.tl.reestimate(adata)
```

## Accession Workflow

- `load_accession_adata()` is not a standalone first step. It depends on a completed taxid-level stage.
- Required taxid-level state includes `adata.uns["trident"]["run_id"]` and `adata.var["marginal_proba"]`.
- For standard behavior, call `trident.estimate_accessions(args, taxonomic_data=shared_taxonomic_data)`. It will run taxid fitting first if needed.
- Use lower-level calls only when writing specialized validation/reporting code.

Lower-level sequence:

```text
load_adata -> train_trident -> get_posteriors -> _get_read_probabilities
-> var_marginal -> load_accession_adata
```

## Throughput And Sweeps

- Do not rely on repository example files being present; conda/package installs may not include `examples/`.
- For many samples, use a two-stage batch pattern directly in the script.
- Stage 1: create one shared `taxonomic_data` object, loop over read pairs, build `TridentConfig`, and call `load_adata(config, taxonomic_data, rebuild=False)` to build/cache data.
- Stage 2: loop over the same read pairs and call `trident.run_trident(...)` or `trident.estimate_accessions(...)` with the same `taxonomic_data`.
- This keeps taxonomy/accession lookup warm, reuses normal Trident caches, and avoids repeated Kraken/database setup overhead.
- When varying only model hyperparameters, reuse cached reads/library/database data under `TRIDENT_GLOBALS`; do not rebuild counts.
- When varying only downstream reporting/evaluation, reload cached AnnData/model outputs rather than retraining.
- Add `--skip-existing` style checks around expensive outputs such as `.treport`, `.accession.treport`, and final summary CSVs.

## Selected-Sample Scripts
- Depends on structure of user's file naming, this may or may not apply.
- Discover read pairs defensively. Match sample IDs before `_R1_`/`_R2_` with delimiter-aware logic so `B3` does not accidentally match `B30`.
- Fail clearly on missing pairs, missing mates, or multiple candidate pairs.
- Preserve user-provided sample order when writing manifests and summaries.

## Check-Equivalent Validation

- `trident check` currently builds its own taxonomy object internally. If strict shared-taxonomy reuse matters, mirror the check steps in-process instead of shelling out, and pass the same shared `taxonomic_data` to `load_accession_adata`.
- The local check-equivalent pieces are: load accession AnnData, get posteriors, populate accession taxids, `_get_read_probabilities`, `reestimate`, `var_marginal`, `read_cluster`, `align`, `coverage`, `cluster_consensus`, `blast_contigs`, and `checkpoint`.
- `coverage(...)` returns a dataframe but does not persist CSV output by itself. If downstream comparison needs coverage values, explicitly save the returned dataframe.
- `cluster_consensus` depends on `megahit`; alignment depends on `minimap2`; `blast_contigs` calls NCBI BLAST over the network and may be slow or unavailable. Make these runtime expectations explicit in generated scripts.
- For agreement summaries, compare `.trident` accession/taxid reestimation against local validation at the same sample scope: best alignment hit per read from `reads_to_accessions.sam`, coverage by accession, and BLAST-annotated assembly contigs by read cluster. Produce both per-sample and overall summaries.

## Taxonomy Mapping Pitfall

- `taxutils.load_a2t(...)` replaces `taxonomic_data.a2t` unless `extend=True` is passed. In long-running shared-taxonomy scripts, extend or merge newly loaded accession mappings rather than discarding the warmed library map.
- Before accession reestimation or validation, ensure `acc_adata.var["taxid"]` is populated from `taxonomic_data.a2t`.

```python
def ensure_a2t(taxonomic_data, accessions, sqlite=True):
    accessions = sorted({str(a) for a in accessions if str(a) and str(a) != "nan"})
    taxonomic_data.load_a2t(accessions, low_memory=not sqlite, extend=True)

ensure_a2t(taxonomic_data, list(acc_adata.var.accession.values), sqlite=config.sqlite)
acc_adata.var["taxid"] = acc_adata.var["accession"].map(taxonomic_data.a2t)
```

## Files To Inspect

- [README.md](README.md): user-facing setup and `TRIDENT_GLOBALS` contract
- [trident/global_utils.py](trident/global_utils.py): global save-folder behavior
- [trident/cli.py](trident/cli.py): CLI command dispatch and argument shape
- [trident/tools/__init__.py](trident/tools/__init__.py): public `trident.tl.*` tool index
- [taxutils](https://github.com/wgobrien/taxutils): taxonomy loading, tree queries, and accession-to-taxid loading
- [trident/tools/data.py](trident/tools/data.py): `load_adata()`, `load_accession_adata()`, and library-specific target matching
- [trident/reestimate.py](trident/reestimate.py): taxid-level pipeline entry point
- [trident/estimate_accessions.py](trident/estimate_accessions.py): accession-level pipeline entry point
- [trident/check.py](trident/check.py): local alignment/coverage/assembly validation workflow
- [trident/tools/align_tl.py](trident/tools/align_tl.py): alignment, consensus assembly, and BLAST helpers
