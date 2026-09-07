import torch
import pyro
from pyro.infer import SVI, TraceEnum_ELBO, Trace_ELBO
from pyro.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import trange

from .data import TridentDataset, to_torch
from .model import Trident
from .utils import save_model, load_model, TridentConfig
from trident.global_utils import get_logger, is_notebook


logger = get_logger(__name__)


def train_trident(
    adata,
    config: TridentConfig,
    continue_training: bool = False,
    verbose: bool = True,
    save_loss_plot: str = None,
    build_outputs: bool = False,
):
    """Train a model, or reuse the selected saved model when rebuilding outputs.

    Returns ``True`` when training ran and ``False`` when ``build_outputs``
    successfully reused an existing model.
    """
    if build_outputs:
        try:
            load_model(adata)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "--build-outputs requested, but no trained model is available "
                "(%s). Training a new model.",
                exc,
            )
            # There is no model to continue, even if both flags were supplied.
            continue_training = False
        else:
            logger.info(
                "Using trained model %s to rebuild outputs; skipping training.",
                adata.uns["trident"]["run_id"],
            )
            return False

    torch.set_num_threads(config.threads)
    config.num_reads = adata.shape[0]
    config.num_targets = adata.shape[1]
    config.subsample_size = config.subsample_size if config.subsample_size is not None else config.num_reads
    _, L = to_torch(adata)
    L = L.to(config.device)

    dataset = TridentDataset(adata)
    batch_size = min(config.subsample_size, config.num_reads)

    if continue_training and "run_id" in adata.uns.get("trident", {}):
        model_obj = load_model(adata)
    else:
        model_obj = Trident(config)
    model_obj.to(config.device)
    losses, optimizer_state = train_loop(
        model_obj.model, model_obj.guide, dataset, L,
        batch_size=batch_size, steps=config.n_steps,
        clear_param_store=not continue_training,
        verbose=verbose, device=config.device,
        optimizer_state=model_obj.optimizer,
        seed=config.seed,
    )
    model_obj.losses.extend(losses)
    model_obj.optimizer = optimizer_state
    if verbose:
        plot_loss(model_obj.losses, save_path=save_loss_plot)
    save_model(adata, model_obj, overwrite=continue_training, simulation=False)
    return True

def plot_loss(losses: list, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.lineplot(x=range(len(losses)), y=losses, ax=axes[0])
    sns.lineplot(x=range(len(losses)), y=np.log(losses), ax=axes[1])
    axes[0].set_title("Loss")
    axes[1].set_title("Log Loss")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if is_notebook():
        plt.show()
    plt.close()

def train_loop(
    model, guide,
    dataset, L,
    batch_size=512,
    steps=500,
    lr=1e-2,
    lrd=5e-3,
    verbose=True,
    clear_param_store=False,
    device="cpu",
    optimizer_state=None,
    seed=0,
):
    if clear_param_store is True:
        pyro.clear_param_store()
    optim = Adam({"lr": lr})
    svi = SVI(model, guide, optim, loss=Trace_ELBO(max_plate_nesting=2))
    
    if optimizer_state is not None:
        optim.set_state(optimizer_state)
    
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))
    position = 0
    it = trange(steps, desc="train", leave=True) if verbose else range(steps)
    losses = []
    for step in it:
        if position + batch_size > len(order):
            order = rng.permutation(len(dataset))
            position = 0
        batch_ids = order[position:position + batch_size]
        position += batch_size

        X_batch = torch.stack([dataset[int(idx)][0] for idx in batch_ids])
        r_idx = torch.as_tensor(batch_ids, dtype=torch.long)
        X_batch = X_batch.to(device=device, dtype=L.dtype)
        r_idx = r_idx.to(device)
        loss = svi.step(X_batch, L, r_idx)
        losses.append(loss)
        if verbose and step % 10 == 0:
            it.set_postfix({"loss": f"{np.mean(losses[-10:]):.3f}", "step": step})
    return losses, optim.get_state()
