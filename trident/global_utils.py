# global_utils.py

import json
import os
import sys
import time
from contextlib import ContextDecorator
from datetime import datetime, timezone
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

TRIDENT_GLOBALS = dict()
TRIDENT_GLOBALS["save_folder"] = os.path.expanduser(os.environ.get("TRIDENT_GLOBALS", "./_trident/"))
os.environ.setdefault("TAXUTILS_GLOBALS", TRIDENT_GLOBALS["save_folder"])
TRIDENT_GLOBALS["pathogen_dict_urls"] = [
    "https://raw.githubusercontent.com/wgobrien/Trident/main/pathogen_dict.json",
    "https://web.cs.ucla.edu/~wob/projects/taxutils/targets.json",
]
TRIDENT_GLOBALS["UNCLASSIFIED"] = 0
TRIDENT_GLOBALS["UNMAPPED"] = -1

def _utcnow_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _normalize_maxrss_bytes(ru_maxrss):
    if ru_maxrss is None:
        return None
    if sys.platform == "darwin":
        return int(ru_maxrss)
    return int(ru_maxrss * 1024)

def _get_ru_maxrss_bytes(who):
    if resource is None:
        return None
    try:
        return _normalize_maxrss_bytes(resource.getrusage(who).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None

def _format_bytes(num_bytes):
    if num_bytes is None:
        return "n/a"
    gib = num_bytes / float(1024 ** 3)
    if gib >= 1:
        return f"{gib:.2f} GiB"
    mib = num_bytes / float(1024 ** 2)
    return f"{mib:.1f} MiB"

def _namespace_to_dict(args):
    if args is None:
        return {}
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out

def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)

class RunResourceTracker:
    def __init__(self, command_name, args, logger=None):
        self.command_name = command_name
        self.args = args
        self.logger = logger or get_logger(__name__)
        reads = getattr(args, "reads", command_name)
        self.basename = get_basename_from_reads(reads)
        self.stage_name = {
            "reestimate": "taxa",
            "taxa": "taxa",
            "accession": "accession",
            "check": "check",
        }.get(command_name, command_name)
        self.output_dir = os.path.abspath(getattr(args, "output", "."))
        self.output_path = os.path.join(
            self.output_dir,
            f"{self.basename}.runtime.json",
        )
        self.started_at = _utcnow_iso()
        self._start_perf = time.perf_counter()
        self.extra = {}
        self.summary = None
        self.steps = {}

    def __enter__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        return self

    def mark(self, **kwargs):
        self.extra.update(kwargs)

    def step(self, name, **metadata):
        return _TrackedStep(self, name, metadata)

    def finalize(self, status="completed", error=None):
        if self.summary is not None:
            return self.summary

        elapsed_seconds = time.perf_counter() - self._start_perf
        max_ram_self_bytes = _get_ru_maxrss_bytes(resource.RUSAGE_SELF) if resource is not None else None
        max_ram_children_bytes = _get_ru_maxrss_bytes(resource.RUSAGE_CHILDREN) if resource is not None else None
        max_ram_bytes = max(
            [value for value in (max_ram_self_bytes, max_ram_children_bytes) if value is not None],
            default=None,
        )
        summary = {
            "command": self.command_name,
            "stage": self.stage_name,
            "status": status,
            "started_at_utc": self.started_at,
            "finished_at_utc": _utcnow_iso(),
            "wallclock_seconds": round(elapsed_seconds, 3),
            "max_ram_bytes": max_ram_bytes,
            "max_ram_scope": "max(max_ram_self_bytes, max_ram_children_bytes)",
            "max_ram_human": _format_bytes(max_ram_bytes),
            "max_ram_self_bytes": max_ram_self_bytes,
            "max_ram_children_bytes": max_ram_children_bytes,
            "output_dir": self.output_dir,
            "pid": os.getpid(),
            "args": _namespace_to_dict(self.args),
            **self.extra,
        }
        if self.steps:
            summary["steps"] = self.steps
        if error is not None:
            summary["error"] = error

        payload = {}
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[self.stage_name] = _json_safe(summary)

        with open(self.output_path, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

        self.logger.info(
            "%s runtime | wallclock=%.2fs | max_ram=%s | metrics=%s",
            self.stage_name,
            elapsed_seconds,
            summary["max_ram_human"],
            self.output_path,
        )
        self.summary = summary
        return summary

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.finalize(status="completed")
        else:
            self.finalize(status="failed", error=f"{exc_type.__name__}: {exc}")
        return False

class _TrackedStep(ContextDecorator):
    def __init__(self, tracker, name, metadata=None):
        self.tracker = tracker
        self.name = name
        self.metadata = dict(metadata or {})
        self._start_perf = None

    def __enter__(self):
        self._start_perf = time.perf_counter()
        return self

    def mark(self, **kwargs):
        self.metadata.update(kwargs)

    def __exit__(self, exc_type, exc, tb):
        elapsed_seconds = time.perf_counter() - self._start_perf
        entry = {
            "status": "completed" if exc_type is None else "failed",
            "wallclock_seconds": round(elapsed_seconds, 3),
            **self.metadata,
        }
        if exc is not None:
            entry["error"] = f"{exc_type.__name__}: {exc}"
        self.tracker.steps[self.name] = entry
        return False

def _get_var_label(adata):
    """Determine whether to use 'accession' or 'taxid' based on adata.var columns."""
    return "accession" if "accession" in adata.var.columns else "taxid"

def get_sample_folder(basename):
    save_folder = os.environ.get("TRIDENT_GLOBALS", "./_trident/")
    save_folder = os.path.expanduser(save_folder)
    folder = os.path.join(save_folder, basename)
    os.makedirs(folder, exist_ok=True)
    return folder

def get_basename_from_reads(reads):
    return os.path.basename(reads).replace('.fastq', '').replace('.gz', '')

def get_database_name(database):
    """Return a stable database directory name, ignoring trailing separators."""
    return Path(database).name

def create_trident_uns_key(adata):
    """
    Create the `trident` uns key if it doesn't exist.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    """
    if "trident" not in adata.uns:
        adata.uns["trident"] = dict()
    if "save_data" not in adata.uns["trident"]:
        adata.uns["trident"]["save_data"] = dict()

def is_notebook():
    """
    Check if the script is running in a Jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False
    except NameError:
        return False

def get_logger(name):
    import logging
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s : %(message)s",
        level=logging.INFO,
    )
    return logger
