# utils/io_utils.py
# ============================================================
# I/O utilities for directory management, result serialization,
# and model persistence.
# ============================================================

import os
import json
import numpy as np
import pickle
from datetime import datetime


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file (handles numpy types)."""
    ensure_dir(os.path.dirname(filepath))

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(filepath: str) -> dict:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def save_pickle(obj, filepath: str):
    """Save object using pickle."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str):
    """Load object from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_timestamp() -> str:
    """Get formatted timestamp string for naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_results_directory(base_dir: str, experiment_name: str) -> dict:
    """
    Create organized results directory structure.

    Returns dict with paths to each subdirectory:
        - models/   : saved model weights
        - metrics/  : JSON/CSV metric files
        - plots/    : visualization outputs
        - tables/   : comparison tables
        - logs/     : experiment logs
    """
    root = os.path.join(base_dir, experiment_name)

    paths = {
        "root": ensure_dir(root),
        "models": ensure_dir(os.path.join(root, "models")),
        "metrics": ensure_dir(os.path.join(root, "metrics")),
        "plots": ensure_dir(os.path.join(root, "plots")),
        "tables": ensure_dir(os.path.join(root, "tables")),
        "logs": ensure_dir(os.path.join(root, "logs")),
    }

    return paths


def log_experiment_start(results_paths: dict, config) -> str:
    """Log experiment metadata at start."""
    from dataclasses import asdict

    meta = {
        "timestamp": get_timestamp(),
        "config": asdict(config),
    }

    meta_path = os.path.join(results_paths["root"], "experiment_meta.json")
    save_json(meta, meta_path)

    return meta_path
