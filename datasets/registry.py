# datasets/registry.py
# ============================================================
# Dataset registry — maps string names to dataset loader classes.
# Enables dynamic dataset selection from config.
# ============================================================

from datasets.bnci2014001 import BNCI2014001Dataset
from datasets.physionet_mi import PhysionetMIDataset
from datasets.cho2017 import Cho2017Dataset
from datasets.base_dataset import BaseDataset

# Registry mapping dataset names → loader classes
DATASET_REGISTRY = {
    "BNCI2014001": BNCI2014001Dataset,
    "PhysionetMI": PhysionetMIDataset,
    "Cho2017": Cho2017Dataset,
}


def get_dataset(name: str) -> BaseDataset:
    """
    Get a dataset loader by name.

    Parameters
    ----------
    name : str
        Dataset name (must be a key in DATASET_REGISTRY).

    Returns
    -------
    BaseDataset instance
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")

    return DATASET_REGISTRY[name]()
