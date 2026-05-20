# models package
# ============================================================
# Model registry — maps string names to model classes.
# Adding a new model (e.g., EEG Transformer) only requires
# importing it here and adding to MODEL_REGISTRY.
# ============================================================

from models.csp_svm import CSPSVMModel
from models.cnn import CNNModel
from models.eegnet import EEGNet
from models.eeg_conformer import EEGConformer

MODEL_REGISTRY = {
    "CSP_SVM": CSPSVMModel,
    "CNN": CNNModel,
    "EEGNet": EEGNet,
    "EEGConformer": EEGConformer,
}


def get_model(name: str, **kwargs):
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        Model name (key in MODEL_REGISTRY).
    **kwargs
        Passed to the model constructor.

    Returns
    -------
    BaseModel instance
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available: {available}")

    return MODEL_REGISTRY[name](**kwargs)
