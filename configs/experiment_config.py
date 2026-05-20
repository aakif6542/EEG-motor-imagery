# configs/experiment_config.py
# ============================================================
# Centralized experiment configuration for the EEG benchmarking
# framework. Uses Python dataclasses for type safety without
# external YAML/JSON dependencies.
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


# ---------------------------------------------------------------
# DATASET CONFIGURATION
# ---------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    # Which datasets to benchmark (keys from registry)
    datasets: List[str] = field(default_factory=lambda: [
        "BNCI2014001",
        "PhysionetMI",
        "Cho2017",
    ])

    # Number of subjects to use for training (rest go to test)
    # If None, uses dataset-specific defaults
    train_subject_ratio: float = 0.7

    # Time window for epoching (seconds)
    tmin: float = 0.5
    tmax: float = 3.5

    # Target sample rate (Hz) — all datasets resampled to this
    target_sfreq: float = 128.0

    # Event selection (left/right motor imagery)
    # Handled per dataset, but this controls binary vs multi-class
    binary_classification: bool = True


# ---------------------------------------------------------------
# PREPROCESSING CONFIGURATION
# ---------------------------------------------------------------
@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""

    # Bandpass filter (Hz)
    bandpass_low: float = 8.0
    bandpass_high: float = 30.0

    # Normalization
    normalize: bool = True

    # Data augmentation (training only)
    augment: bool = True
    augment_noise: bool = True
    augment_noise_level: float = 0.01
    augment_time_shift: bool = True
    augment_max_shift: int = 50
    augment_amplitude_scaling: bool = True
    augment_scale_range: Tuple[float, float] = (0.9, 1.1)


# ---------------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------------
@dataclass
class ModelConfig:
    """Configuration for model selection and hyperparameters."""

    # Which models to train (keys from model registry)
    models: List[str] = field(default_factory=lambda: [
        "CSP_SVM",
        "CNN",
        "EEGNet",
        "EEGConformer",
    ])

    # CSP+SVM parameters
    csp_n_components: int = 4
    svm_kernel: str = "linear"

    # Deep learning shared parameters
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.001

    # EEGNet-specific parameters
    eegnet_dropout: float = 0.5
    eegnet_F1: int = 8
    eegnet_D: int = 2
    eegnet_F2: int = 16
    eegnet_kernel_length: int = 64

    # CNN-specific parameters
    cnn_dropout: float = 0.5

    # EEG Conformer-specific parameters
    conformer_d_model: int = 64
    conformer_num_heads: int = 4
    conformer_num_blocks: int = 2
    conformer_ff_dim: int = 128
    conformer_conv_F1: int = 16
    conformer_conv_F2: int = 32
    conformer_conv_kernel1: int = 25
    conformer_conv_kernel2: int = 15
    conformer_dropout: float = 0.3
    conformer_learning_rate: float = 0.0005


# ---------------------------------------------------------------
# EVALUATION CONFIGURATION
# ---------------------------------------------------------------
@dataclass
class EvalConfig:
    """Configuration for evaluation and visualization."""

    # Metrics to compute
    compute_accuracy: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    compute_f1: bool = True
    compute_confusion_matrix: bool = True

    # Plots to generate
    plot_training_curves: bool = True
    plot_confusion_matrices: bool = True
    plot_model_comparison: bool = True
    plot_dataset_comparison: bool = True
    plot_cross_dataset_heatmap: bool = True

    # Output formats
    save_csv: bool = True
    save_latex: bool = True

    # Plot style
    figure_dpi: int = 150
    figure_format: str = "png"


# ---------------------------------------------------------------
# MASTER EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Top-level configuration combining all sub-configs."""

    # Sub-configs
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # Experiment metadata
    experiment_name: str = "eeg_benchmark"
    random_seed: int = 42

    # Output directory (relative to project root)
    results_dir: str = "results"

    # Verbose logging
    verbose: bool = True

    def get_results_path(self, base_dir: str) -> str:
        """Get full results path for this experiment."""
        return os.path.join(base_dir, self.results_dir, self.experiment_name)
