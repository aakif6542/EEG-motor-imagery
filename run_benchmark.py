# run_benchmark.py
# ============================================================
# Main entry point for the EEG Benchmarking Framework.
#
# Usage:
#   python run_benchmark.py                    # Run full benchmark
#   python run_benchmark.py --datasets BNCI2014001 PhysionetMI
#   python run_benchmark.py --models CSP_SVM EEGNet
#   python run_benchmark.py --epochs 20
# ============================================================

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import setup_logging, suppress_tf_logging, get_logger
import logging

# Suppress TF logging early
suppress_tf_logging()

# Configure centralized cache for MOABB/MNE using pathlib for robust absolute paths
cache_dir_abs = str(Path(__file__).resolve().parent / "datasets_cache")
os.makedirs(cache_dir_abs, exist_ok=True)

# MOABB has a bug where it sanitizes absolute Windows paths by replacing ':' with '-'
# (e.g. 'E:' becomes 'E-'), causing downloads to go into malformed nested directories.
# We bypass this by stripping the drive letter. The path remains absolute on the current drive.
_, cache_dir_no_drive = os.path.splitdrive(cache_dir_abs)
cache_dir = cache_dir_no_drive.replace("\\", "/")

# Set environment variables for MNE
os.environ["MNE_DATA"] = cache_dir
os.environ["MNE_DATASETS_BNCI_PATH"] = cache_dir
os.environ["MNE_DATASETS_PHYSIONET_MI_PATH"] = cache_dir
os.environ["MNE_DATASETS_CHO2017_PATH"] = cache_dir
os.environ["MNE_DATASETS_EEGBCI_PATH"] = cache_dir

# Set MNE config and MOABB config globally
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import mne
        mne.set_config("MNE_DATA", cache_dir)
        mne.set_config("MNE_DATASETS_BNCI_PATH", cache_dir)
        mne.set_config("MNE_DATASETS_PHYSIONET_MI_PATH", cache_dir)
        mne.set_config("MNE_DATASETS_CHO2017_PATH", cache_dir)
        mne.set_config("MNE_DATASETS_EEGBCI_PATH", cache_dir)
        
        # Overwrite any existing MNE dataset paths to prevent C-/ or E-/ bugs
        for key in mne.get_config().keys():
            if key.startswith("MNE_DATASETS_") and key.endswith("_PATH"):
                mne.set_config(key, cache_dir)
                
        mne.set_config('MNE_LOGGING_LEVEL', 'WARNING')
    except Exception:
        pass

    try:
        import moabb
        moabb.set_download_dir(cache_dir)
    except Exception:
        pass

from configs.experiment_config import ExperimentConfig
from experiments.runner import ExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="EEG Motor Imagery Benchmarking Framework"
    )

    parser.add_argument(
        "--datasets", nargs="+",
        default=None,
        help="Datasets to benchmark (e.g., BNCI2014001 PhysionetMI Cho2017)"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=None,
        help="Models to train (e.g., CSP_SVM CNN EEGNet EEGConformer)"
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Number of training epochs for DL models"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for DL models"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="eeg_benchmark",
        help="Name for this experiment run"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        random_seed=args.seed,
    )

    # Override datasets if specified
    if args.datasets:
        config.dataset.datasets = args.datasets

    # Override models if specified
    if args.models:
        config.model.models = args.models

    # Override training parameters
    config.model.epochs = args.epochs
    config.model.batch_size = args.batch_size
    config.model.learning_rate = args.lr

    # Augmentation toggle
    if args.no_augment:
        config.preprocess.augment = False

    # Setup central logging
    # Note: we need to pass the results directory to setup_logging to save the file
    base_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.results_dir)
    results_path = os.path.join(base_results_dir, config.experiment_name)
    logger = setup_logging(log_dir=os.path.join(results_path, "logs"), verbose=config.verbose)

    logger.info(f"Dataset cache directory: {cache_dir_abs}")

    # Run
    runner = ExperimentRunner(config)
    results = runner.run()

    return results


if __name__ == "__main__":
    main()
