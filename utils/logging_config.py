# utils/logging_config.py
# ============================================================
# Centralized logging & verbosity control for the EEG
# benchmarking framework.
#
# Suppresses noisy MNE/TensorFlow/MOABB preprocessing logs
# while keeping concise, research-friendly progress output.
# ============================================================

import os
import sys
import logging
import warnings
from datetime import datetime


def setup_logging(log_dir: str = None, verbose: bool = False):
    """
    Configure logging for the entire framework.

    - Suppresses repetitive MNE FIR filter / epoch creation logs
    - Suppresses TensorFlow INFO and WARNING spam
    - Keeps important warnings and errors visible
    - Optionally writes full debug log to file

    Parameters
    ----------
    log_dir : str, optional
        Directory to write log file to. If None, no file logging.
    verbose : bool
        If True, show more detail. If False, minimal output.
    """
    # ----------------------------------------------------------
    # 1. Suppress TensorFlow verbosity
    # ----------------------------------------------------------
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # ERROR only
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'        # Suppress oneDNN

    # ----------------------------------------------------------
    # 2. Suppress MNE verbose logging
    # ----------------------------------------------------------
    import mne
    mne.set_log_level('WARNING')                      # Suppress INFO/DEBUG

    # ----------------------------------------------------------
    # 3. Suppress MOABB verbose logging
    # ----------------------------------------------------------
    logging.getLogger('moabb').setLevel(logging.WARNING)
    logging.getLogger('moabb.datasets').setLevel(logging.WARNING)
    logging.getLogger('moabb.paradigms').setLevel(logging.WARNING)

    # ----------------------------------------------------------
    # 4. Suppress other noisy loggers
    # ----------------------------------------------------------
    logging.getLogger('mne').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('pooch').setLevel(logging.WARNING)

    # Suppress sklearn convergence / future warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Concatenation of Annotations.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*non-standard config.*')
    warnings.filterwarnings('ignore', message='.*FIR.*')
    warnings.filterwarnings('ignore', message='.*filter.*')
    warnings.filterwarnings('ignore', message='.*InsecureRequestWarning.*')
    warnings.filterwarnings('ignore', message='.*Unverified HTTPS request.*')

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    # ----------------------------------------------------------
    # 5. Configure framework logger
    # ----------------------------------------------------------
    logger = logging.getLogger('eeg_benchmark')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler — concise format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler — detailed format (if log_dir provided)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger():
    """Get the framework logger (call setup_logging first)."""
    return logging.getLogger('eeg_benchmark')


def suppress_tf_logging():
    """
    Suppress TensorFlow logging. Call BEFORE importing tensorflow.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # If TF is already imported, try to set its logger
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
    except ImportError:
        pass
