# evaluation/metrics.py
# ============================================================
# Metric computation for EEG classification evaluation.
# Computes accuracy, precision, recall, F1, and confusion matrix.
# ============================================================

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from typing import Dict, Any


def compute_metrics(y_true, y_pred, class_names=None) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    class_names : list of str, optional
        Class names for the report.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1,
    confusion_matrix, classification_report
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(
            y_true, y_pred, average='weighted', zero_division=0)),
        "f1": float(f1_score(
            y_true, y_pred, average='weighted', zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        ),
    }

    return metrics


def format_metrics_table(all_results: dict) -> str:
    """
    Format results into a readable ASCII table.

    Parameters
    ----------
    all_results : dict
        Nested dict: {dataset: {model: metrics_dict}}

    Returns
    -------
    str : formatted table
    """
    lines = []
    header = f"{'Dataset':<15} {'Model':<12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Train(s)':>8} {'Inf(s)':>8}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))

    for dataset, models in all_results.items():
        for model_name, metrics in models.items():
            train_t = metrics.get('train_time_seconds', 0)
            inf_t = metrics.get('inference_time_seconds', 0)
            line = (f"{dataset:<15} {model_name:<12} "
                    f"{metrics['accuracy']:>7.4f} "
                    f"{metrics['precision']:>7.4f} "
                    f"{metrics['recall']:>7.4f} "
                    f"{metrics['f1']:>7.4f} "
                    f"{train_t:>8.1f} "
                    f"{inf_t:>8.3f}")
            lines.append(line)
        lines.append("-" * len(header))

    return "\n".join(lines)
