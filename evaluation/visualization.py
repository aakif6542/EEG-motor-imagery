# evaluation/visualization.py
# ============================================================
# Publication-quality visualization for EEG benchmarking.
#
# PRESERVED: Original plot concepts (training curves, confusion
#   matrices, model comparison bars).
# IMPROVED: Auto-save to files, seaborn styling, cross-dataset
#   comparison heatmaps, comprehensive summary plots.
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for auto-saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.io_utils import ensure_dir


from utils.logging_config import get_logger

class Visualizer:
    """
    Publication-quality visualization generator.

    All plots are automatically saved to the specified output
    directory. Uses seaborn styling for clean aesthetics.
    """

    def __init__(self, output_dir: str, dpi=150, fmt="png"):
        self.output_dir = ensure_dir(output_dir)
        self.dpi = dpi
        self.fmt = fmt
        self.logger = get_logger()

        # Set publication style
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.rcParams.update({
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'savefig.bbox': 'tight',
            'font.family': 'sans-serif',
        })

    def _save(self, fig, name):
        """Save figure and close."""
        path = os.path.join(self.output_dir, f"{name}.{self.fmt}")
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        self.logger.debug(f"Saved: {path}")
        return path

    # -----------------------------------------------------------
    # TRAINING CURVES
    # -----------------------------------------------------------
    def plot_training_curves(self, history, model_name,
                             dataset_name, prefix=""):
        """
        Plot training and validation accuracy/loss curves.

        Parameters
        ----------
        history : keras History object
        model_name : str
        dataset_name : str
        """
        if history is None:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(history.history['accuracy'],
                     linewidth=2, label='Train')
        if 'val_accuracy' in history.history:
            axes[0].plot(history.history['val_accuracy'],
                         linewidth=2, label='Validation')
        axes[0].set_title(f'{model_name} — Accuracy ({dataset_name})',
                          fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history['loss'],
                     linewidth=2, label='Train')
        if 'val_loss' in history.history:
            axes[1].plot(history.history['val_loss'],
                         linewidth=2, label='Validation')
        axes[1].set_title(f'{model_name} — Loss ({dataset_name})',
                          fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        name = f"{prefix}{dataset_name}_{model_name}_training_curves"
        return self._save(fig, name)

    # -----------------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------------
    def plot_confusion_matrix(self, y_true, y_pred, model_name,
                               dataset_name, class_names=None,
                               prefix=""):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names or ['Class 0', 'Class 1'],
            yticklabels=class_names or ['Class 0', 'Class 1'],
            ax=ax, cbar_kws={'shrink': 0.8}
        )
        ax.set_title(f'{model_name} — Confusion Matrix ({dataset_name})',
                      fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        name = f"{prefix}{dataset_name}_{model_name}_confusion_matrix"
        return self._save(fig, name)

    # -----------------------------------------------------------
    # MODEL COMPARISON (BAR CHART)
    # -----------------------------------------------------------
    def plot_model_comparison(self, results_dict, dataset_name,
                               metric="accuracy", prefix=""):
        """
        Bar chart comparing models on a single dataset.

        Parameters
        ----------
        results_dict : dict
            {model_name: metrics_dict}
        dataset_name : str
        metric : str
            Which metric to plot.
        """
        names = list(results_dict.keys())
        values = [results_dict[n][metric] for n in names]

        colors = sns.color_palette("viridis", len(names))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values, color=colors, edgecolor='white',
                      linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_title(f'Model Comparison — {dataset_name}',
                      fontweight='bold', fontsize=14)
        ax.set_ylabel(metric.capitalize())
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        name = f"{prefix}{dataset_name}_model_comparison_{metric}"
        return self._save(fig, name)

    # -----------------------------------------------------------
    # DATASET COMPARISON (GROUPED BAR CHART)
    # -----------------------------------------------------------
    def plot_dataset_comparison(self, all_results, metric="accuracy",
                                 prefix=""):
        """
        Grouped bar chart: models × datasets.

        Parameters
        ----------
        all_results : dict
            {dataset_name: {model_name: metrics_dict}}
        """
        datasets = list(all_results.keys())
        models = list(next(iter(all_results.values())).keys())

        x = np.arange(len(datasets))
        width = 0.8 / len(models)
        colors = sns.color_palette("Set2", len(models))

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            vals = [all_results[d].get(model, {}).get(metric, 0)
                    for d in datasets]
            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9,
                          label=model, color=colors[i],
                          edgecolor='white', linewidth=1)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.set_title(f'Cross-Dataset {metric.capitalize()} Comparison',
                      fontweight='bold', fontsize=14)
        ax.legend(title='Model')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

        name = f"{prefix}cross_dataset_{metric}_comparison"
        return self._save(fig, name)

    # -----------------------------------------------------------
    # CROSS-DATASET HEATMAP
    # -----------------------------------------------------------
    def plot_cross_dataset_heatmap(self, all_results,
                                    metric="accuracy", prefix=""):
        """
        Heatmap: models (rows) × datasets (columns).
        """
        datasets = list(all_results.keys())
        models = list(next(iter(all_results.values())).keys())

        matrix = np.zeros((len(models), len(datasets)))
        for j, ds in enumerate(datasets):
            for i, model in enumerate(models):
                matrix[i, j] = all_results[ds].get(model, {}).get(metric, 0)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=datasets, yticklabels=models,
            ax=ax, linewidths=2, linecolor='white',
            cbar_kws={'label': metric.capitalize(), 'shrink': 0.8}
        )
        ax.set_title(f'Cross-Dataset Benchmark — {metric.capitalize()}',
                      fontweight='bold', fontsize=14)
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')

        name = f"{prefix}cross_dataset_heatmap_{metric}"
        return self._save(fig, name)

    # -----------------------------------------------------------
    # MULTI-METRIC BAR CHART
    # -----------------------------------------------------------
    def plot_multi_metric(self, results_dict, dataset_name, prefix=""):
        """
        Grouped bar chart showing acc, precision, recall, F1
        for all models on one dataset.
        """
        models = list(results_dict.keys())
        metric_names = ['accuracy', 'precision', 'recall', 'f1']

        x = np.arange(len(models))
        width = 0.8 / len(metric_names)
        colors = sns.color_palette("muted", len(metric_names))

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, met in enumerate(metric_names):
            vals = [results_dict[m].get(met, 0) for m in models]
            offset = (i - len(metric_names) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9,
                   label=met.capitalize(), color=colors[i],
                   edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'All Metrics — {dataset_name}',
                      fontweight='bold', fontsize=14)
        ax.legend(title='Metric')
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

        name = f"{prefix}{dataset_name}_multi_metric"
        return self._save(fig, name)

    # -----------------------------------------------------------
    # SUMMARY TABLE (CSV + LaTeX)
    # -----------------------------------------------------------
    def save_summary_table(self, all_results, prefix=""):
        """
        Save cross-dataset results as CSV and LaTeX table.
        """
        import pandas as pd

        rows = []
        for dataset, models in all_results.items():
            for model_name, metrics in models.items():
                rows.append({
                    'Dataset': dataset,
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0),
                    'TrainTime(s)': metrics.get('train_time_seconds', 0),
                    'InfTime(s)': metrics.get('inference_time_seconds', 0),
                })

        df = pd.DataFrame(rows)

        # Save CSV
        csv_path = os.path.join(self.output_dir, f"{prefix}benchmark_results.csv")
        df.to_csv(csv_path, index=False, float_format='%.4f')
        self.logger.info(f"Saved: {csv_path}")

        # Save LaTeX
        latex_path = os.path.join(self.output_dir, f"{prefix}benchmark_results.tex")
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format='%.4f'))
        self.logger.info(f"Saved: {latex_path}")

        return csv_path, latex_path
