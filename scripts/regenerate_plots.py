# regenerate_plots.py
# ============================================================
# Regenerate ALL plots from saved metrics/results.
# Reads from results/eeg_benchmark/ (which has all 4 models
# including EEGConformer) and writes updated plots to
# results/final_results/.
#
# Usage:
#   python regenerate_plots.py
#
# Does NOT retrain any models.
# ============================================================

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.visualization import Visualizer
from utils.io_utils import ensure_dir

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Source: latest benchmark run (has all 4 models)
SOURCE_DIR = os.path.join(PROJECT_ROOT, "results", "eeg_benchmark")

# Destination: final_results (needs updating)
DEST_DIR = os.path.join(PROJECT_ROOT, "results", "final_results")


def load_all_results(source_dir):
    """Load all_results.json from the source directory."""
    path = os.path.join(source_dir, "all_results.json")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        sys.exit(1)

    with open(path, "r") as f:
        return json.load(f)


def load_per_model_metrics(source_dir):
    """Load individual model metric JSON files."""
    metrics_dir = os.path.join(source_dir, "metrics")
    metrics = {}
    if not os.path.exists(metrics_dir):
        return metrics

    for fname in os.listdir(metrics_dir):
        if fname.endswith("_metrics.json"):
            fpath = os.path.join(metrics_dir, fname)
            with open(fpath, "r") as f:
                metrics[fname] = json.load(f)
    return metrics


def copy_missing_metrics(source_dir, dest_dir):
    """Copy any metric JSON files missing from dest (e.g. EEGConformer)."""
    src_metrics = os.path.join(source_dir, "metrics")
    dst_metrics = ensure_dir(os.path.join(dest_dir, "metrics"))

    copied = []
    for fname in os.listdir(src_metrics):
        if fname.endswith("_metrics.json"):
            dst_path = os.path.join(dst_metrics, fname)
            if not os.path.exists(dst_path):
                src_path = os.path.join(src_metrics, fname)
                shutil.copy2(src_path, dst_path)
                copied.append(fname)
                print(f"  Copied: {fname}")

    return copied


def copy_missing_per_model_plots(source_dir, dest_dir):
    """Copy per-model plots (training curves, confusion matrices) that are
    missing from dest. These are generated during training and can't be
    regenerated from JSON alone."""
    src_plots = os.path.join(source_dir, "plots")
    dst_plots = ensure_dir(os.path.join(dest_dir, "figures"))

    copied = []
    for fname in os.listdir(src_plots):
        # Per-model plots: training curves and confusion matrices
        if ("training_curves" in fname or "confusion_matrix" in fname):
            dst_path = os.path.join(dst_plots, fname)
            if not os.path.exists(dst_path):
                src_path = os.path.join(src_plots, fname)
                shutil.copy2(src_path, dst_path)
                copied.append(fname)
                print(f"  Copied: {fname}")

    return copied


def regenerate_comparison_plots(all_results, dest_dir):
    """Regenerate all comparison/aggregate plots using full model set."""
    plots_dir = ensure_dir(os.path.join(dest_dir, "figures"))
    viz = Visualizer(plots_dir, dpi=150, fmt="png")

    datasets = list(all_results.keys())
    all_models = set()
    for ds in all_results.values():
        all_models.update(ds.keys())

    print(f"\n  Datasets: {datasets}")
    print(f"  Models:   {sorted(all_models)}")

    generated = []

    # --- Per-dataset comparison plots ---
    for ds_name in datasets:
        ds_results = all_results[ds_name]

        # Model comparison bar chart (accuracy)
        path = viz.plot_model_comparison(ds_results, ds_name, metric="accuracy")
        generated.append(os.path.basename(path))
        print(f"  Generated: {os.path.basename(path)}")

        # Multi-metric chart
        path = viz.plot_multi_metric(ds_results, ds_name)
        generated.append(os.path.basename(path))
        print(f"  Generated: {os.path.basename(path)}")

    # --- Cross-dataset plots (need >= 2 datasets) ---
    if len(datasets) >= 2:
        # Cross-dataset grouped bar charts
        for metric in ['accuracy', 'f1']:
            path = viz.plot_dataset_comparison(all_results, metric=metric)
            generated.append(os.path.basename(path))
            print(f"  Generated: {os.path.basename(path)}")

        # Cross-dataset heatmaps
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            path = viz.plot_cross_dataset_heatmap(all_results, metric=metric)
            generated.append(os.path.basename(path))
            print(f"  Generated: {os.path.basename(path)}")

        # Summary tables (CSV + LaTeX)
        csv_path, tex_path = viz.save_summary_table(all_results)
        # Move tables to the tables directory
        tables_dir = ensure_dir(os.path.join(dest_dir, "tables"))
        for src in [csv_path, tex_path]:
            dst = os.path.join(tables_dir, os.path.basename(src))
            shutil.move(src, dst)
            print(f"  Generated: tables/{os.path.basename(src)}")

    return generated


def update_all_results_json(all_results, dest_dir):
    """Write the complete all_results.json with all models."""
    metrics_dir = ensure_dir(os.path.join(dest_dir, "metrics"))
    path = os.path.join(metrics_dir, "all_results.json")
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Updated: metrics/all_results.json")


def main():
    print("=" * 60)
    print("  PLOT REGENERATION - Including EEGConformer")
    print("=" * 60)

    # 1. Load the complete results (with EEGConformer)
    print("\n[1/5] Loading results from eeg_benchmark/...")
    all_results = load_all_results(SOURCE_DIR)

    # Verify EEGConformer is present
    for ds_name, models in all_results.items():
        model_names = list(models.keys())
        has_conformer = "EEGConformer" in model_names
        status = "OK" if has_conformer else "MISSING"
        print(f"  {ds_name}: {model_names} [{status}]")

    # 2. Copy missing per-model metrics
    print("\n[2/5] Copying missing metric files...")
    copied_metrics = copy_missing_metrics(SOURCE_DIR, DEST_DIR)
    if not copied_metrics:
        print("  All metric files already present.")

    # 3. Copy missing per-model plots (training curves, confusion matrices)
    print("\n[3/5] Copying missing per-model plots...")
    copied_plots = copy_missing_per_model_plots(SOURCE_DIR, DEST_DIR)
    if not copied_plots:
        print("  All per-model plots already present.")

    # 4. Regenerate all comparison plots with complete model set
    print("\n[4/5] Regenerating comparison plots...")
    generated = regenerate_comparison_plots(all_results, DEST_DIR)

    # 5. Update all_results.json
    print("\n[5/5] Updating all_results.json...")
    update_all_results_json(all_results, DEST_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("  REGENERATION COMPLETE")
    print("=" * 60)
    print(f"\n  Metrics copied:  {len(copied_metrics)}")
    print(f"  Plots copied:    {len(copied_plots)}")
    print(f"  Plots generated: {len(generated)}")
    print(f"\n  Output: {DEST_DIR}")

    # List final figures
    figures_dir = os.path.join(DEST_DIR, "figures")
    if os.path.exists(figures_dir):
        figs = sorted(os.listdir(figures_dir))
        print(f"\n  Final figures ({len(figs)}):")
        for f in figs:
            print(f"    - {f}")

    print()


if __name__ == "__main__":
    main()
