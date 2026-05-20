# experiments/runner.py
# ============================================================
# Experiment runner — orchestrates the full benchmarking
# pipeline: dataset loading → preprocessing → training →
# evaluation → visualization for all (dataset × model)
# combinations.
# ============================================================

import os
import time
import numpy as np
from typing import Dict, Any, Optional

from configs.experiment_config import ExperimentConfig
from datasets import get_dataset
from datasets.base_dataset import EEGDataBundle
from models import get_model, MODEL_REGISTRY
from preprocessing.pipeline import preprocess_pipeline, add_channel_dim
from evaluation.metrics import compute_metrics, format_metrics_table
from evaluation.visualization import Visualizer
from utils.io_utils import (
    setup_results_directory, save_json, log_experiment_start
)


from utils.logging_config import get_logger

class ExperimentRunner:
    """
    Orchestrates multi-dataset, multi-model benchmarking.

    Usage:
        config = ExperimentConfig()
        runner = ExperimentRunner(config)
        results = runner.run()
    """

    def __init__(self, config: ExperimentConfig,
                 project_dir: Optional[str] = None):
        self.config = config
        self.logger = get_logger()

        if project_dir is None:
            project_dir = os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))

        self.project_dir = project_dir
        self.results_paths = setup_results_directory(
            os.path.join(project_dir, config.results_dir),
            config.experiment_name
        )
        self.viz = Visualizer(
            self.results_paths["plots"],
            dpi=config.evaluation.figure_dpi,
            fmt=config.evaluation.figure_format
        )

        # Storage for all results
        # Structure: {dataset: {model: metrics_dict}}
        self.all_results: Dict[str, Dict[str, Any]] = {}
        self.all_histories: Dict[str, Dict[str, Any]] = {}
        self.all_predictions: Dict[str, Dict[str, np.ndarray]] = {}

        # Set random seed
        np.random.seed(config.random_seed)

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the full benchmarking pipeline.

        Returns
        -------
        dict : {dataset_name: {model_name: metrics_dict}}
        """
        self.logger.info("=" * 60)
        self.logger.info("  EEG BENCHMARKING FRAMEWORK")
        self.logger.info("=" * 60)

        # Log experiment metadata
        log_experiment_start(self.results_paths, self.config)

        total_start = time.time()

        # Iterate over all datasets
        for ds_name in self.config.dataset.datasets:
            self._run_dataset(ds_name)

        total_time = time.time() - total_start

        # Generate cross-dataset comparisons
        self.logger.info("=" * 60)
        self.logger.info("  GENERATING CROSS-DATASET ANALYSIS")
        self.logger.info("=" * 60)
        self._generate_cross_dataset_analysis()

        # Print final summary
        self._print_summary(total_time)

        return self.all_results

    def _run_dataset(self, dataset_name: str):
        """Run all models on a single dataset."""

        self.logger.info(f"{'='*60}")
        self.logger.info(f"  DATASET: {dataset_name}")
        self.logger.info(f"{'='*60}")

        # --- Load dataset ---
        ds_loader = get_dataset(dataset_name)
        data = ds_loader.load(
            tmin=self.config.dataset.tmin,
            tmax=self.config.dataset.tmax,
            bandpass_low=self.config.preprocess.bandpass_low,
            bandpass_high=self.config.preprocess.bandpass_high,
            resample_freq=self.config.dataset.target_sfreq,
        )

        if self.config.verbose:
            # logger info handles line breaks differently, let's print summary line by line
            for line in data.summary().split('\n'):
                self.logger.info(line)

        # --- Preprocess ---
        self.logger.info(f"Preprocessing {dataset_name}...")
        X_train, X_test = preprocess_pipeline(
            data.X_train, data.X_test,
            config=self.config.preprocess
        )

        # Prepare DL version (with channel dim)
        X_train_dl = add_channel_dim(X_train)
        X_test_dl = add_channel_dim(X_test)

        y_train = data.y_train
        y_test = data.y_test

        self.all_results[dataset_name] = {}
        self.all_histories[dataset_name] = {}
        self.all_predictions[dataset_name] = {}

        # --- Train each model ---
        for model_name in self.config.model.models:
            self._run_model(
                model_name, dataset_name,
                X_train, X_test,
                X_train_dl, X_test_dl,
                y_train, y_test,
                data
            )

        # --- Per-dataset visualizations ---
        self.logger.info(f"Generating plots for {dataset_name}...")

        # Model comparison bar chart
        self.viz.plot_model_comparison(
            self.all_results[dataset_name], dataset_name)

        # Multi-metric chart
        self.viz.plot_multi_metric(
            self.all_results[dataset_name], dataset_name)

    def _run_model(self, model_name, dataset_name,
                   X_train, X_test, X_train_dl, X_test_dl,
                   y_train, y_test, data):
        """Train and evaluate a single model on a single dataset."""

        self.logger.info(f"--- Training {model_name} on {dataset_name} ---")

        start_train = time.time()

        # Build model with appropriate parameters
        model = self._build_model(model_name, data)

        # Select correct input format
        if model.needs_channel_dim():
            Xtr, Xte = X_train_dl, X_test_dl
        else:
            Xtr, Xte = X_train, X_test

        # Train
        history = model.fit(
            Xtr, y_train,
            X_val=Xte, y_val=y_test,
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
            verbose=0 # Suppress keras progress bar
        )

        train_time = time.time() - start_train

        # Predict
        start_inf = time.time()
        y_pred = model.predict(Xte)
        inf_time = time.time() - start_inf

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, data.class_names)
        metrics['train_time_seconds'] = train_time
        metrics['inference_time_seconds'] = inf_time

        # Store results
        self.all_results[dataset_name][model_name] = metrics
        self.all_histories[dataset_name][model_name] = history
        self.all_predictions[dataset_name][model_name] = y_pred

        # Print summary
        self.logger.info(f"{model_name} on {dataset_name}: "
              f"Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1']:.4f} "
              f"(Train: {train_time:.1f}s, Inf: {inf_time:.3f}s)")

        # Save model
        model_dir = os.path.join(
            self.results_paths["models"],
            f"{dataset_name}_{model_name}"
        )
        model.save(model_dir)

        # Save metrics JSON
        metrics_path = os.path.join(
            self.results_paths["metrics"],
            f"{dataset_name}_{model_name}_metrics.json"
        )
        save_json(metrics, metrics_path)

        # Training curves (DL models only)
        if history is not None:
            self.viz.plot_training_curves(
                history, model_name, dataset_name)

        # Confusion matrix
        self.viz.plot_confusion_matrix(
            y_test, y_pred, model_name, dataset_name,
            class_names=data.class_names)

    def _build_model(self, model_name, data: EEGDataBundle):
        """Instantiate a model with dataset-appropriate parameters."""

        cfg = self.config.model

        if model_name == "CSP_SVM":
            return get_model(
                model_name,
                n_components=cfg.csp_n_components,
                kernel=cfg.svm_kernel,
            )

        elif model_name == "CNN":
            return get_model(
                model_name,
                n_channels=data.n_channels,
                n_timepoints=data.n_timepoints,
                dropout_rate=cfg.cnn_dropout,
                learning_rate=cfg.learning_rate,
            )

        elif model_name == "EEGNet":
            return get_model(
                model_name,
                n_channels=data.n_channels,
                n_timepoints=data.n_timepoints,
                dropout_rate=cfg.eegnet_dropout,
                kernel_length=cfg.eegnet_kernel_length,
                F1=cfg.eegnet_F1,
                D=cfg.eegnet_D,
                F2=cfg.eegnet_F2,
                learning_rate=cfg.learning_rate,
            )

        elif model_name == "EEGConformer":
            return get_model(
                model_name,
                n_channels=data.n_channels,
                n_timepoints=data.n_timepoints,
                d_model=cfg.conformer_d_model,
                num_heads=cfg.conformer_num_heads,
                num_transformer_blocks=cfg.conformer_num_blocks,
                ff_dim=cfg.conformer_ff_dim,
                conv_F1=cfg.conformer_conv_F1,
                conv_F2=cfg.conformer_conv_F2,
                conv_kernel1=cfg.conformer_conv_kernel1,
                conv_kernel2=cfg.conformer_conv_kernel2,
                dropout_rate=cfg.conformer_dropout,
                learning_rate=cfg.conformer_learning_rate,
            )

        else:
            # Future models — attempt generic construction
            return get_model(
                model_name,
                n_channels=data.n_channels,
                n_timepoints=data.n_timepoints,
            )

    def _generate_cross_dataset_analysis(self):
        """Generate cross-dataset comparison visualizations."""

        if len(self.all_results) < 2:
            self.logger.info("Skipping cross-dataset analysis (only 1 dataset)")
            return

        # Cross-dataset bar chart
        for metric in ['accuracy', 'f1']:
            self.viz.plot_dataset_comparison(
                self.all_results, metric=metric)

        # Cross-dataset heatmap
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            self.viz.plot_cross_dataset_heatmap(
                self.all_results, metric=metric)

        # Summary tables (CSV + LaTeX)
        self.viz.save_summary_table(self.all_results)

        # Save all results JSON
        save_json(
            self.all_results,
            os.path.join(self.results_paths["root"],
                         "all_results.json")
        )

    def _print_summary(self, total_time):
        """Print final results summary to console."""
        self.logger.info("=" * 60)
        self.logger.info("  BENCHMARK RESULTS SUMMARY")
        self.logger.info("=" * 60)

        table = format_metrics_table(self.all_results)
        # print line by line
        for line in table.split('\n'):
            self.logger.info(line)

        self.logger.info(f"Total time: {total_time:.1f} seconds")
        self.logger.info(f"Results saved to: {self.results_paths['root']}")
        self.logger.info("=" * 60)
