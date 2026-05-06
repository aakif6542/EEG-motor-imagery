# evaluate.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# -------------------------------
# PLOT TRAINING CURVES
# -------------------------------
def plot_history(history, title="Model"):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.grid()
    plt.show()


# -------------------------------
# CONFUSION MATRIX
# -------------------------------
def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(title)
    plt.show()


# -------------------------------
# COMPARE MODELS (BAR PLOT)
# -------------------------------
def compare_models(results_dict):
    names = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure()
    plt.bar(names, values)

    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

    plt.title("Model Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis='y')

    plt.show()


# -------------------------------
# SAVE RESULTS
# -------------------------------
def save_results(results_dict, save_path="results.txt"):
    with open(save_path, "w") as f:
        f.write("=== MODEL RESULTS ===\n")
        for k, v in results_dict.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Results saved to {save_path}")


# -------------------------------
# FULL EVALUATION PIPELINE
# -------------------------------
def evaluate_all(
    cnn_history=None,
    eegnet_history=None,
    y_test=None,
    y_pred_cnn=None,
    y_pred_eegnet=None,
    results=None
):
    """
    Central evaluation function
    """

    # Training curves
    if cnn_history:
        plot_history(cnn_history, "CNN")

    if eegnet_history:
        plot_history(eegnet_history, "EEGNet")

    # Confusion matrices
    if y_pred_cnn is not None:
        plot_confusion(y_test, y_pred_cnn, "CNN Confusion Matrix")

    if y_pred_eegnet is not None:
        plot_confusion(y_test, y_pred_eegnet, "EEGNet Confusion Matrix")

    # Model comparison
    if results:
        compare_models(results)
        save_results(results)

    plt.show(block=True)