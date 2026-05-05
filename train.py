# train.py

from data_loader import load_data
from preprocess import preprocess_pipeline, add_channel_dim

from models.csp_svm import CSPSVMModel
from models.cnn import CNNModel
from models.eegnet import EEGNet

from evaluate import evaluate_all


# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = r"E:\Downloads\BCICIV_2a_gdf"


def run_experiment():
    print("\n===== LOADING DATA =====")
    X_train, y_train, X_test, y_test = load_data(DATA_PATH)

    print("\n===== PREPROCESSING =====")
    X_train, X_test = preprocess_pipeline(X_train, X_test, apply_aug=True)

    # =========================================================
    # CSP + SVM
    # =========================================================
    print("\n===== TRAINING CSP + SVM =====")

    csp_model = CSPSVMModel(n_components=4)

    csp_model.fit(X_train, y_train)
    csp_acc, y_pred_csp = csp_model.evaluate(X_test, y_test)

    print(f"CSP + SVM Accuracy: {csp_acc:.4f}")

    # =========================================================
    # Prepare for Deep Learning Models
    # =========================================================
    X_train_dl = add_channel_dim(X_train)
    X_test_dl = add_channel_dim(X_test)

    # =========================================================
    # CNN
    # =========================================================
    print("\n===== TRAINING CNN =====")

    cnn = CNNModel(input_shape=X_train_dl.shape[1:])
    cnn.summary()

    cnn_history = cnn.fit(
        X_train_dl, y_train,
        X_test_dl, y_test
    )

    cnn_acc = cnn.evaluate(X_test_dl, y_test)
    y_pred_cnn = cnn.predict(X_test_dl)

    print(f"CNN Accuracy: {cnn_acc:.4f}")

    # =========================================================
    # EEGNet
    # =========================================================
    print("\n===== TRAINING EEGNet =====")

    eegnet = EEGNet(input_shape=X_train_dl.shape[1:])
    eegnet.summary()

    eegnet_history = eegnet.fit(
        X_train_dl, y_train,
        X_test_dl, y_test
    )

    eegnet_acc = eegnet.evaluate(X_test_dl, y_test)
    y_pred_eegnet = eegnet.predict(X_test_dl)

    print(f"EEGNet Accuracy: {eegnet_acc:.4f}")

    # =========================================================
    # FINAL RESULTS
    # =========================================================
    results = {
        "CSP": csp_acc,
        "CNN": cnn_acc,
        "EEGNet": eegnet_acc
    }

    print("\n===== FINAL RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # =========================================================
    # EVALUATION (PLOTS + ANALYSIS)
    # =========================================================
    print("\n===== GENERATING EVALUATION =====")

    evaluate_all(
        cnn_history=cnn_history,
        eegnet_history=eegnet_history,
        y_test=y_test,
        y_pred_cnn=y_pred_cnn,
        y_pred_eegnet=y_pred_eegnet,
        results=results
    )

    return results


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    run_experiment()