# 🧠 EEG Motor Imagery Classification: Cross-Subject Generalization Study

## 📌 Overview

This project investigates **cross-subject generalization in EEG-based motor imagery classification**, a key challenge in Brain-Computer Interface (BCI) systems.

We conduct a **controlled experimental comparison** between:

* **CSP + SVM** (traditional method)
* **CNN** (generic deep learning)
* **EEGNet** (EEG-specific deep learning model)

The goal is to understand:

> *Do deep learning models generalize better than traditional approaches in EEG classification?*

---

## 📊 Dataset

* **Dataset:** BCI Competition IV 2a
* **Subjects:** 9
* **Task:** Motor imagery (Left vs Right hand)
* **Evaluation:** **Cross-subject split**

  * Train: Subjects 1–6
  * Test: Subjects 7–9

---

## ⚙️ Pipeline

```
Raw EEG → Preprocessing → Model → Evaluation
```

### Preprocessing:

* Bandpass filtering: **8–30 Hz (Mu & Beta bands)**
* Normalization (based on training data)
* Optional data augmentation:

  * Gaussian noise
  * Time shifting
  * Amplitude scaling

---

## 🧠 Models Compared

### 1. CSP + SVM

* Handcrafted spatial filtering using Common Spatial Patterns
* Features classified using Support Vector Machine

---

### 2. CNN (Generic)

* Standard convolutional neural network
* Learns features automatically
* No EEG-specific design

---

### 3. EEGNet

* Lightweight CNN designed specifically for EEG
* Learns:

  * Temporal patterns (frequency)
  * Spatial patterns (channel relationships)

---

## 🧪 Experiments Conducted

### 🔹 Experiment 1: Baseline (0–4 sec window)

| Model     | Accuracy |
| --------- | -------- |
| CSP + SVM | ~0.69    |
| CNN       | ~0.49    |
| EEGNet    | ~0.78    |

---

### 🔹 Experiment 2: Data Augmentation

**Change introduced:**

```python
apply_aug = True
```

**Effect:**

* Increased robustness to noise and variability

| Model  | Result                  |
| ------ | ----------------------- |
| CNN    | ❌ No improvement        |
| EEGNet | ✅ Improved (~0.85 peak) |
| CSP    | ➖ No change             |

---

### 🔹 Experiment 3: Time Window Cropping (0.5–3.5 sec)

**Change introduced:**

```python
tmin = 0.5
tmax = 3.5
```

**Effect:**

* Removed noisy transition periods
* Focused on stable motor imagery signals

| Model     | Accuracy |
| --------- | -------- |
| CSP + SVM | ~0.77 🔥 |
| CNN       | ~0.49 ❌  |
| EEGNet    | ~0.76    |

---

## 📈 Key Findings

### 🔥 1. CNN fails in cross-subject setting

* Severe overfitting observed
* Performance remains near random (~50%)
* Data augmentation does not help

---

### 🔥 2. EEGNet shows strong generalization

* Consistent performance across experiments
* Benefits from augmentation
* Learns meaningful EEG-specific patterns

---

### 🔥 3. CSP remains competitive

* Significant improvement with proper preprocessing
* Sensitive to signal quality and time window selection

---

### 🔥 4. Preprocessing matters

* Bandpass filtering and time-window selection significantly affect performance
* Proper signal conditioning can boost traditional methods

---

## 🧠 Core Insight

> **Generic CNN models fail to generalize in cross-subject EEG classification, while EEGNet demonstrates robust performance due to its domain-specific architecture.**

---

## 📂 Project Structure

```
EEGNet-Project/
│
├── data_loader.py        # Loads EEG data, applies filtering, creates epochs
├── preprocess.py         # Normalization, augmentation, preprocessing pipeline
├── train.py              # Runs all experiments
├── evaluate.py           # Visualization and comparison
│
├── models/
│   ├── csp_svm.py        # CSP + SVM implementation
│   ├── cnn.py            # Generic CNN baseline
│   ├── eegnet.py         # EEGNet implementation
│
└── README.md
```

---

## ▶️ How to Run

```bash
python train.py
```

---

## 📌 Future Work

* Evaluate on additional EEG datasets (e.g., PhysioNet, DEAP)
* Improve CNN architecture for better generalization
* Explore transformer-based EEG models
* Investigate domain adaptation techniques

---

## ⭐ Final Note

This project focuses on **understanding model behavior**, not just maximizing accuracy.

It highlights the importance of:

* domain-specific architectures
* proper preprocessing
* and rigorous experimental design

---

💥 *Designed as a research-oriented study for BCI and EEG decoding.*
