# Seizure Detection

### By: Rutva Patel

---

## 📌 Introduction

This project aims to improve the safety and quality of life for individuals with epilepsy by developing an accurate, real-time seizure detection system using EEG data and machine learning. Seizures are unpredictable and potentially dangerous; a timely detection system allows for earlier interventions such as medication alerts or caregiver response. We explore multiple ML models to classify EEG data into seizure and non-seizure categories.

This system combines classical machine learning (Logistic Regression, Random Forest) and deep learning (Feedforward Neural Network) techniques. It includes sophisticated preprocessing such as filtering, class balancing, and feature extraction from time, frequency, and time-frequency domains.

---

## 📊 Datasets

Three EEG datasets were used:

* **Bonn Dataset**: Prefiltered EEG segments.
* **CHB-MIT Dataset**: Raw continuous recordings with seizure annotations.
* **Zenodo Dataset**: Raw EEG data with annotated seizure events.

**Sources:**

* Bonn: [Data](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/)
* CHB-MIT: [Data](https://physionet.org/content/chbmit/1.0.0/)
* Zenodo: [Data](https://zenodo.org/records/2547147#.Y7eU5uxBwlI)

---

## ⚖️ Seizure Detection vs Prediction

* **Detection**: Classifying seizure onset in real-time (ictal vs non-ictal).
* **Prediction**: Forecasting seizures before they happen using preictal windows.

The Bonn dataset is suited for **detection** only, while CHB-MIT and Zenodo support **both detection and prediction**, depending on how data is segmented and labeled.

---

## 🧠 EEG Signal Preprocessing

**Tools Used**: `MNE`, `NumPy`, `PyWavelets`, `SciPy`

### Filtering Steps:

1. **Bandpass filter**: 0.5–40 Hz
2. **Notch filter**: 50/60 Hz (for powerline interference)

These steps clean raw EEG data and improve signal-to-noise ratio.

---

## 🤖 Models Implemented

### 1. Feedforward Neural Network (FNN)

* Input: Engineered features
* Layers: Dense (64, relu) → Dropout(0.3) → Dense (32, relu) → Dropout(0.3) → Sigmoid Output
* Optimizer: Adam
* Loss: Binary Crossentropy
* Regularization: L2 (for unfiltered datasets)

### 2. Logistic Regression

* a) **Fourier Transform**: Frequency-domain features (z-score normalized)
* b) **Wavelet Transform**: DWT (db4), 4-level decomposition

### 3. Random Forest

* a) **FFT Features**
* b) **Raw EEG Time-Series**
* c) **Statistical Features**: mean, std, min, max, RMS, kurtosis, skewness

For large datasets, downsampling is used to maintain a 5:1 non-seizure\:seizure ratio.

---

## 📈 Results Summary

### ✅ Prefiltered Dataset (Bonn)

| Model                   | Accuracy |
| ----------------------- | -------- |
| FNN                     | 0.9800   |
| Logistic (FFT)          | 0.9900   |
| Logistic (Wavelet)      | 0.9400   |
| Random Forest (FFT)     | 0.9900   |
| Random Forest (Raw EEG) | 0.9800   |

### ✅ Raw Data (CHB-MIT, Zenodo)

| Model                                | Accuracy |
| ------------------------------------ | -------- |
| FNN (with L2 regularization)         | 0.9493   |
| Logistic Regression (FFT)            | 0.9982   |
| Logistic Regression (Wavelet)        | 0.9907   |
| Random Forest (FFT)                  | 0.9638   |
| Random Forest (Statistical Features) | 0.9734   |

**Note**: The logistic regression (wavelet) model often overfit to the majority class (non-seizure) despite high accuracy. Other models showed more balanced performance.

---

## 🔍 Average Model Performance

Across all models and datasets:

* **Average Accuracy**: \~96.5%
* **Average Precision**: \~94.2%

---

## ✅ Conclusion

This project demonstrated the power of combining signal processing with machine learning to build a robust seizure detection system. Logistic Regression using Fourier-transformed features achieved the best performance across datasets. However, all models benefited significantly from good preprocessing.

The pipeline is extendable to seizure **prediction** by re-segmenting CHB-MIT and Zenodo data to extract preictal windows.

Future enhancements could involve:

* Using more advanced neural networks like CNNs or LSTMs
* Real-time streaming model integration
* Improving preictal feature extraction for prediction

---

## 🧪 Getting Started

This project runs as a single Python script that performs all steps: preprocessing, training, and evaluation.

### 1️⃣ Install Dependencies

Install the required Python packages using pip:

```bash
pip install mne numpy scikit-learn matplotlib tensorflow pywavelets
#or 
pip install -r requirements.txt
```

User can install any addtional dataset they want, make sure to add the dataset path to the edf_files list and interval time of seizure to seizure_intervals_list.

seizure-detection.py and seizure-detection2.py were used for the analysis and to make the report. 

---

*This project is a part of my independent machine learning exploration on biomedical signals, combining my interests in AI and healthcare technology.*
