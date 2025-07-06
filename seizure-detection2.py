import mne
import numpy as np
from sklearn.utils import resample


edf_files = [r"C:\Python\python\seizure_detection\CHB-MIT\chb01_03.edf",
             r"C:\Python\python\seizure_detection\CHB-MIT\chb01_04.edf",
             r"C:\Python\python\seizure_detection\CHB-MIT\chb01_15.edf",
             r"C:\Python\python\seizure_detection\CHB-MIT\chb01_16.edf",
             r"C:\Python\python\seizure_detection\CHB-MIT\chb01_18.edf",
             r"C:\Python\python\seizure_detection\CHB-MIT\chb01_21.edf",
             r"C:\Python\python\seizure_detection\CHB-MIT\chb01_26.edf"]

seizure_intervals_list = [[(2996, 3036)], 
                          [(1467, 1494)],
                          [(1732, 1772)], 
                          [(1015, 1066)], 
                          [(1720, 1810)], 
                          [(327, 420)],
                          [(1862, 1963)]
                          ]



def segment_signal_custom_logic(data, fs, seizure_intervals, window_size=5000):
    """
    Segments multi-channel EEG into independent rows with your specified logic.

    Parameters:
        data (np.array): shape (channels, timepoints)
        fs (int): sampling frequency
        seizure_intervals (list of tuples): [(start_sec, end_sec), ...]
        window_size (int): number of samples per window

    Returns:
        data_with_labels (np.array): (num_windows, window_size + 1)
    """
    n_channels, n_samples = data.shape
    segments = []
    labels = []

    for ch in range(n_channels):
        signal = data[ch]
        num_windows = n_samples // window_size  # ignore incomplete last window

        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size

            # Time of this segment (in seconds)
            segment_start_time = start_idx / fs
            segment_end_time = end_idx / fs

            # Check if no seizure intervals
            if not seizure_intervals:
                label = 0
                segments.append(signal[start_idx:end_idx])
                labels.append(label)
            else:
                # Check if segment is fully inside any seizure interval
                added = False
                for seizure_start, seizure_end in seizure_intervals:
                    if (segment_start_time >= seizure_start) and (segment_end_time <= seizure_end):
                        # fully inside → label=1
                        segments.append(signal[start_idx:end_idx])
                        labels.append(1)
                        added = True
                        break
                    elif (segment_end_time >= seizure_start and segment_start_time < seizure_start) or \
                         (segment_start_time <= seizure_end and segment_end_time > seizure_end) or \
                         (segment_start_time < seizure_start and segment_end_time > seizure_end):
                        # overlaps but not fully inside → skip
                        added = True
                        break
                if not added:
                    # fully outside all seizure intervals → label=0
                    segments.append(signal[start_idx:end_idx])
                    labels.append(0)

    X = np.array(segments)
    y = np.array(labels)

    data_with_labels = np.hstack((X, y[:, None]))
    return data_with_labels


all_data = []

for edf_file, seizure_intervals in zip(edf_files, seizure_intervals_list):
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    raw.filter(1., 40.)
    raw.notch_filter(60.)

    data, times = raw[:]
    fs = int(raw.info['sfreq'])

    dataset = segment_signal_custom_logic(data, fs, seizure_intervals, window_size=5000)
    all_data.append(dataset)



final_dataset = np.vstack(all_data)

print("Final combined dataset shape:", final_dataset.shape)



all_data = final_dataset



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. Split features and labels ---
X_time = all_data[:, :-1]  # EEG time-domain data
y = all_data[:, -1]        # Labels

X_majority = X_time[y == 0]
y_majority = y[y == 0]
X_minority = X_time[y == 1]
y_minority = y[y == 1]

# Desired number of majority samples (5x minority samples)
n_samples_majority = len(y_minority) * 5

# Downsample majority class
X_majority_downsampled, y_majority_downsampled = resample(
    X_majority, y_majority,
    replace=False,
    n_samples=n_samples_majority,
    random_state=42
)

# Combine minority class with downsampled majority class
X_balanced = np.vstack((X_minority, X_majority_downsampled))
y_balanced = np.hstack((y_minority, y_majority_downsampled))
X_time = X_balanced
y = y_balanced

# --- 2. Apply Fourier Transform (magnitude only) ---
X_freq = np.abs(np.fft.fft(X_time, axis=1))

# Keep only first half (due to symmetry)
X_freq = X_freq[:, :X_freq.shape[1] // 2]

print("Fourier transformed data shape:", X_freq.shape)

# --- 3. (Optional) Scale features (RF doesn't require scaling, but harmless) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_freq)

# --- 4. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# --- 5. Random Forest Classifier ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=50,
    max_features='log2',  # fewer features evaluated per split
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- 6. Predictions ---
y_pred = model.predict(X_test)

# --- 7. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- 8. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0,1], ['Non-seizure', 'Seizure'])
plt.yticks([0,1], ['Non-seizure', 'Seizure'])

# Annotate counts
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.show()

# --- 9. (Optional) Feature Importance ---
importances = model.feature_importances_

plt.figure(figsize=(12, 4))
plt.bar(range(len(importances)), importances)
plt.title('Feature Importance (Frequency Components)')
plt.xlabel('Frequency Index')
plt.ylabel('Importance')
plt.show()




import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.sqrt(np.mean(signal ** 2)),  # RMS
        kurtosis(signal),
        skew(signal)
    ]

# --- 1. Split raw signals and labels ---
X_raw = all_data[:, :-1]  # shape: (num_rows, num_samples_per_window)
y = all_data[:, -1]

print("Raw EEG data shape:", X_raw.shape)

# --- 2. Extract features for each row ---
features_list = []
for row in X_raw:
    features = extract_features(row)
    features_list.append(features)

X_features = np.array(features_list)


X_majority = X_features[y == 0]
y_majority = y[y == 0]
X_minority = X_features[y == 1]
y_minority = y[y == 1]

# Desired non-seizure samples (5:1 ratio)
n_samples_majority = len(y_minority) * 5

# Downsample majority to 5x minority size
X_majority_downsampled, y_majority_downsampled = resample(
    X_majority, y_majority,
    replace=False,
    n_samples=n_samples_majority,
    random_state=42
)

# Combine balanced dataset (minority + downsampled majority)
X_balanced = np.vstack((X_minority, X_majority_downsampled))
y_balanced = np.hstack((y_minority, y_majority_downsampled))
y = y_balanced

print(f"Balanced dataset shape: {X_balanced.shape}")
print(f"Class counts → non-seizure: {np.sum(y_balanced==0)}, seizure: {np.sum(y_balanced==1)}")

# --- 3. Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)


# --- 4. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# --- 5. Random Forest Classifier ---
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=40,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- 6. Predictions ---
y_pred = model.predict(X_test)

# --- 7. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy (Feature-based): {accuracy:.4f}\n")

print("Classification Report (Feature-based):")
print(classification_report(y_test, y_pred))

# --- 8. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix (Feature-based)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0,1], ['Non-seizure', 'Seizure'])
plt.yticks([0,1], ['Non-seizure', 'Seizure'])

# Annotate counts
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.show()

# --- 8. (Optional) Feature Importance ---
importances = model.feature_importances_

plt.figure(figsize=(12, 4))
plt.bar(range(len(importances)), importances)
plt.title('Feature Importance (Raw EEG Timepoints)')
plt.xlabel('Time Index')
plt.ylabel('Importance')
plt.show()