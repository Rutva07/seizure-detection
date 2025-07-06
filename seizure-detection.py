import mne
import numpy as np

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
from sklearn.utils import resample

# Split into features (X) and labels (y)
X = all_data[:, :-1]  # EEG data
y = all_data[:, -1]   # Labels

X_majority = X[y == 0]
y_majority = y[y == 0]
X_minority = X[y == 1]
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
X = X_balanced
y = y_balanced

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)


import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Build the model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],),
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")


import matplotlib.pyplot as plt

# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model_training_results.png', dpi=300)
plt.show()




import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. Split features and labels ---
X_time = all_data[:, :-1]  # EEG time-domain data
y = all_data[:, -1]        # Labels

# --- 2. Fourier Transform (absolute magnitude) ---
X_freq = np.abs(np.fft.fft(X_time, axis=1))

# (Optional) Keep only first half of frequencies due to symmetry
X_freq = X_freq[:, :X_freq.shape[1] // 2]

print("Shape of frequency-domain data:", X_freq.shape)

# --- 3. Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_freq)

# --- 4. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y)

# --- 5. Logistic Regression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 6. Predictions ---
y_pred = model.predict(X_test)

# --- 7. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
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



import numpy as np
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. Split features and labels ---
X_time = all_data[:, :-1]  # EEG time-domain data
y = all_data[:, -1]        # Labels

# --- 2. Wavelet Transform function ---
def compute_wavelet_features(data, wavelet='db4', level=4):
    features = []
    for signal in data:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        coeff_vector = np.hstack(coeffs)  # Flatten coefficients into one vector
        features.append(coeff_vector)
    return np.array(features)

# --- 3. Apply wavelet transform ---
X_wavelet = compute_wavelet_features(X_time)

print("Shape of wavelet-transformed data:", X_wavelet.shape)

# --- 4. Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_wavelet)

# --- 5. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y)

# --- 6. Logistic Regression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- 7. Predictions ---
y_pred = model.predict(X_test)

# --- 8. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- 9. Confusion Matrix ---
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


