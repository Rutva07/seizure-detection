import os
import numpy as np

# Helper function to load and label data
def load_and_label(path, label):
    files = sorted([f for f in os.listdir(path) if f.lower().endswith('.txt')])
    data = np.array([np.loadtxt(os.path.join(path, file)) for file in files])
    labels = np.full((data.shape[0], 1), label)
    return np.hstack((data, labels))

# Load all datasets with appropriate labels
f_data = load_and_label(r"C:\Python\python\seizure_detection\bonn_dataset\f\F", 0)
n_data = load_and_label(r"C:\Python\python\seizure_detection\bonn_dataset\n\N", 0)
o_data = load_and_label(r"C:\Python\python\seizure_detection\bonn_dataset\o\O", 0)
z_data = load_and_label(r"C:\Python\python\seizure_detection\bonn_dataset\z\Z", 0)
s_data = load_and_label(r"C:\Python\python\seizure_detection\bonn_dataset\s\S", 1)

# Combine all datasets into one
all_data = np.vstack([f_data, n_data, o_data, z_data, s_data])

print("Shape of full dataset with labels:", all_data.shape)
print("Sample row (last value is label):", all_data[0])




import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Split into features (X) and labels (y)
X = all_data[:, :-1]  # EEG data
y = all_data[:, -1]   # Labels

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Build the model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
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
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Split features and labels 
X_time = all_data[:, :-1]  # EEG time-domain data
y = all_data[:, -1]        # Labels

# Fourier Transform (absolute magnitude) 
X_freq = np.abs(np.fft.fft(X_time, axis=1))
X_freq = X_freq[:, :X_freq.shape[1] // 2]

print("Shape of frequency-domain data:", X_freq.shape)

#Scale features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_freq)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

#Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Predictions 
y_pred = model.predict(X_test)

# Evaluation
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



#Split features and labels
X_time = all_data[:, :-1]  # EEG time-domain data
y = all_data[:, -1]        # Labels

#Wavelet Transform function 
def compute_wavelet_features(data, wavelet='db4', level=4):
    features = []
    for signal in data:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        coeff_vector = np.hstack(coeffs)  # Flatten coefficients into one vector
        features.append(coeff_vector)
    return np.array(features)

#Apply wavelet transform 
X_wavelet = compute_wavelet_features(X_time)

print("Shape of wavelet-transformed data:", X_wavelet.shape)

#Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_wavelet)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

#Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Predictions 
y_pred = model.predict(X_test)

#Evaluation 
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix 
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


X_time = all_data[:, :-1]  
y = all_data[:, -1]       
X_freq = np.abs(np.fft.fft(X_time, axis=1))
X_freq = X_freq[:, :X_freq.shape[1] // 2]

print("Fourier transformed data shape:", X_freq.shape)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_freq)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

#Random Forest Classifier 
model = RandomForestClassifier(
    n_estimators=100,      
    max_depth=None,       
    random_state=42,
    n_jobs=-1            
)
model.fit(X_train, y_train)

#Predictions
y_pred = model.predict(X_test)
#Evaluation 
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
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

#Feature Importance
importances = model.feature_importances_

plt.figure(figsize=(12, 4))
plt.bar(range(len(importances)), importances)
plt.title('Feature Importance (Frequency Components)')
plt.xlabel('Frequency Index')
plt.ylabel('Importance')
plt.show()




X_raw = all_data[:, :-1]  
y = all_data[:, -1]    

print("Raw EEG data shape:", X_raw.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

#Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,      
    max_depth=None,       
    random_state=42,
    n_jobs=-1             
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

#Evaluation 
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy (Raw EEG): {accuracy:.4f}\n")

print("Classification Report (Raw EEG):")
print(classification_report(y_test, y_pred))

#Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix (Raw EEG)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0,1], ['Non-seizure', 'Seizure'])
plt.yticks([0,1], ['Non-seizure', 'Seizure'])

# Annotate counts
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.show()
importances = model.feature_importances_

plt.figure(figsize=(12, 4))
plt.bar(range(len(importances)), importances)
plt.title('Feature Importance (Raw EEG Timepoints)')
plt.xlabel('Time Index')
plt.ylabel('Importance')
plt.show()