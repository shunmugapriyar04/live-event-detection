
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 1. Load and preprocess data
def load_data(csv_path, audio_folder, sample_rate=22050, n_mfcc=40, max_pad_len=44):
    df = pd.read_csv(csv_path)
    features = []
    labels = []
    
    for index, row in df.iterrows():
        try:
            file_path = os.path.join(audio_folder, row["slice_file_name"])
            label = row["class"]
            y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            
            # Pad or truncate
            if mfccs.shape[1] < max_pad_len:
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]
            
            features.append(mfccs)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {row['slice_file_name']}: {e}")
    
    return np.array(features), np.array(labels)

# Set paths
csv_path = '/Users/SHUNMU/Downloads/mini1/UrbanSound8k.csv'
audio_folder = '/Users/SHUNMU/Downloads/mini1/audio'

# Load data
X, y = load_data(csv_path, audio_folder)

# Reshape for CNN input
X = X[..., np.newaxis]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 2. Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(40, 44, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 4. Save model to .h5 file
model.save('2dd_model.h5')

print("Model saved as 'environmental_sound_model.h5'")
