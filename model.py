import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical

# Function to extract MFCC features
def extract_mfcc(audio_file_path, max_frames=300):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Pad or truncate mfccs to have max_frames
    if mfccs.shape[1] < max_frames:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - mfccs.shape[1])), mode='constant', constant_values=0)
    else:
        mfccs = mfccs[:, :max_frames]
    return mfccs

# Path to the RAVDESS dataset directory
dataset_dir = "E:\python\VoiceSentimentAnalysis\\archive"

speech_data_MFCC = []  # stores the MFCC data
speech_labels = []  # stores the labels

for subdirs, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".wav"):
            emotion = int(file.split("-")[2]) - 1  # Extract emotion from filename
            audio_file_path = os.path.join(subdirs, file)
            mfccs = extract_mfcc(audio_file_path)
            speech_data_MFCC.append(mfccs)
            speech_labels.append(emotion)

# Convert lists to numpy arrays
max_frames = 300  # Maximum number of frames
speech_data_array = np.asarray(speech_data_MFCC)
speech_data_array = np.reshape(speech_data_array, (speech_data_array.shape[0], 40, max_frames))
speech_labels_array = np.asarray(speech_labels)

# Normalize MFCC features
mean = np.mean(speech_data_array, axis=0)
std = np.std(speech_data_array, axis=0)
speech_data_array = (speech_data_array - mean) / std

# One-hot encode the labels
labels_cat = to_categorical(speech_labels_array)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(speech_data_array, labels_cat, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(92, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(92))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

# Compile the model
optimizer = RMSprop(learning_rate=0.001) # Specify learning rate explicitly
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=32)

# Save the model
model.save('my_model1.keras')