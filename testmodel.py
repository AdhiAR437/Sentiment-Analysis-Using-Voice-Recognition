import librosa
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('my_model.keras')

# Function to extract MFCC features from a single audio file
def extract_mfcc_single(audio_file_path, max_frames=300):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Pad or truncate mfccs to have max_frames
    if mfccs.shape[1] < max_frames:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_frames - mfccs.shape[1])), mode='constant', constant_values=0)
    else:
        mfccs = mfccs[:, :max_frames]
    return mfccs

# Path to the single .wav file you want to predict
wav_file_path = '03-01-01-01-01-01-01.wav'

# Extract MFCC features from the single audio file
single_sample = extract_mfcc_single(wav_file_path)

# Reshape the single sample to match the model input shape
single_sample = np.reshape(single_sample, (1, 40, 300))  # Assuming max_frames=300

# Predict the emotion for the single sample
y_pred = model.predict(single_sample)
print(y_pred)
# Convert predicted probabilities to class label
predicted_emotion = np.argmax(y_pred)
# Print the predicted emotion
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
print("Predicted Emotion:", emotion_labels[predicted_emotion])
