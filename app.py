from flask import Flask, render_template, request, redirect, url_for
import os
import librosa
import numpy as np
from keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Define a route for the home page
@app.route('/')
def index():
    return render_template('test_voice.html')

@app.route('/demo_voice')
def demo_voice():
    return render_template('demo_voice.html')
@app.route('/test_voice')
def test_voice():
    return render_template('test_voice.html')

# Define a route to handle the form submission and save data to the database
@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
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
        wav_file_path = file_path

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
        # return emotion_labels[predicted_emotion]
        return emotion_labels[predicted_emotion]
    return 'No file selected.'

if __name__ == '__main__':
    app.run(debug=True)