def get_emotion_from_filename(filename):
    # Split the file name using the '-' separator
    parts = filename.split("-")

    # Extract the emotion code (third element)
    emotion_code = int(parts[2])

    # Define the mapping from emotion code to emotion label
    emotion_mapping = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }

    # Return the emotion label corresponding to the emotion code
    return emotion_mapping.get(emotion_code, 'unknown')


# Example usage
filename = '03-01-07-02-02-02-10.wav'
emotion = get_emotion_from_filename(filename)
print(emotion)  # Output: 'fearful'
