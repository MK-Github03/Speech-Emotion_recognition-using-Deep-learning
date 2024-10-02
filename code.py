

# Import necessary libraries
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
import matplotlib.pyplot as plt

# === FEATURE EXTRACTION FUNCTION ===
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    """
    Extracts MFCC, chroma, and mel spectrogram features from an audio file.
    
    Parameters:
    - file_name: Path to the audio file
    - mfcc: Boolean to determine whether to extract MFCC features
    - chroma: Boolean to determine whether to extract Chroma features
    - mel: Boolean to determine whether to extract Mel spectrogram features
    
    Returns:
    - result: A numpy array containing the extracted features
    """
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if chroma:
            stft = np.abs(librosa.stft(X))

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

    return result

# === EMOTIONS MAPPING ===
emotions = {
  '01': 'neutral',
  '02': 'calm',
  '03': 'happy',
  '04': 'sad',
  '05': 'angry',
  '06': 'fearful',
  '07': 'disgust',
  '08': 'surprised'
}

# Observed emotions
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# === DATA LOADING FUNCTION ===
def load_data(data_directory, test_size=0.2):
    """
    Loads the dataset and extracts features for each audio file in the specified directory.
    
    Parameters:
    - data_directory: Path to the dataset directory
    - test_size: Proportion of the dataset to include in the test split
    
    Returns:
    - x_train, x_test, y_train, y_test: Training and test datasets
    """
    x, y = [], []
    
    for file in glob.glob(os.path.join(data_directory, "Actor_*", "*.wav")):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    
    return train_test_split(np.array(x), y, test_size=test_size, shuffle=True, random_state=9)

# === SINGLE FILE FEATURE EXTRACTION ===
def extract_single_file(file_path):
    """
    Extract features from a single audio file.
    
    Parameters:
    - file_path: Path to the audio file
    
    Returns:
    - feature array for the classifier
    """
    feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    return np.array([feature])

# === IMAGE DISPLAY FUNCTION ===
def display_emotion_image(prediction, image_paths):
    """
    Display the corresponding emotion image based on the predicted emotion.
    
    Parameters:
    - prediction: Predicted emotion label
    - image_paths: Dictionary of emotion to image paths
    """
    emotion = prediction[0]  # Get the first predicted emotion
    img = Image.open(image_paths[emotion])
    
    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.title(emotion.capitalize())  # Capitalize the title
    plt.show()

# === MAIN SCRIPT ===
if __name__ == "__main__":
    # Path to dataset
    data_directory = 'user/path/to/dataset' #make sure you select the file that you downloaded from the github

    # Load data and split into training and testing sets
    x_train, x_test, y_train, y_test = load_data(data_directory, test_size=0.2)

    # Train an SVM classifier
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x_train, y_train)

    # Evaluate on the test set
    y_pred = classifier.predict(x_test)
    print(f"Test set accuracy: {accuracy_score(y_test, y_pred)}")

    # Example usage for a single file prediction
    single_file_path = 'Actor_15/03-01-06-02-01-01-15.wav'  #select the wav audio file from the dataset that I have uploaded
    
    # Extract features for the selected file
    x_single = extract_single_file(single_file_path)

    # Image paths for each emotion (ensure correct paths are set)
    image_paths = {
        'calm': '/Desktop/Project/calm.jpg',
        'disgust': '/Project/disgust.jpg',
        'fearful': '/Desktop/Project/fearful.jpg',
        'happy': '/Desktop/Project/happy.jpg'
    }

    # Make prediction for the selected file
    y_single_pred = classifier.predict(x_single)
    print(f"The predicted emotion for the selected file is: {y_single_pred[0]}")

    # Display the corresponding emotion image
    display_emotion_image(y_single_pred, image_paths)
