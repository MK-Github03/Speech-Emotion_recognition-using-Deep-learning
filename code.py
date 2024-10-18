
# Import necessary libraries
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# === FEATURE EXTRACTION FUNCTION ===
def extract_feature(file_name, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True):
    """
    Extracts MFCC, chroma, mel spectrogram, spectral contrast, and tonnetz features from an audio file.
    
    Parameters:
    - file_name: Path to the audio file
    - mfcc: Boolean to determine whether to extract MFCC features
    - chroma: Boolean to determine whether to extract Chroma features
    - mel: Boolean to determine whether to extract Mel spectrogram features
    - contrast: Boolean to determine whether to extract Spectral Contrast features
    - tonnetz: Boolean to determine whether to extract Tonnetz features
    
    Returns:
    - result: A numpy array containing the extracted features
    """
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if chroma or contrast:
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

        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))

        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))

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
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True)
        x.append(feature)
        y.append(emotion)
    
    return train_test_split(np.array(x), y, test_size=test_size, shuffle=True, random_state=9)

# === BALANCE DATA FUNCTION ===
def balance_data(x_train, y_train):
    """
    Balances the dataset using SMOTE.
    
    Parameters:
    - x_train: Training feature set
    - y_train: Training labels
    
    Returns:
    - x_train_res, y_train_res: Balanced feature set and labels
    """
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train, y_train)
    return x_train_res, y_train_res

# === CNN MODEL FUNCTION ===
def build_cnn_model(input_shape):
    """
    Builds and compiles a Convolutional Neural Network (CNN) model.
    
    Parameters:
    - input_shape: Shape of the input data
    
    Returns:
    - model: Compiled CNN model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(observed_emotions), activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# === PREDICTION FUNCTION FOR SINGLE FILE ===
def extract_single_file(file_path):
    """
    Extract features from a single audio file.
    
    Parameters:
    - file_path: Path to the audio file
    
    Returns:
    - feature array for the classifier
    """
    feature = extract_feature(file_path, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True)
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

# === CONFUSION MATRIX FUNCTION ===
def plot_confusion_matrix(y_test, y_pred):
    """
    Plots the confusion matrix for model evaluation.
    
    Parameters:
    - y_test: True labels
    - y_pred: Predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# === MAIN SCRIPT ===
if __name__ == "__main__":
    # Path to dataset
    data_directory = 'user/path/to/dataset'  # Change this to the path of your dataset

    # Load data and split into training and testing sets
    x_train, x_test, y_train, y_test = load_data(data_directory, test_size=0.2)

    # Balance the training data
    x_train_res, y_train_res = balance_data(x_train, y_train)

    # Train an SVM classifier
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x_train_res, y_train_res)

    # Evaluate on the test set
    y_pred = classifier.predict(x_test)
    print(f"Test set accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Example usage for a single file prediction
    single_file_path = 'Actor_15/03-01-06-02-01-01-15.wav'  # Change to the correct path

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
