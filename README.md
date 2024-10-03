# Speech Emotion Detection using Deep Learning

Ps: this is an open source project, so feel free to replicate it!

This project implements a Speech Emotion Detection (SED) system using deep learning techniques alongside feature extraction methods such as MFCC, Chroma, and Mel Spectrograms. The model is trained to classify emotions from speech signals, utilizing tools like SVM and deep learning models for accurate emotion recognition. The system can also display an emotion-specific image based on the predicted emotion.


## Introduction

This project focuses on recognizing emotions from speech using deep learning models, specifically leveraging **Recurrent Neural Networks (RNN)** to learn the temporal patterns in speech. By combining feature extraction from the LibROSA library (e.g., MFCC, Chroma) with a deep learning classifier, the model aims to classify human emotions from audio recordings effectively.

## Requirements

To run this project, you need to have the following libraries installed:

- `librosa`
- `soundfile`
- `numpy`
- `scikit-learn`
- `PIL` (Python Imaging Library)
- `matplotlib`
- `tensorflow` (or your deep learning framework, if you're using RNN)

Install the necessary dependencies using the following command:

```
pip install librosa soundfile numpy scikit-learn Pillow matplotlib tensorflow
```

## Installation

1. Clone the repository to your local machine:

```
git clone https://github.com/MK-Github03/Speech-Emotion_recognition-using-Deep-learning
```

2. Navigate to the project directory:

```
cd https://github.com/MK-Github03/Speech-Emotion_recognition-using-Deep-learning
```

3. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Place the audio files (in `.wav` format) inside a directory following this structure:

```
Project/
│
├── Actor_01/
│   └── 03-01-06-02-01-01-01.wav
├── Actor_02/
│   └── 03-01-04-01-01-01-02.wav
│
└── ... (more audio files)
```

Ensure that the dataset follows this structure where each actor's files are placed inside their respective folders.

## Running the Model

To train and test the model, modify the `data_directory` variable in the script to point to your dataset location:

```
data_directory = '/path/to/your/Project'
```

Run the script:

```
python emotion_recognition.py
```

## Feature Extraction

The model uses the following features from each audio file:
- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Chroma Feature**
- **Mel Spectrogram**

These features are extracted using the `librosa` library and concatenated into a feature vector for classification.

(All the audio files will be uploaded in github or uploaded as a link)

## Emotion Labels

The emotion labels are extracted from the filenames following the naming convention `03-01-XX-XX-XX-XX-XX.wav`. The third field of the filename corresponds to the emotion, which is mapped as follows:
- `01` = Neutral
- `02` = Calm
- `03` = Happy
- `04` = Sad
- `05` = Angry
- `06` = Fearful
- `07` = Disgust
- `08` = Surprised

## Training and Testing

The dataset is split into training and testing sets using an 80/20 split. The model, which uses a **Recurrent Neural Network (RNN)** for sequence learning, is trained on the training set and evaluated on the test set. Additionally, **SVM** is used as a baseline for comparison. The system outputs accuracy and confusion matrix results.

## Prediction for a Single File

You can also test the model on a single file by providing the path to the audio file in the `single_file_path` variable:

```python
single_file_path = '/path/to/your/audio/file.wav'
```

The script will output the predicted emotion for the file and display the corresponding emotion image.

## Emotion Images

Ensure that you have the corresponding images for the classified emotions (calm, happy, fearful, disgust) stored at the following paths:

```plaintext
Project/
├── calm.jpg
├── happy.jpg
├── fearful.jpg
├── disgust.jpg
├──angry.jpg
├── fearful.jpg

```

The image will be displayed using `matplotlib` based on the predicted emotion.

## Results

The model is evaluated on the test set using accuracy scores and a confusion matrix. You can also test individual audio files to view the predicted emotion and corresponding image.

## References

For a deeper understanding of the topics covered, you can refer to:

1. El Ayadi, M., Kamel, M.S., & Karray, F. (2011). *Survey on speech emotion recognition*. Pattern Recognition.
2. Harár, P., Burget, R., & Dutta, M.K. (2021). *Speech emotion recognition using Deep Neural Networks*. IEEE.
3. Various other papers and sources (included in the project documentation).


