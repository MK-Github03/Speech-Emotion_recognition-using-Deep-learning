 Speech Emotion Recognition Using Deep Learning

Overview

**#ps: this is an open source project, so feel free to replicate the project :)**

This project implements a Speech Emotion Recognition system using Recurrent Neural Networks (RNN) with Long Short-Term Memory layers. The model processes speech audio data, extracting MFCC (Mel-frequency cepstral coefficients) features, to classify emotions such as anger, happiness, sadness, and others. This technology can be applied in fields such as mental health monitoring, customer service, and interactive applications.

Features

- Emotion Prediction: Classifies emotions based on speech audio signals.
- Deep Learning Model: Utilizes an LSTM architecture for processing temporal data like speech.
- MFCC Feature Extraction: Extracts features from audio files to feed into the model.
- Wide Application: Useful in various fields like healthcare, customer service, and virtual assistants.

Dataset

The project uses `.wav` format audio files for training and testing. The dataset consists of speech samples categorized by emotion, and MFCC features are extracted from the audio signals for the LSTM model.

Example Directory Structure:
```
emotion_dataset/
    ├── train/
    │   ├── angry/
    │   ├── happy/
    │   ├── sad/
    │   └── neutral/
    └── test/
        ├── angry/
        ├── happy/
        ├── sad/
        └── neutral/
```

Each folder contains `.wav` files representing different emotions. The labels are automatically inferred from the file or folder names.

Installation

1. Clone the repository:
   ```
   git clone https://github.com/mk-github03/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. Install dependencies:
   Ensure you have [Python 3.x](https://www.python.org/downloads/) installed. Then, install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

   

3. Organizing the dataset:
   Make sure your dataset of `.wav` files is organized as shown in the example structure above.



1. Train the Model
   To train the model on your dataset, run the following command:
   ```
   python your_script_name.py
   ```
   (modify the file name based on your code)
   
   This will:
   - Load the training data.
   - Extract MFCC features from the audio.
   - Train the LSTM model on the data.

 2. Test the Model
   The testing step will evaluate the model's accuracy on the test data:
   ```
   python your_script_name.py
   ```
 (modify the file name based on your code)
 
   During testing, the model's performance will be evaluated on the held-out test dataset, and accuracy/loss will be displayed.

 Model Architecture

- MFCC Feature Extraction: The model uses **Librosa** to extract 13 MFCC features from each audio file.
- LSTM Layers: The LSTM model captures temporal dependencies in the MFCC sequences.
- Fully Connected Layers: After the LSTM layers, the model uses fully connected layers to classify the speech signal into one of the emotion categories (e.g., angry, happy, sad, neutral).

 Model Summary:
- Input: 13 MFCC coefficients per time step
- Hidden Layers: 2 LSTM layers (128 hidden units)
- Fully Connected Layer: 64 units
- Output: 4 emotion classes



Expected output (after training):
```
Train Epoch: 1 [0/1000 (0%)]    Loss: 1.234567
...
Test set: Average loss: 0.4567, Accuracy: 850/1000 (85%)
```

 Applications

- **Mental Health**: Monitor patients’ emotional state over time.
- **Customer Service**: Analyze customer calls to detect emotional responses and provide real-time feedback.
- **Entertainment**: Enhance gaming experiences or virtual assistants.
- **Marketing**: Analyze emotional reactions to advertisements or products.

Contribution

Contributions are welcome! If you would like to contribute to improving this model, feel free to fork the repository and open a pull request with your changes.

