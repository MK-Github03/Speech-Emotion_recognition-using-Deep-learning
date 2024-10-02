 **Speech Emotion Recognition Using Deep Learning**

#ps: this is an open source project, so feel free to replicate the project :)

This project implements a Speech Emotion Recognition system using Recurrent Neural Networks (RNN) with Long Short-Term Memory (LSTM) layers. The system processes speech audio files, extracting Mel-frequency cepstral coefficients (MFCC) features, to classify emotions such as anger, happiness, sadness, and neutrality.

The application is highly relevant for areas such as mental health monitoring, customer service, and interactive AI applications like virtual assistants.


Features

- Emotion Classification: The system identifies and categorizes emotions from audio data.
- Deep Learning Architecture: Utilizes RNN with LSTM to model the temporal sequence of speech.
- MFCC Feature Extraction: Extracts MFCC features from audio data to represent sound for processing.
- Versatile Applications: Suitable for healthcare, entertainment, customer service, virtual assistants, and more.

Dataset

This project processes `.wav` format audio files for training and testing. The dataset consists of categorized speech samples representing different emotions. MFCC features are extracted from these samples and fed into the LSTM-based model.
You can find the dataset from my repo.

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

Preparing the Dataset

Make sure the `.wav` files are organized in the specified folder structure, where the folder names correspond to emotion labels (e.g., "angry", "happy", etc.). The labels will be automatically inferred during training.


Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. Install dependencies:
   Ensure you have [Python 3.x](https://www.python.org/downloads/) installed, then run:
   ```
   pip install -r requirements.txt
   ```
   (Find the requirements.txt attachment in the main branch)
   
3. Prepare the dataset:
   Organize the `.wav` files as per the directory structure provided above.

Model Training and Testing

Train the Model

To train the model, run:
```
python train_model.py
```

This will:
- Load the training data
- Extract MFCC features from the audio files
- Train the LSTM-based RNN model

Test the Model

To evaluate the model’s performance, run:
```
python test_model.py
```
The model will be tested on a separate test dataset, and metrics such as accuracy and loss will be displayed.


Model Architecture

- MFCC Feature Extraction: 13 MFCC features are extracted using Librosa.
- RNN Layers: Two LSTM layers are used to model temporal dependencies in the speech signal.
- Fully Connected Layers: After the LSTM layers, fully connected layers are applied to classify the input speech into one of the predefined emotions (angry, happy, sad, neutral).

Model Summary:
- Input: 13 MFCC coefficients per time step
- LSTM Layers: 2 layers, each with 128 hidden units
- Fully Connected Layer: 64 units
- Output Layer: 4 classes (angry, happy, sad, neutral)


Applications

- Mental Health: Analyze speech patterns to monitor emotional health.
- Customer Service: Detect emotional states in customer interactions and adapt responses accordingly.
- Virtual Assistants: Enhance virtual assistant interactions by understanding user emotions.
- Entertainment and Gaming: Create more immersive user experiences.
- Marketing: Analyze customer feedback and emotional reactions to improve engagement.

Contribution

Contributions are welcome! Fork the repository and open a pull request with your proposed changes. Feel free to add new features, improve performance, or expand the dataset for additional emotions.
