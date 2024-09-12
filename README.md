# Spam Detection Application 📩

**Click here for the app:** [Spam Detection Application](https://spamdetection-deployment.streamlit.app/)

This Streamlit application uses a pre-trained model to detect whether a given text message is spam or not. Users can enter a message, and the model will classify it as spam or not spam.

## Features

- **Home** 🏠: Displays a welcome message and instructions on how to use the app.
- **Message Input** ✍️: Allows users to enter a message to check if it is spam or not.
- **Prediction** 🔮: The app predicts whether the entered message is spam or not and displays the result.

## Code Explanation

### Page Configuration
Sets the title of the web page to "Spam Detection."

### Model Loading
Defines and loads a pre-trained model from a saved file (`spam_detection.h5`) used for predicting whether a message is spam or not.

### Text Preprocessing
Includes text preprocessing steps such as:
- Removing non-alphabetic characters
- Converting text to lowercase
- Tokenizing the text
- Removing stopwords
- Applying stemming

### Prediction Function
- **`predict_spam(text)`**: Preprocesses the text, tokenizes and pads the sequence, and uses the model to make a prediction. Returns `0` for spam and `1` for not spam.

### Streamlit Application
- **Title and Instructions**: Displays the title and provides instructions on how to use the app.
- **Message Input**: Users enter a message into a text area.
- **Prediction**: When the "Predict" button is clicked, the app displays whether the message is spam or not.

## Trained Model Link

The trained model used in this application is available for download at the following link:

[Trained Model - Spam Detection](https://github.com/Abeshith/Spam-Detection)

## Usage

1. **Home** 🏠: Visit the home page for an introduction and usage instructions.
2. **Message Input** ✍️: Enter a message to classify it. The app will display the prediction as "Spam" or "Not Spam."
3. **Prediction** 🔮: Click the "Predict" button to get the classification result.

## Requirements

- Streamlit
- TensorFlow
- NLTK
- NumPy

Ensure all required packages are installed before running the app. 📦

