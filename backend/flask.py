
from flask import Flask, request, jsonify

from flask_cors import CORS

import tensorflow as tf

import numpy as np


app = Flask(__name__)

CORS(app)

# Load your TensorFlow model (replace 'model_path' with the actual path to your trained models)

model_path_cnn = 'path/to/your/cnn_model'

model_path_rnn = 'path/to/your/rnn_model'

# Load other models similarly


model_cnn = tf.keras.models.load_model(model_path_cnn)

model_rnn = tf.keras.models.load_model(model_path_rnn)

# Load other models similarly

tokenizer = tf.keras.preprocessing.text.Tokenizer()


# Dummy function for tokenization (replace with your actual tokenization logic)

def tokenize_text(text):

    return tokenizer.texts_to_sequences([text])[0]


def predict_cnn(text):

    # Implement your CNN prediction logic

    # Replace this with actual preprocessing and prediction steps

    tokenized_text = tokenize_text(text)

    prediction = model_cnn.predict(np.array([tokenized_text]))

    return prediction.tolist()

def predict_rnn(text):

    # Implement your RNN prediction logic

    # Replace this with actual preprocessing and prediction steps

    tokenized_text = tokenize_text(text)

    prediction = model_rnn.predict(np.array([tokenized_text]))

    return prediction.tolist()

# Implement similar functions for other models
@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()

    if 'text' not in data:

        return jsonify({"error": "Please provide 'text' in the request body"}), 400

    text = data['text']


    # Example: Use CNN for prediction

    prediction_cnn = predict_cnn(text)



    # Example: Use RNN for prediction

    prediction_rnn = predict_rnn(text)


    # Combine predictions or choose the most relevant one based on your use case

    combined_prediction = {"cnn": prediction_cnn, "rnn": prediction_rnn}


    return jsonify({"prediction": combined_prediction})


if __name__ == '__main__':

    app.run(debug=True)
