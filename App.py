from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import tensorflow_hub as hub
from text_processing import stem_or_lemmatize2
from text_processing import process_input_text
app = Flask(__name__)

# Load the saved models and other necessary components
ovr_lr_model = joblib.load('best_ovr_lr_SB_model.pkl')
mlb = joblib.load('mlb.pkl')
model_SBERT_loaded = hub.load("model_SBERT.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        lemmatized_text = stem_or_lemmatize2(input_text)
        
        # Encode the lemmatized text using the loaded Universal Sentence Encoder model
        X_new = model_SBERT_loaded.encode([lemmatized_text])
        
        y_new_pred = ovr_lr_model.predict(X_new)
        predicted_tags = mlb.inverse_transform(y_new_pred)

        return render_template('index.html', prediction_text='Predicted tags: {}'.format(predicted_tags), lemmatized_text=lemmatized_text)

if __name__ == '__main__':
    app.run(debug=True)
