# -*- coding: utf-8 -*-
# Ouvrir terminal jupiter notebook
# cd C:\Users\Lemel\OPC-P5
# python app3.py
# Exemple : This is a new  input code example for using py in panda with jupyter notebook please how i do it.

import gradio as gr
import nltk
import pandas as pd
import joblib
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load models
model_SBERT_loaded = joblib.load('model_SBERT.pkl')
ovr_lr_model = joblib.load('best_ovr_lr_SB_model.pkl')
mlb = joblib.load('mlb.pkl')

def stem_or_lemmatize2(text, method, remove_stopwords=True):
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    if method not in ('stem', 'lemmatize'):
        raise ValueError("Error: method must be 'stem' or 'lemmatize'")

    words = word_tokenize(text)
    if remove_stopwords:
        words = [word for word in words if word.lower() not in stop_words]

    if method == 'stem':
        words = [stemmer.stem(word) for word in words]
    else:  # method == 'lemmatize'
        words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def make_prediction(text, method):
    # Preprocess the input text
    processed_text = stem_or_lemmatize2(text, method)

    # Encode the processed text using the loaded Sentence-BERT model
    X_new = model_SBERT_loaded.encode([processed_text])

    # Make a prediction using the loaded OneVsRestClassifier model
    y_new_pred = ovr_lr_model.predict(X_new)

    # Convert the binary prediction back to tags using the trained MultiLabelBinarizer
    predicted_tags = mlb.inverse_transform(y_new_pred)

    # Return the processed text and predicted tags as a tuple
    return processed_text, ', '.join(predicted_tags[0])

output = gr.Interface(
    fn=make_prediction,
    inputs=[
        gr.inputs.Textbox(lines=5, placeholder="Enter text here..."),
        gr.inputs.Dropdown(choices=["stem", "lemmatize"], label="Method")
    ],
    outputs=[
        gr.outputs.Textbox(label="Processed Text"),
        gr.outputs.Textbox(label="Predicted Tags")
    ]
)

output.launch(share=True)
