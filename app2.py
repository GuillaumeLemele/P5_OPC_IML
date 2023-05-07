# -*- coding: utf-8 -*-
# Ouvrir terminal jupiter notebook
# cd C:\Users\Lemel\OPC-P5
# python app2.py


import gradio as gr
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

output = gr.Interface(
    fn=stem_or_lemmatize2,
    inputs=[
        gr.inputs.Textbox(lines=5, placeholder="Enter text here..."),
        gr.inputs.Dropdown(choices=["stem", "lemmatize"], label="Method")
    ],
    outputs=gr.outputs.Textbox()
)

output.launch(share=True)
