# -*- coding: utf-8 -*-
# Ouvrir terminal jupiter notebook
# cd C:\Users\Lemel\OPC-P5
# python app.py


import gradio as gr


def greet(text):
    return text

output = gr.Interface(
    fn=greet,
    inputs=gr.inputs.Textbox(lines=5, placeholder="Enter text here..."),
    outputs=gr.outputs.Textbox()
)

output.launch(share=True)



