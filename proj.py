import gradio as gr
from transformers import pipeline

# Load the emotion analysis model
emotion_model = pipeline("sentiment-analysis")

def analyze_emotion(text):
    # Analyze the emotion in the provided text
    result = emotion_model(text)[0]
    emotion = result['label']
    score = result['score']
    return f"Emotion: {emotion}, Score: {score:.4f}"

# Create a Gradio interface
iface = gr.Interface(
    fn=analyze_emotion,
    inputs=gr.Textbox(),
    outputs="text"
)

# Launch the interface on a local server
iface.launch()
