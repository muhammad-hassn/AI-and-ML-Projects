from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from PIL import Image
import os

# Model (note: big model — see tip below)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(img: Image.Image):
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning (BLIP + Gradio)"
)
if __name__ == "__main__":
    interface.launch(share=True)
