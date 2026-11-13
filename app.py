"""Small web demo. Run:  python app.py   (then open the local link it prints)"""
import gradio as gr
from model import CLASSES
from predict import load_model, predict

model = load_model("best.pt")


def classify(img):
    if img is None:  # user cleared the image box
        return {}
    return {name: p for name, p in predict(model, img, topk=len(CLASSES))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="CIFAR-10 ResNet classifier",
    description=(
        "Trained from scratch on CIFAR-10 (test accuracy 88.07%). It only knows 10 classes: "
        "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. "
        "Anything else will be forced into one of them, often with high confidence."
    ),
)

if __name__ == "__main__":
    demo.launch()
