import mss
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define a list of possible labels (semantic classes)
labels = [
    "person", "car", "dog", "cat", "keyboard", "mouse", "window", "button", "icon",
    "menu", "image", "video", "document", "spreadsheet", "code", "browser", "folder",
    "file", "application", "desktop", "text"
]

# Set up screen capturing
with mss.mss() as sct:
    monitor = sct.monitors[1]  # You can change this index if you have multiple monitors
    while True:
        # Capture the screen
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)

        # Preprocess the image and labels for the CLIP model
        inputs = processor(
            text=labels,
            images=img,
            return_tensors="pt",
            padding=True
        )

        # Perform inference
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Get the top semantic labels
        top_probs, top_idxs = probs.topk(5)

        # Print the semantic labels with their probabilities
        print("Detected objects and their probabilities:")
        for idx, prob in zip(top_idxs[0], top_probs[0]):
            print(f"{labels[idx]}: {prob.item():.4f}")
        print("-" * 50)

        # Optional: Add a delay or a condition to break the loop
        # time.sleep(1)
