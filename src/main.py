from pathlib import Path

import streamlit as st
import torch

from loader import model_loader
from preprocessor import preprocess

st.title("Lung Cancer Histopathological Images Classifier")

# classes
classes = ["lung_adenocarcinoma", "lung_benign_tissue", "lung_squamous_cell_carcinoma"]

# checkpoint loader
model_path = Path("src/checkpoint/model_v4_0199.pt")
model = model_loader(model_path)

# file uploader
image_file = st.file_uploader("Choose a file")

# inference
if image_file is not None:
    # To read file as bytes:
    bytes_data = image_file.getvalue()

    # preprocessing and predictions
    image_tensor = preprocess(bytes_data)
    prediction = model.forward(image_tensor)

    # outputs
    _, pred = torch.max(prediction, 1)

    # file viewer
    st.image(bytes_data, caption="Uploaded image")
    st.write(f"Predicted Class: {classes[pred.item()]}")

st.subheader("Data Details")
st.text("Lorem Ipsum Dolor")

st.subheader("Model Details")
st.text("Lorem Ipsum Dolor")
