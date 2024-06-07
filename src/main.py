from pathlib import Path
import time

import streamlit as st
import torch

from loader import model_loader
from preprocessor import preprocess

st.title("Lung Cancer Histopathological Images Classifier")
if st.checkbox("Data and Model Details"):
    st.subheader("Dataset Details")
    st.markdown("""
        The dataset contains 25,000 histopathological images with 5 classes. All images are 768 x 768 pixels in size and are in jpeg file format.
        The images were generated from an original sample of HIPAA compliant and validated sources, consisting of 750 total images of lung tissue (250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas) and 500 total images of colon tissue (250 benign colon tissue and 250 colon adenocarcinomas) and augmented to 25,000 using the Augmentor package.
        There are five classes in the dataset, each with 5,000 images, being:

        - Lung benign tissue
        - Lung adenocarcinoma
        - Lung squamous cell carcinoma
        - Colon adenocarcinoma
        - Colon benign tissue
        """)

    st.markdown("---")

    st.subheader("Model Details")
    st.image("src/images/training_metrics_overtime.png", caption="Training Metrics Overtime")
    st.markdown("""
        ```
        Test Loss: 0.018140
        ```
        """)
    st.markdown("""
           ```
           Test Accuracy of Lung Adenocarcinoma: 99% (716/717)
           Test Accuracy of Lung Benign Tissue: 100% (763/763)
           Test Accuracy of Lung Squamous Cell Carcinoma: 97% (753/770)
           ```
           """)
    st.markdown("""
        ```
        Test Accuracy (Overall): 99% (2232/2250)
        ```
        """)
st.markdown("---")

# classes
classes = ["Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinomaa"]

# checkpoint loader
model_path = Path("src/checkpoint/model_v4_0199.pt")
model = model_loader(model_path)

# file uploader
image_file = st.file_uploader("Upload Lung Image", type=["png", "jpg", "jpeg"])

# inference
if image_file is not None:
    # To read file as bytes:
    bytes_data = image_file.getvalue()

    # preprocessing and predictions
    start = time.time()
    image_tensor = preprocess(bytes_data)
    prediction = model.forward(image_tensor)
    end = time.time()

    # outputs
    _, pred = torch.max(prediction, 1)

    # file viewer
    st.image(bytes_data, caption="Uploaded image")
    st.markdown(f"""
    ```
    Predicted Class: {classes[pred.item()]}
    ```
    ```
    Inference Time: {(end - start):.4f} seconds
    ```
    """)
