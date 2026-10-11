
from pathlib import Path
import json

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "crop_classifier.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Crop Disease Classifier",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Disease Classification")

st.write(
    "Upload a crop-leaf image to obtain a prediction "
    "from our trained CNN model."
)

st.info(
    "This version classifies the 38 categories learned "
    "from the PlantVillage dataset."
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    return json.loads(
        CLASS_NAMES_PATH.read_text(encoding="utf-8")
    )


if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error(
        "Model files are missing. Please train the model first."
    )
    st.stop()


model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        if st.button("Classify Image"):
            with st.spinner("Analyzing the image..."):

                resized_image = image.resize(IMG_SIZE)

                image_array = np.array(
                    resized_image,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = float(
                    predictions[predicted_index] * 100
                )

            st.subheader("Prediction Results")

            st.write(
                "**Predicted category:**",
                predicted_class.replace("___", " — ")
            )

            st.metric(
                "Model confidence",
                f"{confidence:.2f}%"
            )

            if confidence < 60:
                st.warning(
                    "The model has low confidence in this "
                    "prediction. Try a clear leaf image."
                )

            st.caption(
                "The confidence score is the model's "
                "predicted probability, not a guarantee "
                "that the diagnosis is correct."
            )

    except Exception as error:
        st.error(f"Unable to process this image: {error}")


st.divider()

st.caption(
    "Academic project: Deep Learning-Based Crop Disease "
    "Classification Using CNNs."
)

st.warning(
    "This model is an educational prototype trained on "
    "PlantVillage images. Predictions on real-world images "
    "may be less reliable. Do not use it as the sole basis "
    "for crop treatment decisions."
)
