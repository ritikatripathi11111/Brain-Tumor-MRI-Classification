import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "custom_cnn_best.keras"
LABELS_PATH = ARTIFACT_DIR / "class_names.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Brain MRI Image Classification", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model_and_labels():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run the training notebook and save the model first."
        )
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Class labels not found: {LABELS_PATH}. Run the notebook's class mapping cell first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)
    labels = [mapping[str(i)] for i in range(len(mapping))]
    return model, labels

st.title("🧠 Brain Tumor MRI Image Classification")
st.write(
    "Upload an MRI image to see the class predicted by the trained educational model."
)
st.warning(
    "Research/educational use only. This tool is not a medical diagnosis and must not "
    "be used to make treatment decisions. Consult a qualified medical professional."
)

uploaded_file = st.file_uploader(
    "Choose an MRI image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        model, class_names = load_model_and_labels()

        st.image(image, caption="Uploaded MRI image", use_container_width=True)

        model, class_names = load_model_and_labels()
        resized = image.resize(IMG_SIZE)
        image_array = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(image_array, axis=0)

        with st.spinner("Running prediction..."):
            scores = model.predict(batch, verbose=0)[0]

        best_index = int(np.argmax(scores))
        st.subheader(f"Predicted dataset class: {class_names[best_index]}")
        st.caption(
            f"Model score: {scores[best_index] * 100:.2f}% "
            "(not a calibrated medical probability)"
        )

        score_table = {
            "Class": class_names,
            "Model score (%)": [round(float(score) * 100, 2) for score in scores],
        }
        st.dataframe(score_table, hide_index=True, use_container_width=True)
        import pandas as pd

        score_df = pd.DataFrame({
        "Class": class_names,
        "Model score (%)": [float(score) * 100 for score in scores]
        })

        st.bar_chart(
        score_df,
        x="Class",
        y="Model score (%)"
    )
    except Exception as exc:
        st.error(f"Could not run prediction: {exc}")
        st.info(
            "Check that the trained model and class_names.json are saved in the "
            "'artifacts' folder and that the required packages are installed."
        )

st.divider()
st.caption(
    "The prediction is limited to the dataset's four labels. It does not establish "
    "whether a person has a tumor and is not a substitute for radiologist review."
)
