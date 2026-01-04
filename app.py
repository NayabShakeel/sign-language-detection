import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Set page title
st.set_page_config(page_title="ASL Recognition", page_icon="🤟")

# Load your model and class names
@st.cache_resource # Keeps model in memory so it's fast
def load_assets():
    model = tf.keras.models.load_model("asl_model.keras")
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_assets()

st.title("🤟 ASL Sign Language Detection")
st.write("Upload an image of a hand sign to see what it means.")

# File uploader
file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("Waiting for an image...")
else:
    # Display the uploaded image
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image for the model
    # (Must match your 64x64 training size and 1/255 rescaling)
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # Show result
    st.success(f"Result: **{class_names[predicted_index]}**")
    st.progress(int(confidence))
    st.write(f"Confidence Score: {confidence:.2f}%")
