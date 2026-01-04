import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="ASL Sign Detector", page_icon="🤟", layout="centered")

# 2. Load Assets
@st.cache_resource
def load_assets():
    # Make sure these filenames match exactly what you downloaded
    model = tf.keras.models.load_model("asl_model.keras")
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return model, class_names

model, class_names = load_assets()

# Open the image
image = Image.open(uploaded_file).convert('RGB') # Ensure 3 color channels

# 1. Resize to match training
img_resized = image.resize((64, 64))

# 2. Convert to array and SCALE (Crucial Step!)
img_array = np.array(img_resized) / 255.0 

# 3. Add the batch dimension (1, 64, 64, 3)
img_array = np.expand_dims(img_array, axis=0)

# Now predict
predictions = model.predict(img_array)

# 3. Sidebar for Settings
st.sidebar.header("Settings")
# Adding the slider you requested
threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=0, max_value=100, value=85, step=5)
threshold_decimal = threshold / 100

st.sidebar.info(f"The model will only give a result if it is more than {threshold}% sure.")

# 4. Main UI
st.title("🤟 ASL Sign Language Detector")
st.write("Upload a hand sign image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and show image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Hand Sign', use_container_width=True)
    
    # Preprocessing (64x64 and 1/255 rescaling)
    img_resized = image.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    with st.spinner('Analyzing sign...'):
        predictions = model.predict(img_array)
        
    conf_scores = predictions[0] # All 36 probabilities
    max_conf = np.max(conf_scores)
    predicted_class = class_names[np.argmax(conf_scores)]

    # 5. Threshold Logic
    st.divider()
    if max_conf >= threshold_decimal:
        st.success(f"### Prediction: **{predicted_class}**")
        st.metric("Confidence Score", f"{max_conf*100:.2f}%")
    else:
        st.warning(f"### Result: **Uncertain**")
        st.write(f"The model is only {max_conf*100:.2f}% sure, which is below your {threshold}% threshold.")

    # 6. Result Graph (Probability Distribution)
    st.write("### Confidence Distribution")
    # Create a simple DataFrame for the bar chart
    chart_data = pd.DataFrame({
        'Sign': class_names,
        'Probability': conf_scores
    }).set_index('Sign')
    
    # Sort so the highest bars are at the top/front
    st.bar_chart(chart_data)

else:
    st.info("Please upload an image file to begin.")
