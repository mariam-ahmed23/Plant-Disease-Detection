import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from PIL import Image

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('plant_disease_model.h5')

model = load_trained_model()

# Define class names
class_names = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Tomato_Early_blight", "Tomato_Leaf_Mold"]

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    image = image.resize((150, 150))
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image_array, axis=0)

def predict(image):
    """Make a prediction on the processed image."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_index]
    class_name = class_names[class_index]  # Get the class name
    return class_name, confidence

def main():
    """Main function to run the Streamlit app."""
    st.title("plant disease classification")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            class_name, confidence = predict(image)
            st.success(f"Predicted Class: {class_name}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
