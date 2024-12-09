import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Set the page layout to wide mode for better control of elements
st.set_page_config(layout="wide")

# Custom CSS for dark background and contrast
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }
        .sidebar .sidebar-header {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying your image...")
        
        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"Predicted: {label} with {score * 100:.2f}% confidence")

# Function for CIFAR-10 model
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying your image...")
        
        # Load CIFAR-10 model
        model = tf.keras.models.load_model('cifar10_model.h5')
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

# Main function to control the navigation
def main():
    # Add a two-column layout for better space distribution
    col1, col2 = st.columns([1, 3])

    with col1:
        st.sidebar.title("Choose a Model")
        choice = st.sidebar.selectbox("Select a Model", ("CIFAR-10", "MobileNetV2 (ImageNet)"))
    
    with col2:
        st.sidebar.markdown("---")
        st.sidebar.write("Upload an image and select a model for classification.")
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()
