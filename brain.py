import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# --- Constants ---
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10


# --- Data Loader ---
def load_data(data_dir):
    categories = ['yes', 'no']
    data = []

    for category in categories:
        path = os.path.join(data_dir, category)
        label = 1 if category == 'yes' else 0

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append([img, label])

    np.random.shuffle(data)
    X, y = zip(*data)
    return np.array(X), np.array(y)


# --- Model Builder ---
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- Main Training Pipeline ---
def train_model(data_dir):
    X, y = load_data(data_dir)
    X = X / 255.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
    model.save("brain_tumor_model.h5")


# --- Prediction Function ---
def predict_image(model, image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))
    pred = model.predict(img)[0][0]
    return 'Tumor is present' if pred > 0.5 else 'No Tumor is not present'


# --- Streamlit UI ---
st.title("🧠 Brain Tumor Detector")

if st.button("Train Model"):
    train_model("augmented data")
    st.success("Model trained and saved as 'brain_tumor_model.h5'")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    if os.path.exists("brain_tumor_model.h5"):
        from tensorflow.keras.models import load_model

        model = load_model("brain_tumor_model.h5")
        result = predict_image(model, image)
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Model not found. Please train the model first.")
