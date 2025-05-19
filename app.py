import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

@st.cache_resource
def load_gru_model():
    model = load_model('cheating_gru_model(0.88ac, 0.97 val).keras')
    return model

model = load_gru_model()
input_shape = model.input_shape  # e.g., (None, 30, 34)

st.title("GRU Model Prediction")

input_data = st.text_area("Enter your input sequence as comma-separated numbers:")

if st.button("Predict"):
    try:
        data = np.array([float(x) for x in input_data.strip().split(",")])
        seq_len, features = input_shape[1], input_shape[2]

        if data.size != seq_len * features:
            st.error(f"Expected {seq_len * features} values (sequence_length={seq_len}, features={features}), but got {data.size}.")
        else:
            data = data.reshape(1, seq_len, features)
            prediction = model.predict(data)
            st.write("Prediction:", prediction)
    except Exception as e:
        st.error(f"Error: {e}")
