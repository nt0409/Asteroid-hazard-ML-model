import xgboost
import streamlit as st
import joblib

# Load the model and scaler
model = joblib.load('asteroid_hazard_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

def prediction(a, b, c, d, v):
    # Scale the input
    scaled_input = scaler.transform([[a, b, c, d, v]])
    pred = model.predict(scaled_input)
    
    # Map numerical prediction to True/False
    return 'True' if pred[0] == 1 else 'False'

def prediction_using():
    st.title("Asteroid Hazard Predictor")
    
    # Get user input
    a = st.number_input('Estimated Diameter Min (km):', min_value=0.0, format="%.6f")
    b = st.number_input('Estimated Diameter Max (km):', min_value=0.0, format="%.6f")
    c = st.number_input('Relative Velocity (km/s):', min_value=0.0, format="%.6f")
    d = st.number_input('Absolute Magnitude:', min_value=0.0, format="%.2f")
    v = st.number_input('Miss Distance (km):', min_value=0.0, format="%.6f")

    if st.button("Predict"):
        res = prediction(a, b, c, d, v)
        st.success(f'Prediction: {res}')

if __name__ == "__main__":
    prediction_using()


'''import streamlit as st

st.title("Hello Streamlit")
'''