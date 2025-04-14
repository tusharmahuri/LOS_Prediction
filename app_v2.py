import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)
    return rf_model, encoders, target_encoder

rf_model, encoders, target_encoder = load_model_and_encoders()

# App title
st.title("🏥 Hospital Stay Duration Prediction")
st.markdown("Fill in the details below to predict the **expected duration** of a patient's hospital stay.")

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    available_extra_rooms = st.number_input("Available Extra Rooms in Hospital", min_value=0, value=1)
    department = st.selectbox("Department", ['anesthesia', 'radiotherapy', 'gynecology', 'surgery', 'TB & Chest', 'orthopedics'])
    ward_type = st.selectbox("Ward Type", ['R', 'Q', 'P', 'S', 'T', 'U'])
    ward_facility_code = st.selectbox("Ward Facility Code", ['F', 'E', 'D', 'C', 'B', 'A'])
    bed_grade = st.selectbox("Bed Grade", [1.0, 2.0, 3.0, 4.0])

with col2:
    admission_type = st.selectbox("Type of Admission", ['Emergency', 'Trauma', 'Urgent'])
    severity = st.selectbox("Severity of Illness", ['Extreme', 'Moderate', 'Minor'])
    visitors = st.number_input("Visitors with Patient", min_value=0, value=4)
    age = st.selectbox("Age Range", ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'])
    deposit = st.number_input("Admission Deposit", min_value=0, value=5000)

# Centered predict button
center_button = st.columns([2, 1, 2])[1]
with center_button:
    predict = st.button("🔍 Predict Stay Duration")

# Prediction logic
if predict:
    # Create input DataFrame
    user_input = {
        'Available Extra Rooms in Hospital': available_extra_rooms,
        'Department': department,
        'Ward_Type': ward_type,
        'Ward_Facility_Code': ward_facility_code,
        'Bed Grade': bed_grade,
        'Type of Admission': admission_type,
        'Severity of Illness': severity,
        'Visitors with Patient': visitors,
        'Age': age,
        'Admission_Deposit': deposit
    }
    input_df = pd.DataFrame([user_input])

    # Encode categorical columns
    for col in input_df.columns:
        if col in encoders:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])

    # Predict and decode result
    prediction = rf_model.predict(input_df)
    decoded_prediction = target_encoder.inverse_transform(prediction)

    st.markdown("---")
    st.success(f"🩺 **Predicted Stay Duration:** {decoded_prediction[0]}")
