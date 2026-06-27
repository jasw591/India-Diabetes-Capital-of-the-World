import streamlit as st
import pandas as pd
import numpy as np

# Set up the look of the web page
st.set_page_config(page_title="Diabetes Risk App", layout="centered")

st.title("🩺 Deployed Diabetes Prediction App")
st.write("Input patient data below to run the live K-Means + CatBoost hybrid pipeline.")

st.markdown("---")

# 1. Create columns for user inputs
col1, col2 = st.columns(2)

with col1:
    glucose = st.slider("Glucose Level", 0, 200, 120)
    blood_pressure = st.slider("Blood Pressure", 0, 130, 70)
    insulin = st.slider("Insulin Level", 0, 800, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 32.0, step=0.1)

with col2:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 21, 100, 33)

# 2. When the user clicks the button, run the predictions
if st.button("Predict Diabetes Risk", type="primary"):
    # Access the models currently active in the main Colab space
    import __main__
    
    # Structure the inputs into a dataframe matching your training setup
    raw_data = pd.DataFrame([{
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }])
    
    # Scale the metrics for the unsupervised model just like we did in EDA
    scaled_data = __main__.scaler.transform(raw_data)
    
    # Step A: Unsupervised K-Means predicts the cluster profile
    assigned_cluster = __main__.kmeans.predict(scaled_data)[0]
    
    # Step B: Add the cluster column back to the raw inputs for CatBoost
    raw_data['Cluster'] = assigned_cluster
    
    # Step C: Supervised CatBoost predicts the final probability and binary label
    prob = __main__.cat_model.predict_proba(raw_data)[0][1]
    prediction = __main__.cat_model.predict(raw_data)[0]
    
    # 3. Show the final results
    st.markdown("---")
    st.subheader("Results:")
    st.write(f"🧬 **Automated Patient Group:** Cluster {assigned_cluster}")
    st.write(f"📊 **Calculated Statistical Risk:** {int(prob * 100)}%")
    
    if prediction == 1:
        st.error("⚠️ **Result:** High Risk (The model predicts the patient is positive for diabetes).")
    else:
        st.success("✅ **Result:** Low Risk (The model predicts the patient is negative for diabetes).")
