import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

# Page config
st.set_page_config(page_title="Tourism Package Prediction", layout="wide")

# Download and load the model
model_path = hf_hub_download(repo_id="pragmat/Tourism", filename="best_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Main title and description
st.title("Tourism Package Purchase Prediction")
st.markdown("""
**Predict whether a customer will purchase a tourism package** based on their demographic and travel preferences.
Enter customer details below to get a prediction!
""")

# Sidebar for inputs with proper feature names from your dataset
st.sidebar.header("Customer Profile")
st.sidebar.markdown("---")

# Numeric features (matching your dataset)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=6, value=2)
    num_trips = st.number_input("Number of Trips", min_value=0, max_value=10, value=1)
with col2:
    monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=100000, value=50000)
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x else "No")

col3, col4 = st.columns(2)
with col3:
    own_car = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
with col4:
    duration_pitch = st.slider("Duration of Pitch (days)", 1, 30, 7)
    preferred_star = st.slider("Preferred Property Star Rating", 1, 5, 3)

# Categorical features
st.sidebar.markdown("---")
st.sidebar.subheader("Demographics & Preferences")

city_tier = st.sidebar.selectbox("City Tier", [1, 2, 3], format_func=lambda x: f"Tier {x}")
occupation = st.sidebar.selectbox("Occupation", [
    "Employee", "Self Employed", "Housewife", "Student", "Business"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Single", "Married", "Divorced"
])
designation = st.sidebar.selectbox("Designation", [
    "Executive", "Manager", "Senior Manager", "Director"
])
product_pitched = st.sidebar.selectbox("Product Pitched", [
    "Basic", "Standard", "Premium", "Deluxe"
])
type_of_contact = st.sidebar.selectbox("Type of Contact", ["Email", "Self Enquiry"])

# Prepare input data with correct column names and proper encoding
input_data_dict = {
    'Age': age,
    'NumberOfPersonVisiting': num_person,
    'PreferredPropertyStar': preferred_star,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction,
    'NumberOfFollowups': num_followups,
    'DurationOfPitch': duration_pitch,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'MaritalStatus': marital_status,
    'Designation': designation,
    'ProductPitched': product_pitched
}

input_df = pd.DataFrame([input_data_dict])

# Main prediction section
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
  summary_df = pd.DataFrame({
      "Feature": ["Age", "Income", "City Tier", "Product", "Trips"],
      "Value": [
        f"{age} yrs",
        f"₹{monthly_income:,}",
        f"Tier {city_tier}",
        product_pitched,
        str(num_trips)
      ]
    })
  st.dataframe(summary_df, use_container_width=True)

    # st.subheader("Customer Summary")
    # summary_df = pd.DataFrame([{
    #     "Feature": ["Age", "Income", "City Tier", "Product", "Trips"],
    #     "Value": [f"{age} yrs", f"₹{monthly_income:,}", f"Tier {city_tier}", product_pitched, num_trips]
    # }])
    # st.dataframe(summary_df, use_container_width=True)

if st.button("Predict Package Purchase", type="primary", use_container_width=True):
    with st.spinner("Generating prediction..."):
        try:
            # Get prediction probabilities
            prediction_proba = model.predict_proba(input_df)[:, 1][0]
            prediction = model.predict(input_df)[0]

            # Results
            st.subheader("Prediction Results")

            col_a, col_b = st.columns(2)
            with col_a:
                probability = prediction_proba * 100
                st.metric(
                    label="Purchase Probability",
                    value=f"{probability:.1f}%",
                    delta=f"{probability:.1f}% chance"
                )

            with col_b:
                result = "**Will Purchase**" if prediction == 1 else "**Won't Purchase**"
                st.markdown(result)

            # Confidence bar
            st.progress(prediction_proba)

            # Recommendation
            if prediction == 1:
                st.success("**High conversion potential!** Prioritize follow-up calls and personalized offers.")
            else:
                st.warning("**Low conversion likelihood.** Consider alternative products or nurturing strategy.")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Ensure all input features match your training data exactly.")


