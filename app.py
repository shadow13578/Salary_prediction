import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Load the trained model ---
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'linear_regression_model.pkl' not found. Make sure it's in the same directory.")
    st.stop()

# --- Recreate LabelEncoders for consistency ---
# In a real deployment, you would save these encoders during training.
# For this example, we re-run the preprocessing steps on the original data
# to ensure consistent mappings.

# Load the raw data (or a representative sample) to fit encoders
# Ensure this path is correct if running locally
original_df = pd.read_csv('Salary_Data.csv') 

# Apply the same preprocessing steps for missing values as in the notebook
for column in ['Age', 'Years of Experience', 'Salary']:
    if column in original_df.columns:
        original_df[column].fillna(original_df[column].mean(), inplace=True)
for column in ['Gender', 'Education Level', 'Job Title']:
    if column in original_df.columns:
        original_df[column].fillna(original_df[column].mode()[0], inplace=True)
original_df.drop_duplicates(inplace=True)

le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()

# Fit encoders on the full preprocessed original data
le_gender.fit(original_df['Gender'].astype(str))
le_education.fit(original_df['Education Level'].astype(str))
le_job.fit(original_df['Job Title'].astype(str))

label_encoders = {
    'Gender': le_gender,
    'Education Level': le_education,
    'Job Title': le_job
}

# --- Streamlit UI ---
st.set_page_config(page_title='Salary Prediction App', layout='centered')
st.title('📊 Salary Prediction App')
st.markdown('Enter the details below to get a predicted salary based on our trained Linear Regression model.')

# Input fields
with st.form("prediction_form"):
    st.header("Employee Details")
    
    age = st.slider('Age', 18, 65, 30)
    
    gender_options = sorted(original_df['Gender'].astype(str).unique().tolist())
    gender = st.selectbox('Gender', gender_options)
    
    education_options = sorted(original_df['Education Level'].astype(str).unique().tolist())
    education_level = st.selectbox('Education Level', education_options)
    
    job_title_options = sorted(original_df['Job Title'].astype(str).unique().tolist())
    job_title = st.selectbox('Job Title', job_title_options)
    
    years_experience = st.slider('Years of Experience', 0.0, 40.0, 5.0, step=0.5)
    
    submitted = st.form_submit_button("Predict Salary")

    if submitted:
        # Encode categorical inputs
        try:
            encoded_gender = label_encoders['Gender'].transform([gender])[0]
            encoded_education = label_encoders['Education Level'].transform([education_level])[0]
            encoded_job_title = label_encoders['Job Title'].transform([job_title])[0]
        except ValueError as e:
            st.error(f"Error encoding categorical features: {e}. Please ensure all selected options are valid.")
            st.stop()
        
        # Create a DataFrame for prediction
        input_data = pd.DataFrame([[
            age, 
            encoded_gender, 
            encoded_education, 
            encoded_job_title, 
            years_experience
        ]], 
        columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
        
        # Predict
        predicted_salary = model.predict(input_data)[0]
        
        st.success(f'### Predicted Salary: **${predicted_salary:,.2f}**')

st.markdown("---")
st.info("This prediction is based on a Linear Regression model trained on the provided dataset. Performance may vary.")
