import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt




# ---------- Load model & metadata ----------
@st.cache_resource
def load_model_and_meta():
    pipe = joblib.load('model_pipeline.pkl')
    meta = json.load(open('feature_metadata.json'))
    return pipe, meta

pipe, meta = load_model_and_meta()

# ---------- Page Config ----------
st.set_page_config(
    page_title='Heart Disease Risk Predictor',
    layout='centered'
)

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    html, body {
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton > button {
        border-radius: 8px;
        background-color: #FF6B81;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #e95f74;
    }
    .stSelectbox > div, .stNumberInput > div {
        border-radius: 6px;
    }
    .result-box {
        background-color: #fff9fa;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #FF6B81;
        margin-top: 1.2rem;
    }
    .highlight-risk {
        color: #d62828;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .highlight-safe {
        color: #2a9d8f;
        font-size: 1.5rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.title("Heart Disease Risk Predictor")
st.write("This tool helps clinicians and staff identify patients who may be at risk for heart disease. Fill out the form to get an instant prediction.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("💡 About This Tool")
    st.markdown("""
This prediction tool uses a machine learning model trained on clinical and lifestyle data.  
It gives a **quick classification** and a **confidence score** to assist early decisions.

> **Note**: This is not a diagnostic tool. Results should always be reviewed by a physician.
""")
    st.markdown("---")
    threshold_percent = st.slider('🛑 Risk Threshold (%)', 0, 100, 50, 1)
    threshold = threshold_percent / 100

# ---------- Input Form ----------
with st.form("input_form"):
    st.subheader("Enter Patient Details")

    # ---- Health Metrics Category ----
    with st.expander("🔹 Health Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            bmi = st.number_input('BMI (Body Mass Index)', 10.0, 60.0, step=0.1)
            physical_health = st.number_input('Physical Health (days unwell in past 30)', 0.0, 30.0, step=1.0)
        with col2:
            mental_health = st.number_input('Mental Health (days unwell in past 30)', 0.0, 30.0, step=1.0)
            sleep_time = st.number_input('Sleep Time (avg hours/day)', 0.0, 24.0, step=0.5)

    # ---- Medical History & Lifestyle Category ----
    with st.expander("🔸 Medical History & Lifestyle", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            smoking = st.selectbox('Smoking', ['Yes', 'No'])
            alcohol = st.selectbox('Alcohol Drinking', ['Yes', 'No'])
            stroke = st.selectbox('Stroke History', ['Yes', 'No'])
            diff_walking = st.selectbox('Difficulty Walking', ['Yes', 'No'])
            sex = st.selectbox('Sex', ['Male', 'Female'])
            age_category = st.selectbox('Age Category', [
                '18-24','25-29','30-34','35-39','40-44','45-49',
                '50-54','55-59','60-64','65-69','70-74','75-79','80+'
            ])
        with col4:
            race = st.selectbox('Race', ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'])
            diabetic = st.selectbox('Diabetic Status', ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])
            physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])
            gen_health = st.selectbox('General Health', ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
            asthma = st.selectbox('Asthma', ['Yes', 'No'])
            kidney_disease = st.selectbox('Kidney Disease', ['Yes', 'No'])
            skin_cancer = st.selectbox('Skin Cancer', ['Yes', 'No'])

    submitted = st.form_submit_button("🔍 Check Risk")

# ---------- Prediction Output ----------
if submitted:
    sample = pd.DataFrame([{
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'SleepTime': sleep_time,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol,
        'Stroke': stroke,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age_category,
        'Race': race,
        'Diabetic': diabetic,
        'PhysicalActivity': physical_activity,
        'GenHealth': gen_health,
        'Asthma': asthma,
        'KidneyDisease': kidney_disease,
        'SkinCancer': skin_cancer
    }])

    proba = pipe.predict_proba(sample)[0][1]
    label = 'At Risk' if proba >= threshold else 'Not At Risk'
    label_class = 'highlight-risk' if label == 'At Risk' else 'highlight-safe'

    st.markdown("---")
    st.subheader("Prediction Result")
    st.markdown(f"Risk Classification: <span class='{label_class}'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence Score:** `{proba:.2%}`")
    st.markdown(f"_Threshold Used: {threshold:.2f}_")

    # Result explanation paragraph
    st.markdown("""
    <div style="
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        background-color: #f4f4f4;
        border-left: 4px solid #FF6B81;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #333;
    ">
        <p style="margin-bottom: 0.5rem;">
            <strong>Interpretation:</strong> The model predicts that there is a <strong>{:.0f}% probability</strong> that this patient may be <strong>{}</strong> for developing heart disease.
        </p>
        <p style="margin: 0;">
            This result is based on the health and lifestyle information provided. It is not a clinical diagnosis and should only be used as a preliminary screening tool. For accurate diagnosis and treatment, please consult a licensed medical professional.
        </p>
    </div>
    """.format(proba * 100, label.upper()), unsafe_allow_html=True)


# ---------- Feature Importance ----------
if st.checkbox("Show top 10 important features"):
    try:
        importances = pipe.named_steps['clf'].feature_importances_
        raw_feat_names = pipe.named_steps['pre'].get_feature_names_out()
        clean_feat_names = [name.split("__")[1] if "__" in name else name for name in raw_feat_names]

        imp_df = pd.DataFrame({'feature': clean_feat_names, 'importance': importances})
        top10 = imp_df.sort_values(by='importance', ascending=False).head(10)

        fig, ax = plt.subplots()
        top10.plot(kind='barh', x='feature', y='importance', ax=ax, legend=False, color='#FF6B81')
        ax.invert_yaxis()
        ax.set_title("Top 10 Most Important Features", fontsize=14)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Could not display feature importance.")
        st.text(str(e))

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 0.9rem; color: gray;">
    ⚠️ <i>This tool is intended for preliminary health screening purposes only and should not be used for diagnosis or treatment. Always consult a licensed healthcare professional.</i><br><br>
     Designed & Developed with care by <strong>Julia Verzosa</strong> | BS Computer Science – University of Mindanao<br>
    © July 2025 Julia Verzosa. All rights reserved.
</div>
""", unsafe_allow_html=True)
