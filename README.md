# Heart Disease Risk Predictor

A user-friendly web application that uses machine learning to help identify individuals at risk for heart disease. Designed for clinicians, healthcare workers, and early screening scenarios, this tool provides an instant prediction based on lifestyle and clinical data inputs.

---

## About the App

This project is a **heart disease risk screening tool** built with:

- **Streamlit** (Python-based web app framework)
- **scikit-learn** for model training
- A clean and responsive UI with custom CSS

Users can input patient information such as age, BMI, smoking habits, diabetic status, and more. The app instantly classifies the risk as **"At Risk"** or **"Not At Risk"** and provides a **confidence score** and a feature importance chart.

---

## Machine Learning Model

The app uses a **Random Forest Classifier** trained on the [Heart Disease 2020 dataset](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).

### Model Pipeline

The model pipeline includes:

- **Data Cleaning**: Handling missing values and duplicates
- **Numerical Scaling**: Using `StandardScaler`
- **Categorical Encoding**: Using `OneHotEncoder`
- **Classifier**: `RandomForestClassifier` (100 estimators, random_state=42)
- **Train-Test Split**: 80% training, 20% testing

> ✅ Achieved ~**accuracy of 85–87%** on test data.

The full model pipeline is saved as:
- `model_pipeline.pkl` – serialized model
- `feature_metadata.json` – stores numerical and categorical feature names

---

## Screenshots

### Home Page
![Home](screenshots/home.png)

### Input Form
![Form](screenshots/form.png)

### Prediction Output
![Result](screenshots/result.png)

---

## How to Run the App

### Prerequisites

Make sure you have Python 3.9 or later installed. Then install dependencies:

```bash
pip install -r requirements.txt
