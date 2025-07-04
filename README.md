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


If you don’t have a requirements.txt file, you can manually install the necessary packages with:

bash
Copy
Edit
pip install streamlit scikit-learn pandas matplotlib joblib
▶️ Running the App
After installing the dependencies, run the app using:

bash
Copy
Edit
streamlit run app.py
This will open the app in your default web browser at http://localhost:8501.

📁 File Structure
bash
Copy
Edit
heart-disease-predictor/
│
├── app.py                   # Streamlit app
├── model_pipeline.pkl       # Trained ML model (Random Forest)
├── feature_metadata.json    # Feature metadata (numerical/categorical lists)
├── heart_2020_uncleaned.csv # Dataset used for training
├── requirements.txt         # List of dependencies
├── screenshots/             # Folder with UI screenshots
│   ├── home.png
│   ├── form.png
│   └── result.png
└── README.md                # Project documentation
📝 Notes
The ML model is trained on the Heart Disease 2020 dataset.

This tool is intended for screening and assistance, not as a definitive diagnosis.

The Risk Threshold slider lets users adjust the sensitivity of the prediction.

🧑‍💻 Author
Developed by Julia Verzosa
BS Computer Science – University of Mindanao
GitHub Profile

📄 License
This project is licensed under the MIT License.