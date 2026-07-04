import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------
# Load Model
# ------------------------------

model = joblib.load("student_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.title("🎓 Student Performance")

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Model Information", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.metric("Algorithm", "Random Forest")
st.sidebar.metric("Accuracy", "95%")

# ==========================================================
# Prediction Page
# ==========================================================

if page == "Prediction":

    st.title("🎓 Student Performance Prediction System")
    st.markdown("Predict whether a student is likely to **PASS** or **FAIL** using Machine Learning.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input("👤 Student Name")

        gender = st.selectbox(
            "Gender",
            encoders["gender"].classes_
        )

        race = st.selectbox(
            "Race / Ethnicity",
            encoders["race/ethnicity"].classes_
        )

    with col2:

        parent = st.selectbox(
            "Parental Level of Education",
            encoders["parental level of education"].classes_
        )

        lunch = st.selectbox(
            "Lunch",
            encoders["lunch"].classes_
        )

        prep = st.selectbox(
            "Test Preparation Course",
            encoders["test preparation course"].classes_
        )

    st.markdown("---")

    st.subheader("📋 Input Summary")

    summary = pd.DataFrame({
        "Feature":[
            "Student Name",
            "Gender",
            "Race",
            "Parent Education",
            "Lunch",
            "Preparation"
        ],
        "Value":[
            student_name,
            gender,
            race,
            parent,
            lunch,
            prep
        ]
    })

    st.table(summary)

    st.markdown("---")

    if st.button("🚀 Predict Performance", use_container_width=True):

        input_data = pd.DataFrame({

            "gender":[
                encoders["gender"].transform([gender])[0]
            ],

            "race/ethnicity":[
                encoders["race/ethnicity"].transform([race])[0]
            ],

            "parental level of education":[
                encoders["parental level of education"].transform([parent])[0]
            ],

            "lunch":[
                encoders["lunch"].transform([lunch])[0]
            ],

            "test preparation course":[
                encoders["test preparation course"].transform([prep])[0]
            ]

        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("🎉 Student is likely to PASS")

            st.balloons()

        else:

            st.error("❌ Student is likely to FAIL")

        confidence = probability[prediction] * 100

        st.info(f"Prediction Confidence : {confidence:.2f}%")

        st.progress(int(confidence))

# ==========================================================
# Model Information
# ==========================================================

elif page == "Model Information":

    st.title("🤖 Model Information")

    st.markdown("---")

    st.write("### Machine Learning Algorithm")

    st.success("Random Forest Classifier")

    st.write("### Dataset")

    st.info("StudentsPerformance.csv")

    st.write("### Features Used")

    st.write("""
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
""")

    st.write("### Target")

    st.success("PASS / FAIL")

    st.write("### Libraries")

    st.code("""
Python
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
""")

# ==========================================================
# About Project
# ==========================================================

elif page == "About Project":

    st.title("📚 About Project")

    st.markdown("---")

    st.write("""
### Student Performance Prediction using Machine Learning

This project predicts whether a student is likely to PASS or FAIL
based on student demographic and educational information.

### Objective

To help educational institutions identify students who may need
additional academic support.

### Technologies Used

✔ Python

✔ Pandas

✔ NumPy

✔ Scikit-Learn

✔ Streamlit

✔ Joblib

### Machine Learning Algorithm

Random Forest Classifier

### Dataset

StudentsPerformance.csv

### Developed By

Atharva Gujar
""")

st.markdown("---")

st.caption("© 2026 Student Performance Prediction System | Machine Learning Project")