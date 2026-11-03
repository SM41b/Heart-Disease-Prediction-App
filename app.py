
import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")
st.write(
    "A machine learning application that predicts the "
    "likelihood of heart disease using patient health parameters."
)


# --------------------------------------------------
# LOAD DATASET AND TRAINED MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")


@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")


try:
    data = load_data()
    model = load_model()

except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e.filename}. "
        "Make sure heart.csv and heart_disease_model.pkl "
        "are in the project folder."
    )
    st.stop()


# Feature names expected by the trained model
feature_names = list(model.feature_names_in_)

if not set(feature_names).issubset(data.columns):
    st.error("Dataset columns do not match the trained model.")
    st.stop()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Heart Disease Prediction",
        "Model Performance",
        "Dataset Insights"
    ]
)


# --------------------------------------------------
# PAGE 1: HEART DISEASE PREDICTION
# --------------------------------------------------

if page == "Heart Disease Prediction":

    st.header("Patient Health Information")

    st.write(
        "Enter the patient's health parameters below "
        "to generate a prediction."
    )

    # Friendly descriptions and input configurations
    feature_config = {
        "age": {
            "label": "Age",
            "min": 1,
            "max": 120,
            "default": 54,
            "step": 1
        },
        "sex": {
            "label": "Sex (1 = Male, 0 = Female)",
            "min": 0,
            "max": 1,
            "default": 1,
            "step": 1
        },
        "cp": {
            "label": "Chest Pain Type (0–3)",
            "min": 0,
            "max": 3,
            "default": 2,
            "step": 1
        },
        "trestbps": {
            "label": "Resting Blood Pressure (mm Hg)",
            "min": 50,
            "max": 250,
            "default": 120,
            "step": 1
        },
        "chol": {
            "label": "Serum Cholesterol (mg/dl)",
            "min": 50,
            "max": 700,
            "default": 258,
            "step": 1
        },
        "fbs": {
            "label": "Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)",
            "min": 0,
            "max": 1,
            "default": 0,
            "step": 1
        },
        "restecg": {
            "label": "Resting ECG Results (0–2)",
            "min": 0,
            "max": 2,
            "default": 0,
            "step": 1
        },
        "thalach": {
            "label": "Maximum Heart Rate Achieved",
            "min": 50,
            "max": 250,
            "default": 147,
            "step": 1
        },
        "exang": {
            "label": "Exercise-Induced Angina (1 = Yes, 0 = No)",
            "min": 0,
            "max": 1,
            "default": 0,
            "step": 1
        },
        "oldpeak": {
            "label": "ST Depression (Oldpeak)",
            "min": 0.0,
            "max": 10.0,
            "default": 0.4,
            "step": 0.1
        },
        "slope": {
            "label": "Slope of Peak Exercise ST Segment (0–2)",
            "min": 0,
            "max": 2,
            "default": 1,
            "step": 1
        },
        "ca": {
            "label": "Number of Major Vessels (0–4)",
            "min": 0,
            "max": 4,
            "default": 0,
            "step": 1
        },
        "thal": {
            "label": "Thalassemia (Dataset Category)",
            "min": 0,
            "max": 3,
            "default": 3,
            "step": 1
        }
    }

    input_values = {}

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_names):

        if feature not in feature_config:
            st.error(
                f"Input configuration missing for feature: {feature}"
            )
            st.stop()

        config = feature_config[feature]

        with (col1 if index % 2 == 0 else col2):

            input_values[feature] = st.number_input(
                config["label"],
                min_value=config["min"],
                max_value=config["max"],
                value=config["default"],
                step=config["step"],
                key=f"input_{feature}"
            )

    st.divider()

    if st.button("Predict Heart Disease", type="primary"):

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                "The model predicts a positive heart disease result."
            )
        else:
            st.success(
                "The model predicts a negative heart disease result."
            )

        # Display prediction probability when available
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]

            positive_probability = probabilities[
                list(model.classes_).index(1)
            ]

            st.metric(
                "Model-estimated positive class probability",
                f"{positive_probability * 100:.2f}%"
            )

            st.progress(float(positive_probability))

        st.caption(
            "This is a machine learning prediction, not a medical "
            "diagnosis. Consult a qualified healthcare professional "
            "for medical evaluation."
        )


# --------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance")

    X = data[feature_names]
    Y = data["target"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=1,
        stratify=Y
    )

    # Predictions using the saved model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        Y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        Y_test,
        test_predictions
    )

    # Display accuracy metrics
    col1, col2 = st.columns(2)

    col1.metric(
        "Training Accuracy",
        f"{train_accuracy * 100:.2f}%"
    )

    col2.metric(
        "Testing Accuracy",
        f"{test_accuracy * 100:.2f}%"
    )

    st.divider()

    # Classification report
    st.subheader("Classification Report")

    report = classification_report(
        Y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        Y_test,
        test_predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# PAGE 3: DATASET INSIGHTS
# --------------------------------------------------

elif page == "Dataset Insights":

    st.header("Dataset Insights")

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", data.shape[0])
    col2.metric("Total Features", len(feature_names))
    col3.metric("Missing Values", int(data.isnull().sum().sum()))

    st.divider()

    # Dataset preview
    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    st.divider()

    # Statistical summary
    st.subheader("Statistical Summary")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )

    st.divider()

    # Target distribution
    st.subheader("Heart Disease Distribution")

    target_counts = data["target"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        x=target_counts.index,
        y=target_counts.values,
        ax=ax
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_xlabel("Target")
    ax.set_ylabel("Number of Records")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        data.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Feature distribution
    st.subheader("Feature Distributions")

    selected_feature = st.selectbox(
        "Select a feature",
        feature_names
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=data,
        x=selected_feature,
        hue="target",
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {selected_feature}")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Heart Disease Prediction | Logistic Regression | "
    "Built using Python, Scikit-learn and Streamlit"
)
