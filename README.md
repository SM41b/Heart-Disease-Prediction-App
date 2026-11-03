# ❤️ Heart Disease Prediction Using Machine Learning

A machine learning project that predicts the likelihood of heart disease based on patient health parameters using **Logistic Regression**. The project includes data analysis, model training, evaluation, and an interactive web application built with Streamlit.

## 🚀 Live Demo

**Try the application here:** [Heart Disease Prediction App](YOUR_DEPLOYED_APP_URL)

Explore the application by entering patient health parameters and viewing the model's prediction.

---

## 📌 Project Overview

Heart disease is one of the major health concerns worldwide. This project uses machine learning to identify patterns in patient health data and predict the likelihood of heart disease.

A **Logistic Regression** classification model is trained using patient health parameters. The trained model is integrated into a Streamlit web application, allowing users to enter health information and receive predictions.

### Objectives

- Perform exploratory data analysis on the heart disease dataset.
- Understand relationships between patient health parameters.
- Preprocess and standardize input features.
- Train a Logistic Regression classification model.
- Evaluate model performance using classification metrics.
- Deploy the trained model through an interactive Streamlit application.

---

## 📂 Dataset Information

The project uses the `heart.csv` dataset containing patient health records.

The dataset includes the following 13 input features:

| Feature | Description |
|---|---|
| age | Age of the patient |
| sex | Sex (1 = Male, 0 = Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia category |

**Target Variable:**

- `0` – No heart disease
- `1` – Heart disease

---

## 🔍 Exploratory Data Analysis

The following analysis was performed:

- Dataset dimensions and information.
- Missing value detection.
- Statistical summary of numerical features.
- Distribution of target classes.
- Feature correlation analysis.
- Relationship between patient features and heart disease.

Visualizations include:

- Target distribution chart.
- Correlation heatmap.
- Feature distribution histograms.

---

## ⚙️ Machine Learning Workflow

### 1. Feature and Target Separation

The independent variables are separated from the dependent target variable.

### 2. Train-Test Split

The dataset is divided into training and testing sets:

- Training data: 80%
- Testing data: 20%
- Random state: 1
- Stratified sampling to preserve target class proportions.

### 3. Feature Scaling

`StandardScaler` is used to standardize the input features before model training.

### 4. Model Training

Logistic Regression is used as the classification algorithm.

The preprocessing and classification steps are combined into a Scikit-learn Pipeline.

```python
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

model.fit(X_train, Y_train)
```

### 5. Model Evaluation

The model is evaluated using:

- Training Accuracy
- Testing Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Actual evaluation results can be viewed in the Streamlit application.

---

## 🖥️ Streamlit Application Features

The application provides three main sections:

### 1. Heart Disease Prediction
- Enter patient health parameters.
- Generate predictions using the trained Logistic Regression model.
- View the predicted class.
- View the model-estimated positive class probability, when available.

### 2. Model Performance
- View training and testing accuracy.
- Examine the classification report.
- Visualize the confusion matrix.

### 3. Dataset Insights
- Explore the dataset.
- View statistical summaries.
- Analyze heart disease distribution.
- Explore feature correlations.
- Visualize individual feature distributions.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and evaluation |
| Logistic Regression | Classification algorithm |
| StandardScaler | Feature standardization |
| Pipeline | Combining preprocessing and model training |
| Joblib | Saving and loading the trained model |
| Streamlit | Web application development |

---

## 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── Heart Disease Prediction.py
├── app.py
├── heart.csv
├── heart_disease_model.pkl
├── requirements.txt
└── README.md

```

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SM41b/Heart-Disease-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Heart-Disease-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
.\venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the Machine Learning Model

```bash
python "Heart Disease Prediction.py"
```

This generates the trained model file:

`heart_disease_model.pkl`

### 7. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default browser.



