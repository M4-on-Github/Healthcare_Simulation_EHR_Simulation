# 🏥 Healthcare Analytics & EHR Simulation

An interactive MLOps portfolio project demonstrating Exploratory Data Analysis (EDA), predictive modeling, and model explainability using a healthcare dataset.

## 🚀 Overview

This project simulates a relational Electronic Health Record (EHR) environment and provides an interactive dashboard to:
- Explore patient demographics and cost drivers.
- Predict patient billing amounts using a Random Forest Regressor.
- Visualize model decisions using SHAP values for transparency.

**Live App Features:**
- **Home**: Project mission and key metrics.
- **EDA**: Univariate and bivariate analysis of patient data.
- **Prediction**: Real-time billing cost estimation.
- **Explainability**: Static SHAP summary plots for instant insight.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Visualization**: [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/)
- **Explainable AI**: [SHAP](https://shap.readthedocs.io/)

## 📋 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/M4-on-Github/Healthcare_Simulation_EHR_Simulation.git
   cd Healthcare_Simulation_EHR_Simulation
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate artifacts (Optional):**
   Pre-trained model artifacts (`model_artifacts.pkl`) and SHAP images (`shap_summary.png`) are included in the repository for **instant hosting**. To re-train the model:
   ```bash
   python train_model.py
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset

The project uses the [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) from Kaggle. It contains synthetic data representing patient names, ages, medical conditions, billing amounts, and more.

## 🧠 Model & Explainability

- **Model**: Random Forest Regressor optimized for billing amount prediction.
- **Pipeline**: Includes categorical encoding (OneHotEncoder) and numeric scaling.
- **SHAP**: Baseline explanations are pre-calculated and stored as images to ensure the application remains fast and responsive.

---
*Created as part of a Data Science & MLOps portfolio.*
