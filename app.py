import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import shap

# Set page configuration
st.set_page_config(
    page_title="Healthcare Analytics Portfolio",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Data Loading Cache
@st.cache_data
def load_data():
    """
    Loads and preprocesses the healthcare dataset.
    """
    data_path = "healthcare_dataset.csv"
    
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Please ensure the file exists.")
        return None

    try:
        df = pd.read_csv(data_path)
        
        # --- Preprocessing Steps from Notebook ---
        # 1. Standardize column names
        df.columns = [c.replace(" ", "_") for c in df.columns]
        
        # 2. Convert dates
        df["Date_of_Admission"] = pd.to_datetime(df["Date_of_Admission"])
        df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"])
        
        # 3. Normalize categorical strings
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].str.strip().str.title()
            
        # 4. Feature Engineering: Stay Duration
        df["Stay_Duration"] = (df["Discharge_Date"] - df["Date_of_Admission"]).dt.days
        
        # 5. Integrity Check
        invalid_mask = df["Stay_Duration"] < 0
        if invalid_mask.sum() > 0:
            # We filter out invalid rows as per notebook logic
            df = df[~invalid_mask]
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model and artifacts."""
    artifact_path = "model_artifacts.pkl"
    if not os.path.exists(artifact_path):
        return None
    return joblib.load(artifact_path)

def main():
    # Hide default sidebar
    st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

    # Load Data and Artifacts
    df = load_data()
    artifacts = load_model_artifacts()
    
    if df is None:
        return

    # Header
    st.markdown("## 🏥 Healthcare Analytics Portfolio")
    
    # Horizontal Navigation
    page = st.radio(
        "Navigation",
        ["Home", "EDA", "Prediction", "Explainability", "About"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")

    # Content Routing
    if page == "Home":
        render_home(df)
    elif page == "EDA":
        render_eda(df)
    elif page == "Prediction":
        render_prediction(df, artifacts)
    elif page == "Explainability":
        render_explainability(df, artifacts)
    elif page == "About":
        render_about()

def render_home(df):
    st.title("🏥 Healthcare Analytics & Prediction System")
    st.markdown("### Portfolio Project: Relational EHR Simulation & Cost Prediction")
    
    st.write(
        """
        Welcome to this interactive portfolio application. This tool transforms raw healthcare data 
        into actionable insights, simulates a relational EHR environment, and deploys a machine learning 
        model to predict patient billing amounts. The dataset is sourced from Kaggle and is available 
        for download at https://www.kaggle.com/datasets/prasad22/healthcare-dataset. 
        """
    )
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        st.metric("Avg Billing Amount", f"${df['Billing_Amount'].mean():,.2f}")
    with col3:
        st.metric("Avg Stay Duration", f"{df['Stay_Duration'].mean():.1f} days")
    with col4:
        st.metric("Unique Conditions", f"{df['Medical_Condition'].nunique()}")
        
    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.markdown("### Project Workflow")
    st.write(
        """
        1. **EDA**: Explore relationships between insurance, age, and medical conditions.
        2. **Prediction**: Interactive tool for real-time cost estimation using a pre-trained Random Forest.
        3. **Explainability**: Understand feature impact using SHAP values.
        """
    )

def render_eda(df):
    st.title("📊 Exploratory Data Analysis")
    
    st.markdown("### 1. Univariate Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Billing Amount Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Billing_Amount"], kde=True, ax=ax, color="skyblue")
        ax.set_title("Distribution of Billing Amount")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Medical Condition Counts")
        fig, ax = plt.subplots()
        sns.countplot(y="Medical_Condition", data=df, order=df["Medical_Condition"].value_counts().index, palette="viridis", ax=ax)
        ax.set_title("Count of Patients by Condition")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 2. Bivariate Analysis")
    
    st.subheader("Length of Stay vs Insurance Provider")
    # Interactive selection for Hue
    hue_opt = st.selectbox("Color by (Hue):", ["Admission_Type", "Gender", "Medical_Condition"], index=0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Insurance_Provider", y="Stay_Duration", hue=hue_opt, ax=ax, palette="Set2")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("### 3. Multivariate Cost Drivers")
    st.write("Correlation between features (Categorical features are Label Encoded for this view).")
    
    # Encode for correlation
    df_encoded = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    cat_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
    # Select columns of interest for correlation
    cols_interest = ["Age", "Stay_Duration", "Billing_Amount", "Medical_Condition", "Insurance_Provider", "Admission_Type"]
    # Ensure they exist in df
    cols_interest = [c for c in cols_interest if c in df_encoded.columns]
    
    corr = df_encoded[cols_interest].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

def render_prediction(df, artifacts):
    st.title("🔮 Real-time Prediction")
    
    if artifacts is None:
        st.warning("⚠️ Model artifacts not found.")
        return
        
    st.markdown("### Enter Patient Details")
    
    model = artifacts['model']
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
            gender = st.selectbox("Gender", df["Gender"].unique())
            blood_type = st.selectbox("Blood Type", df["Blood_Type"].unique())
            condition = st.selectbox("Medical Condition", df["Medical_Condition"].unique())
            
        with col2:
            insurance = st.selectbox("Insurance Provider", df["Insurance_Provider"].unique())
            admission = st.selectbox("Admission Type", df["Admission_Type"].unique())
            stay_days = st.number_input("Stay Duration (Days)", min_value=1, max_value=365, value=5, step=1)
            
        submit = st.form_submit_button("Predict Billing Amount")
        
    if submit:
        # Create input dataframe matching training data structure
        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Blood_Type": [blood_type],
            "Medical_Condition": [condition],
            "Insurance_Provider": [insurance],
            "Admission_Type": [admission],
            "Stay_Duration": [stay_days]
        })
        
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"### Estimated Billing Amount: ${prediction:,.2f}")
            
            # Context
            avg_bill = df['Billing_Amount'].mean()
            diff = prediction - avg_bill
            if diff > 0:
                st.warning(f"This is **${diff:,.2f} higher** than the average billing amount.")
            else:
                st.info(f"This is **${abs(diff):,.2f} lower** than the average billing amount.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")

def render_explainability(df, artifacts):
    st.title("🔍 Model Explainability")
    
    st.markdown("### Feature Impact (SHAP Values)")
    st.write("This plot shows how each feature contributes to pushing the prediction higher (red) or lower (blue) from the baseline. (Pre-computed image for performance)")
    
    image_path = "shap_summary.png"
    
    if os.path.exists(image_path):
        st.image(image_path, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.warning("⚠️ SHAP Summary Image (shap_summary.png) not found. Please run `python train_model.py` to generate it.")

def render_about():
    st.title("About This Project")
    
    st.markdown("""
    ### 🏥 Healthcare Analytics Portfolio
    
    This application validates data science and MLOps skills by simulating a real-world healthcare analytics workflow.
    
    #### Key Features
    - **Data Pipeline**: Cleaning and processing raw patient data (simulating EHR systems).
    - **EDA**: Visualizing distributions and relationships (e.g., Cost Drivers).
    - **Prediction**: Interactive Streamlit interface for billing prediction.
    - **Explainability**: SHAP (SHapley Additive exPlanations) for model transparency.
    
    #### Tech Stack
    - **Python**: Core logic
    - **Streamlit**: Web Framework
    - **Pandas & NumPy**: Data Manipulation
    - **Scikit-Learn**: Model Training
    - **Matplotlib & Seaborn**: Visualization
    - **SHAP**: Explainable AI
    
    #### Dataset
    Kaggle Healthcare Dataset containing patient details, medical conditions, and billing information.
    """)

if __name__ == "__main__":
    main()
