import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap

def load_and_prep_data(data_path="healthcare_dataset.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
        
    df = pd.read_csv(data_path)
    
    # Preprocessing
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["Date_of_Admission"] = pd.to_datetime(df["Date_of_Admission"])
    df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"])
    
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].str.strip().str.title()
        
    df["Stay_Duration"] = (df["Discharge_Date"] - df["Date_of_Admission"]).dt.days
    df = df[df["Stay_Duration"] >= 0]
    
    return df

def train_and_save():
    print("Loading data...")
    df = load_and_prep_data()
    
    # Drop columns
    drop_cols = ["Name", "Date_of_Admission", "Discharge_Date", "Doctor", "Hospital", "Room_Number", "Medication", "Test_Results"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    model_df = df.drop(columns=drop_cols)
    
    X = model_df.drop(columns=["Billing_Amount"])
    y = model_df["Billing_Amount"]
    
    print("Training model...")
    # Pipeline
    cat_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    # Train
    model.fit(X, y)
    
    print("Generating SHAP images...")
    # Transform X for SHAP
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    
    # Sample data
    shap_sample = X_transformed[:500] 
    
    # Get feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    remainder_cols = [c for c in X.columns if c not in cat_features]
    all_feature_names = list(cat_feature_names) + remainder_cols
    
    # Calculate SHAP
    rf = model.named_steps['regressor']
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(shap_sample)
    
    # Plot and Save SHAP Summary
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, shap_sample, feature_names=all_feature_names, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    print("Saved shap_summary.png")
    
    # Save artifacts (lighter now without shap data)
    artifacts = {
        'model': model,
        'feature_names': all_feature_names,
        # 'shap_sample': shap_sample, # Removed to save space/time
        'data_columns': X.columns.tolist(),
        'cat_features': list(cat_features)
    }
    
    joblib.dump(artifacts, 'model_artifacts.pkl', compress=3)
    print("Artifacts saved to model_artifacts.pkl (compressed)")

if __name__ == "__main__":
    train_and_save()
