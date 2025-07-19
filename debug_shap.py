"""Debug SHAP explainer to understand the issue."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    import shap
    print("SHAP version:", shap.__version__)
    
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model trained successfully")
    print("X_train shape:", X_train.shape)
    print("Feature names:", X_train.columns.tolist())
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    print("SHAP TreeExplainer created")
    
    # Get single instance
    instance = X_test.iloc[0]
    print("Instance shape:", instance.shape)
    print("Instance type:", type(instance))
    
    instance_array = instance.values.reshape(1, -1)
    print("Instance array shape:", instance_array.shape)
    
    # Try old API
    try:
        shap_values_old = explainer.shap_values(instance_array)
        print("Old API - shap_values type:", type(shap_values_old))
        print("Old API - shap_values shape:", np.array(shap_values_old).shape if isinstance(shap_values_old, list) else shap_values_old.shape)
        if isinstance(shap_values_old, list):
            for i, val in enumerate(shap_values_old):
                print(f"  Class {i} shape: {val.shape}")
        print("Old API - expected value:", explainer.expected_value)
    except Exception as e:
        print("Old API failed:", e)
    
    # Try new API
    try:
        explanation = explainer(instance_array)
        print("New API - explanation type:", type(explanation))
        print("New API - explanation.values shape:", explanation.values.shape)
        print("New API - expected value:", explanation.base_values)
    except Exception as e:
        print("New API failed:", e)
    
except ImportError:
    print("SHAP not available")
except Exception as e:
    print("Error:", e)