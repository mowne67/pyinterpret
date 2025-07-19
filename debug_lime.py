"""Debug LIME explainer to understand the issue."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from pyinterpret import LIMEExplainer
    
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
    
    print("Creating LIME explainer...")
    lime_explainer = LIMEExplainer(model, mode='classification')
    print("LIME explainer created successfully")
    print("Mode attribute before fit:", hasattr(lime_explainer, 'mode'), getattr(lime_explainer, 'mode', 'NOT_FOUND'))
    
    print("Calling fit...")
    lime_explainer.fit(X_train)
    print("Fit completed")
    print("Mode attribute after fit:", hasattr(lime_explainer, 'mode'), getattr(lime_explainer, 'mode', 'NOT_FOUND'))
    
    # Get single instance
    instance = X_test.iloc[0]
    print("Instance shape:", instance.shape)
    
    print("Calling explain_instance...")
    lime_result = lime_explainer.explain_instance(instance)
    print("Success!")
    
except Exception as e:
    import traceback
    print("Error:", e)
    print("Traceback:")
    traceback.print_exc()