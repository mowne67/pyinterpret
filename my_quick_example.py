#!/usr/bin/env python3
"""
Quick example of using PyInterpret in Replit
"""

from pyinterpret import SHAPExplainer, LIMEExplainer, PermutationImportanceExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

print("🚀 Quick PyInterpret Demo in Replit")
print("=" * 40)

# Create sample data
print("Creating sample dataset...")
X, y = make_classification(
    n_samples=500, 
    n_features=6, 
    n_informative=4, 
    random_state=42
)

# Convert to DataFrame with feature names
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print(f"Model accuracy: {model.score(X_test, y_test):.3f}")

# Get one instance to explain
instance = X_test.iloc[0]
print(f"\nExplaining instance with prediction: {model.predict([instance])[0]}")

print("\n📊 LOCAL EXPLANATIONS")
print("-" * 30)

# SHAP explanation
print("SHAP explanation:")
shap_explainer = SHAPExplainer(model, explainer_type='tree')
shap_result = shap_explainer.explain_instance(instance)

# Show top 3 features
top_indices = np.argsort(np.abs(shap_result.attributions))[-3:][::-1]
for idx in top_indices:
    feature = shap_result.feature_names[idx]
    importance = shap_result.attributions[idx]
    value = shap_result.feature_values[idx]
    print(f"  {feature}: {importance:+.3f} (value: {value:.2f})")

print("\n🌍 GLOBAL EXPLANATIONS")
print("-" * 30)

# Permutation importance
print("Feature importance:")
perm_explainer = PermutationImportanceExplainer(model, scoring='accuracy', n_repeats=3)
perm_result = perm_explainer.explain_global(X_test, y_test)

# Show top 3 features
top_indices = np.argsort(perm_result.attributions)[-3:][::-1]
for idx in top_indices:
    feature = perm_result.feature_names[idx]
    importance = perm_result.attributions[idx]
    print(f"  {feature}: {importance:.3f}")

print("\n✅ PyInterpret demo complete!")
print("Try modifying this script or run 'python examples/basic_usage.py' for more examples.")