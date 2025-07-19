"""
Basic usage examples for PyInterpret library.

This script demonstrates the fundamental concepts and usage patterns
of the PyInterpret library with simple examples.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

# Import PyInterpret explainers
from pyinterpret import (
    SHAPExplainer,
    LIMEExplainer, 
    PermutationImportanceExplainer,
    PartialDependenceExplainer
)


def basic_classification_example():
    """Demonstrate basic classification interpretation."""
    print("=" * 60)
    print("BASIC CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Create synthetic classification dataset
    print("Creating synthetic classification dataset...")
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
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    print("\n" + "-" * 40)
    print("LOCAL EXPLANATIONS")
    print("-" * 40)
    
    # SHAP local explanation
    print("\n1. SHAP Local Explanation:")
    shap_explainer = SHAPExplainer(model, explainer_type='tree')
    
    # Explain a single instance
    instance = X_test.iloc[0]
    shap_result = shap_explainer.explain_instance(instance)
    
    print(f"Instance prediction: {model.predict([instance])[0]}")
    print(f"SHAP baseline: {shap_result.baseline:.3f}")
    print("\nTop 5 SHAP attributions:")
    
    # Get top attributions
    top_indices = np.argsort(np.abs(shap_result.attributions))[-5:][::-1]
    for idx in top_indices:
        feature = shap_result.feature_names[idx]
        attribution = shap_result.attributions[idx]
        value = shap_result.feature_values[idx]
        print(f"  {feature}: {attribution:+.3f} (value: {value:.3f})")
    
    # LIME local explanation
    print("\n2. LIME Local Explanation:")
    lime_explainer = LIMEExplainer(model, mode='classification')
    lime_explainer.fit(X_train)
    
    lime_result = lime_explainer.explain_instance(instance)
    
    print(f"Instance prediction: {model.predict([instance])[0]}")
    print(f"LIME baseline: {lime_result.baseline:.3f}")
    print("\nTop 5 LIME attributions:")
    
    # Get top attributions
    top_indices = np.argsort(np.abs(lime_result.attributions))[-5:][::-1]
    for idx in top_indices:
        feature = lime_result.feature_names[idx]
        attribution = lime_result.attributions[idx]
        value = lime_result.feature_values[idx]
        print(f"  {feature}: {attribution:+.3f} (value: {value:.3f})")
    
    print("\n" + "-" * 40)
    print("GLOBAL EXPLANATIONS")
    print("-" * 40)
    
    # Permutation importance
    print("\n3. Permutation Importance:")
    perm_explainer = PermutationImportanceExplainer(
        model, 
        scoring='accuracy',
        n_repeats=5,
        random_state=42
    )
    
    perm_result = perm_explainer.explain_global(X_test, y_test)
    
    print(f"Baseline accuracy: {perm_result.metadata['baseline_score']:.3f}")
    print("\nTop 5 most important features:")
    
    # Get top features
    top_indices = np.argsort(perm_result.attributions)[-5:][::-1]
    for idx in top_indices:
        feature = perm_result.feature_names[idx]
        importance = perm_result.attributions[idx]
        std = perm_result.metadata['std_importance'][idx]
        print(f"  {feature}: {importance:.3f} ± {std:.3f}")
    
    # Partial dependence
    print("\n4. Partial Dependence (Top Feature):")
    pd_explainer = PartialDependenceExplainer(model, grid_resolution=20)
    
    # Get the most important feature
    most_important_feature = top_indices[0]
    pd_result = pd_explainer.explain_global(X_test, features=most_important_feature)
    
    print(f"Partial dependence for: {pd_result.feature_names[0]}")
    grid = pd_result.metadata['grid']
    values = pd_result.metadata['partial_dependence_values']
    
    print("Grid points and corresponding predictions:")
    for i in range(0, len(grid), len(grid)//5):  # Show 5 points
        print(f"  {grid[i]:.3f} → {values[i]:.3f}")


def basic_regression_example():
    """Demonstrate basic regression interpretation."""
    print("\n\n" + "=" * 60)
    print("BASIC REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Create synthetic regression dataset
    print("Creating synthetic regression dataset...")
    X, y = make_regression(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Train a Linear Regression model
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R²: {train_score:.3f}")
    print(f"Test R²: {test_score:.3f}")
    
    print("\n" + "-" * 40)
    print("LOCAL EXPLANATIONS")
    print("-" * 40)
    
    # SHAP local explanation
    print("\n1. SHAP Local Explanation:")
    shap_explainer = SHAPExplainer(model, explainer_type='linear', background_data=X_train.values[:100])
    
    # Explain a single instance
    instance = X_test.iloc[0]
    shap_result = shap_explainer.explain_instance(instance)
    
    print(f"Instance prediction: {model.predict([instance])[0]:.3f}")
    print(f"SHAP baseline: {shap_result.baseline:.3f}")
    print("\nTop 5 SHAP attributions:")
    
    # Get top attributions
    top_indices = np.argsort(np.abs(shap_result.attributions))[-5:][::-1]
    for idx in top_indices:
        feature = shap_result.feature_names[idx]
        attribution = shap_result.attributions[idx]
        value = shap_result.feature_values[idx]
        print(f"  {feature}: {attribution:+.3f} (value: {value:.3f})")
    
    print("\n" + "-" * 40)
    print("GLOBAL EXPLANATIONS")
    print("-" * 40)
    
    # Permutation importance
    print("\n2. Permutation Importance:")
    perm_explainer = PermutationImportanceExplainer(
        model,
        scoring='neg_mse',
        n_repeats=5,
        random_state=42
    )
    
    perm_result = perm_explainer.explain_global(X_test, y_test)
    
    print(f"Baseline neg_mse: {perm_result.metadata['baseline_score']:.3f}")
    print("\nTop 5 most important features:")
    
    # Get top features
    top_indices = np.argsort(perm_result.attributions)[-5:][::-1]
    for idx in top_indices:
        feature = perm_result.feature_names[idx]
        importance = perm_result.attributions[idx]
        std = perm_result.metadata['std_importance'][idx]
        print(f"  {feature}: {importance:.3f} ± {std:.3f}")


def compare_explainers_example():
    """Compare different explainer results on the same instance."""
    print("\n\n" + "=" * 60)
    print("EXPLAINER COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Create dataset
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_df, y)
    
    # Select instance to explain
    instance = X_df.iloc[0]
    print(f"Explaining instance: {model.predict([instance])[0]}")
    print(f"Instance values: {instance.values}")
    
    print("\n" + "-" * 40)
    print("ATTRIBUTION COMPARISON")
    print("-" * 40)
    
    # SHAP explanation
    print("\nSHAP Attributions:")
    shap_explainer = SHAPExplainer(model, explainer_type='tree')
    shap_result = shap_explainer.explain_instance(instance)
    
    for i, (feature, attribution) in enumerate(zip(shap_result.feature_names, shap_result.attributions)):
        print(f"  {feature}: {attribution:+.3f}")
    
    # LIME explanation
    print("\nLIME Attributions:")
    lime_explainer = LIMEExplainer(model, mode='classification')
    lime_explainer.fit(X_df)
    lime_result = lime_explainer.explain_instance(instance)
    
    for i, (feature, attribution) in enumerate(zip(lime_result.feature_names, lime_result.attributions)):
        print(f"  {feature}: {attribution:+.3f}")
    
    # Correlation between methods
    print("\nCorrelation between methods:")
    correlation = np.corrcoef(shap_result.attributions, lime_result.attributions)[0, 1]
    print(f"SHAP vs LIME correlation: {correlation:.3f}")


if __name__ == "__main__":
    # Run all examples
    try:
        basic_classification_example()
        basic_regression_example()
        compare_explainers_example()
        
        print("\n\n" + "=" * 60)
        print("EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nTo visualize results, consider using:")
        print("- explainer.plot_importance() for global methods")
        print("- pyinterpret.utils.visualization functions for custom plots")
        print("- Jupyter notebooks for interactive exploration")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install scikit-learn numpy pandas")
        print("pip install shap lime  # for SHAP and LIME functionality")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please check your PyInterpret installation.")
