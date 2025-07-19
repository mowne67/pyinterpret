"""
Comprehensive tabular data interpretation example.

This script demonstrates advanced usage of PyInterpret with real-world
tabular data scenarios, including preprocessing and comprehensive analysis.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import PyInterpret components
from pyinterpret import (
    SHAPExplainer,
    LIMEExplainer,
    PermutationImportanceExplainer,
    PartialDependenceExplainer,
    TabularData
)


def breast_cancer_analysis():
    """Comprehensive breast cancer dataset analysis."""
    print("=" * 70)
    print("BREAST CANCER CLASSIFICATION ANALYSIS")
    print("=" * 70)
    
    # Load breast cancer dataset
    print("Loading breast cancer dataset...")
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = cancer.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target classes: {np.unique(y)} (0=malignant, 1=benign)")
    
    # Create TabularData object
    tabular_data = TabularData(X, y, target_name='diagnosis')
    
    print("\nDataset summary:")
    print(tabular_data)
    
    # Split data
    train_data, test_data = tabular_data.train_test_split(test_size=0.2, random_state=42, stratify=True)
    X_train, y_train = train_data.to_arrays()
    X_test, y_test = test_data.to_arrays()
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train multiple models for comparison
    print("\n" + "-" * 50)
    print("TRAINING MODELS")
    print("-" * 50)
    
    models = {}
    
    # Random Forest
    print("\n1. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    print(f"   Training accuracy: {rf.score(X_train, y_train):.3f}")
    print(f"   Test accuracy: {rf.score(X_test, y_test):.3f}")
    
    # Logistic Regression (with scaling)
    print("\n2. Training Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    models['LogisticRegression'] = (lr, scaler)
    print(f"   Training accuracy: {lr.score(X_train_scaled, y_train):.3f}")
    print(f"   Test accuracy: {lr.score(X_test_scaled, y_test):.3f}")
    
    # Detailed analysis with Random Forest
    print("\n" + "-" * 50)
    print("DETAILED INTERPRETATION (Random Forest)")
    print("-" * 50)
    
    model = rf
    
    # Global feature importance
    print("\n1. PERMUTATION IMPORTANCE ANALYSIS:")
    perm_explainer = PermutationImportanceExplainer(
        model,
        scoring='accuracy',
        n_repeats=10,
        random_state=42
    )
    
    perm_result = perm_explainer.explain_global(X_test, y_test)
    ranking = perm_explainer.get_feature_ranking(X_test, y_test, top_k=10)
    
    print(f"Baseline accuracy: {perm_result.metadata['baseline_score']:.3f}")
    print("\nTop 10 most important features:")
    for item in ranking['ranking']:
        print(f"  {item['rank']:2d}. {item['feature'][:20]:<20} "
              f"Importance: {item['importance']:+.3f} ± {item['std']:.3f}")
    
    # Partial dependence for top features
    print("\n2. PARTIAL DEPENDENCE ANALYSIS:")
    pd_explainer = PartialDependenceExplainer(model, grid_resolution=30)
    
    top_features = [item['feature'] for item in ranking['ranking'][:3]]
    print(f"Analyzing partial dependence for top 3 features: {top_features}")
    
    for feature in top_features:
        pd_result = pd_explainer.explain_global(X_test, features=feature)
        grid = pd_result.metadata['grid']
        values = pd_result.attributions
        
        print(f"\n{feature}:")
        print(f"  Range: [{grid.min():.3f}, {grid.max():.3f}]")
        print(f"  PD Range: [{values.min():.3f}, {values.max():.3f}]")
        print(f"  Effect magnitude: {values.max() - values.min():.3f}")
    
    # Local explanations
    print("\n3. LOCAL EXPLANATION ANALYSIS:")
    
    # Select interesting instances
    # Get one malignant (0) and one benign (1) prediction
    malignant_idx = np.where(y_test == 0)[0][0]
    benign_idx = np.where(y_test == 1)[0][0]
    
    instances = [
        (malignant_idx, "Malignant case"),
        (benign_idx, "Benign case")
    ]
    
    # SHAP explanations
    print("\nSHAP Local Explanations:")
    shap_explainer = SHAPExplainer(model, explainer_type='tree')
    
    for idx, description in instances:
        instance = test_data.to_dataframe().iloc[idx]
        prediction = model.predict_proba([instance.drop('diagnosis')])[0]
        
        print(f"\n{description} (Index {idx}):")
        print(f"  True label: {'Malignant' if y_test[idx] == 0 else 'Benign'}")
        print(f"  Predicted probabilities: Malignant={prediction[0]:.3f}, Benign={prediction[1]:.3f}")
        
        shap_result = shap_explainer.explain_instance(instance.drop('diagnosis'))
        
        # Show top contributing features
        top_indices = np.argsort(np.abs(shap_result.attributions))[-5:][::-1]
        print("  Top 5 SHAP contributions:")
        for i in top_indices:
            feature = shap_result.feature_names[i]
            attribution = shap_result.attributions[i]
            value = shap_result.feature_values[i]
            print(f"    {feature[:20]:<20}: {attribution:+.3f} (value: {value:.3f})")
    
    # LIME explanations
    print("\nLIME Local Explanations:")
    lime_explainer = LIMEExplainer(model, mode='classification')
    lime_explainer.fit(X_train)
    
    for idx, description in instances:
        instance = test_data.to_dataframe().iloc[idx]
        
        print(f"\n{description} (Index {idx}):")
        lime_result = lime_explainer.explain_instance(instance.drop('diagnosis'))
        
        # Show top contributing features
        top_indices = np.argsort(np.abs(lime_result.attributions))[-5:][::-1]
        print("  Top 5 LIME contributions:")
        for i in top_indices:
            feature = lime_result.feature_names[i]
            attribution = lime_result.attributions[i]
            value = lime_result.feature_values[i] if lime_result.feature_values is not None else "N/A"
            print(f"    {feature[:20]:<20}: {attribution:+.3f} (value: {value})")
    
    # Model comparison
    print("\n4. MODEL COMPARISON:")
    compare_models(models, X_test, y_test, X_test_scaled, test_data.feature_names)


def boston_housing_analysis():
    """Comprehensive Boston housing regression analysis."""
    print("\n\n" + "=" * 70)
    print("BOSTON HOUSING REGRESSION ANALYSIS")
    print("=" * 70)
    
    # Load Boston housing dataset
    try:
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = boston.target
    except ImportError:
        print("Boston housing dataset is not available in this sklearn version.")
        print("Creating synthetic regression data instead...")
        from sklearn.datasets import make_regression
        X_syn, y_syn = make_regression(n_samples=506, n_features=13, noise=0.1, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X_syn.shape[1])]
        X = pd.DataFrame(X_syn, columns=feature_names)
        y = y_syn
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Create TabularData object
    tabular_data = TabularData(X, y, target_name='price')
    
    # Split data
    train_data, test_data = tabular_data.train_test_split(test_size=0.2, random_state=42)
    X_train, y_train = train_data.to_arrays()
    X_test, y_test = test_data.to_arrays()
    
    # Train Gradient Boosting model
    print("\n" + "-" * 50)
    print("TRAINING GRADIENT BOOSTING REGRESSOR")
    print("-" * 50)
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R²: {train_score:.3f}")
    print(f"Test R²: {test_score:.3f}")
    
    # Global importance
    print("\n" + "-" * 50)
    print("GLOBAL FEATURE IMPORTANCE")
    print("-" * 50)
    
    perm_explainer = PermutationImportanceExplainer(
        model,
        scoring='r2',
        n_repeats=10,
        random_state=42
    )
    
    perm_result = perm_explainer.explain_global(X_test, y_test)
    ranking = perm_explainer.get_feature_ranking(X_test, y_test, top_k=len(X.columns))
    
    print(f"Baseline R²: {perm_result.metadata['baseline_score']:.3f}")
    print("\nFeature importance ranking:")
    for item in ranking['ranking']:
        print(f"  {item['rank']:2d}. {item['feature']:<15} "
              f"Importance: {item['importance']:+.3f} ± {item['std']:.3f}")
    
    # Partial dependence analysis
    print("\n" + "-" * 50)
    print("PARTIAL DEPENDENCE ANALYSIS")
    print("-" * 50)
    
    pd_explainer = PartialDependenceExplainer(model, grid_resolution=25)
    
    # Analyze top 3 features
    top_features = [item['feature'] for item in ranking['ranking'][:3]]
    
    for feature in top_features:
        print(f"\nPartial dependence for {feature}:")
        pd_result = pd_explainer.explain_global(X_test, features=feature)
        
        grid = pd_result.metadata['grid']
        values = pd_result.attributions
        
        print(f"  Feature range: [{grid.min():.3f}, {grid.max():.3f}]")
        print(f"  PD range: [{values.min():.3f}, {values.max():.3f}]")
        
        # Find the feature value that maximizes/minimizes prediction
        max_idx = np.argmax(values)
        min_idx = np.argmin(values)
        print(f"  Max prediction at {feature}={grid[max_idx]:.3f} → {values[max_idx]:.3f}")
        print(f"  Min prediction at {feature}={grid[min_idx]:.3f} → {values[min_idx]:.3f}")
    
    # 2D partial dependence
    if len(top_features) >= 2:
        print(f"\n2D Partial dependence for {top_features[0]} vs {top_features[1]}:")
        pd_2d_result = pd_explainer.explain_global(X_test, features=top_features[:2])
        
        values_2d = pd_2d_result.attributions
        print(f"  2D PD range: [{values_2d.min():.3f}, {values_2d.max():.3f}]")
        print(f"  2D effect magnitude: {values_2d.max() - values_2d.min():.3f}")
    
    # Local explanations
    print("\n" + "-" * 50)
    print("LOCAL EXPLANATIONS")
    print("-" * 50)
    
    # Select instances with high and low predictions
    predictions = model.predict(X_test)
    high_idx = np.argmax(predictions)
    low_idx = np.argmin(predictions)
    median_idx = np.argsort(predictions)[len(predictions)//2]
    
    instances = [
        (high_idx, "High prediction"),
        (low_idx, "Low prediction"), 
        (median_idx, "Median prediction")
    ]
    
    shap_explainer = SHAPExplainer(model, explainer_type='tree')
    
    for idx, description in instances:
        instance = test_data.to_dataframe().iloc[idx]
        prediction = model.predict([instance.drop('price')])[0]
        true_value = y_test[idx]
        
        print(f"\n{description} (Index {idx}):")
        print(f"  True value: {true_value:.2f}")
        print(f"  Predicted value: {prediction:.2f}")
        print(f"  Error: {abs(true_value - prediction):.2f}")
        
        shap_result = shap_explainer.explain_instance(instance.drop('price'))
        
        # Show top contributing features
        top_indices = np.argsort(np.abs(shap_result.attributions))[-5:][::-1]
        print("  Top 5 SHAP contributions:")
        for i in top_indices:
            feature = shap_result.feature_names[i]
            attribution = shap_result.attributions[i]
            value = shap_result.feature_values[i]
            print(f"    {feature:<15}: {attribution:+.3f} (value: {value:.3f})")


def compare_models(models, X_test, y_test, X_test_scaled, feature_names):
    """Compare permutation importance across different models."""
    print("\nComparing permutation importance across models:")
    
    # Random Forest
    rf = models['RandomForest']
    rf_explainer = PermutationImportanceExplainer(rf, scoring='accuracy', n_repeats=5, random_state=42)
    rf_result = rf_explainer.explain_global(X_test, y_test)
    
    # Logistic Regression (scaled data)
    lr, scaler = models['LogisticRegression']
    lr_explainer = PermutationImportanceExplainer(lr, scoring='accuracy', n_repeats=5, random_state=42)
    lr_result = lr_explainer.explain_global(X_test_scaled, y_test)
    
    # Compare top 5 features for each model
    print("\nTop 5 features by model:")
    
    # Random Forest
    rf_top = np.argsort(rf_result.attributions)[-5:][::-1]
    print("\nRandom Forest:")
    for i, idx in enumerate(rf_top):
        feature = rf_result.feature_names[idx]
        importance = rf_result.attributions[idx]
        print(f"  {i+1}. {feature[:20]:<20}: {importance:.3f}")
    
    # Logistic Regression
    lr_top = np.argsort(lr_result.attributions)[-5:][::-1]
    print("\nLogistic Regression:")
    for i, idx in enumerate(lr_top):
        feature = lr_result.feature_names[idx]
        importance = lr_result.attributions[idx]
        print(f"  {i+1}. {feature[:20]:<20}: {importance:.3f}")
    
    # Feature ranking correlation
    correlation = np.corrcoef(rf_result.attributions, lr_result.attributions)[0, 1]
    print(f"\nFeature importance correlation between models: {correlation:.3f}")


def demonstrate_tabular_data_utilities():
    """Demonstrate TabularData utility functions."""
    print("\n\n" + "=" * 70)
    print("TABULAR DATA UTILITIES DEMONSTRATION")
    print("=" * 70)
    
    # Create synthetic data with mixed types
    np.random.seed(42)
    n_samples = 200
    
    # Numerical features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.lognormal(10, 1, n_samples)
    score = np.random.normal(0, 1, n_samples)
    
    # Categorical features (as integers)
    category = np.random.choice([0, 1, 2], n_samples)
    region = np.random.choice([0, 1, 2, 3], n_samples)
    
    # Create target
    y = (age * 0.01 + income * 0.00001 + score * 0.5 + 
         category * 0.3 + region * 0.2 + 
         np.random.normal(0, 0.1, n_samples))
    
    # Create DataFrame
    X = pd.DataFrame({
        'age': age,
        'income': income, 
        'score': score,
        'category': category,
        'region': region
    })
    
    print("Created synthetic mixed-type dataset:")
    print(X.info())
    print("\nDataset head:")
    print(X.head())
    
    # Create TabularData object
    tabular_data = TabularData(
        X, y,
        categorical_features=['category', 'region'],
        target_name='outcome'
    )
    
    print(f"\nTabularData object: {tabular_data}")
    
    # Feature information
    print("\nFeature information:")
    feature_info = tabular_data.get_feature_info()
    print(feature_info)
    
    # Descriptive statistics
    print("\nDescriptive statistics:")
    print(tabular_data.describe())
    
    # Preprocessing
    print("\nPreprocessing data...")
    preprocessed_data = tabular_data.preprocess(
        scale_features=True,
        scaling_method='standard',
        encode_categorical=True
    )
    
    print("After preprocessing:")
    print(preprocessed_data.describe())
    
    # Sampling
    print("\nSampling 20% of data...")
    sample_data = tabular_data.sample(frac=0.2, random_state=42)
    print(f"Sample size: {len(sample_data)}")
    
    # Train/test split
    print("\nTrain/test split...")
    train_data, test_data = tabular_data.train_test_split(test_size=0.3, random_state=42)
    print(f"Training data: {len(train_data)} samples")
    print(f"Test data: {len(test_data)} samples")


if __name__ == "__main__":
    try:
        # Run comprehensive examples
        breast_cancer_analysis()
        boston_housing_analysis()
        demonstrate_tabular_data_utilities()
        
        print("\n\n" + "=" * 70)
        print("ALL TABULAR EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("- Different explainers can provide complementary insights")
        print("- Permutation importance shows global feature relevance")
        print("- Partial dependence reveals feature-prediction relationships")
        print("- Local explanations help understand individual predictions")
        print("- TabularData class simplifies data preprocessing and management")
        
        print("\nNext steps:")
        print("- Try these methods on your own datasets")
        print("- Experiment with different model types")
        print("- Use visualization functions for better insights")
        print("- Consider ensemble approaches for robust explanations")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install scikit-learn numpy pandas")
        print("pip install shap lime  # for SHAP and LIME functionality")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
