"""
SHAP (SHapley Additive exPlanations) explainer implementation.

This module provides a unified interface to SHAP explainers for different
model types while maintaining consistency with the PyInterpret API.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from pyinterpret.core.base import LocalExplainer, ExplanationResult
from pyinterpret.core.exceptions import ModelError, ValidationError, ExplainerError
from pyinterpret.utils.validation import validate_data


class SHAPExplainer(LocalExplainer):
    """
    SHAP-based explainer for generating Shapley value attributions.
    
    This explainer automatically selects the appropriate SHAP explainer type
    based on the model provided and offers a consistent interface for
    generating explanations.
    """
    
    def __init__(
        self, 
        model: Any, 
        explainer_type: str = 'auto',
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        **kwargs
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The machine learning model to explain
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel', 'deep')
            background_data: Background dataset for explainer initialization
            **kwargs: Additional parameters for SHAP explainer
        """
        if not SHAP_AVAILABLE:
            raise ExplainerError(
                "SHAP library is not available. Please install it with: pip install shap",
                explainer="SHAPExplainer"
            )
        
        super().__init__(model, **kwargs)
        
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.shap_explainer = None
        
        # Initialize SHAP explainer
        self._init_shap_explainer()
    
    def _validate_model(self) -> None:
        """Validate that the model is compatible with SHAP."""
        # Check if model has predict method
        if not hasattr(self.model, 'predict'):
            raise ModelError(
                "Model must have a 'predict' method",
                model_type=type(self.model).__name__,
                required_methods=['predict']
            )
    
    def _init_shap_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer based on model type."""
        if self.explainer_type == 'auto':
            self._auto_select_explainer()
        else:
            self._create_explainer(self.explainer_type)
    
    def _auto_select_explainer(self) -> None:
        """Automatically select the best SHAP explainer for the model."""
        model_name = type(self.model).__name__.lower()
        
        # Tree-based models
        tree_models = [
            'randomforestregressor', 'randomforestclassifier',
            'xgbregressor', 'xgbclassifier', 'xgboost',
            'lgbmregressor', 'lgbmclassifier', 'lightgbm',
            'decisiontreeregressor', 'decisiontreeclassifier',
            'extratreesregressor', 'extratreesclassifier',
            'gradientboostingregressor', 'gradientboostingclassifier'
        ]
        
        # Linear models
        linear_models = [
            'linearregression', 'logisticregression', 'ridge', 'lasso',
            'elasticnet', 'sgdregressor', 'sgdclassifier'
        ]
        
        if any(tree_model in model_name for tree_model in tree_models):
            self._create_explainer('tree')
        elif any(linear_model in model_name for linear_model in linear_models):
            self._create_explainer('linear')
        else:
            # Default to kernel explainer for unknown models
            self._create_explainer('kernel')
    
    def _create_explainer(self, explainer_type: str) -> None:
        """Create the specified SHAP explainer."""
        try:
            if explainer_type == 'tree':
                self.shap_explainer = shap.TreeExplainer(self.model)
            
            elif explainer_type == 'linear':
                # LinearExplainer requires a masker as a required argument
                if self.background_data is not None:
                    masker = shap.maskers.Independent(self.background_data)
                else:
                    # No background data: use a zero vector sized to the model's
                    # actual number of input features as the masker background.
                    n_features = getattr(self.model, 'n_features_in_', None)
                    if n_features is None:
                        coef = getattr(self.model, 'coef_', None)
                        n_features = coef.shape[-1] if coef is not None else 1
                    masker = shap.maskers.Independent(np.zeros((1, n_features)))
                
                self.shap_explainer = shap.LinearExplainer(
                    self.model, 
                    masker,
                    **self.config
                )
            
            elif explainer_type == 'kernel':
                if self.background_data is None:
                    raise ExplainerError(
                        "Background data is required for KernelExplainer",
                        explainer="SHAPExplainer",
                        method="kernel"
                    )
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict,
                    self.background_data,
                    **self.config
                )
            
            elif explainer_type == 'deep':
                self.shap_explainer = shap.DeepExplainer(
                    self.model,
                    self.background_data,
                    **self.config
                )
            
            else:
                raise ExplainerError(
                    f"Unsupported explainer type: {explainer_type}",
                    explainer="SHAPExplainer"
                )
                
        except Exception as e:
            raise ExplainerError(
                f"Failed to initialize SHAP {explainer_type} explainer: {str(e)}",
                explainer="SHAPExplainer",
                method=explainer_type
            )
    
    def explain_instance(
        self, 
        instance: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single instance using SHAP.
        
        Args:
            instance: Single instance to explain
            **kwargs: Additional parameters for SHAP explanation
            
        Returns:
            ExplanationResult containing SHAP values and metadata
        """
        # Validate and prepare instance
        instance = validate_data(instance, expected_dims=1)
        
        if isinstance(instance, pd.Series):
            feature_names = instance.index.tolist()
            feature_values = instance.values
            instance_array = instance.values.reshape(1, -1)
        else:
            feature_names = [f"feature_{i}" for i in range(len(instance))]
            feature_values = instance
            instance_array = instance.reshape(1, -1)
        
        try:
            # Generate SHAP values using the older API (more reliable for tree explainers)
            shap_values = self.shap_explainer.shap_values(instance_array, **kwargs)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class case - use the positive class (class 1) for binary classification
                if len(shap_values) > 1:
                    # For binary classification, use positive class
                    attributions = shap_values[1][0]
                else:
                    attributions = shap_values[0][0]
            elif len(shap_values.shape) == 3:
                # Shape is (n_instances, n_features, n_classes)
                # For binary classification, use positive class (class 1)
                if shap_values.shape[2] > 1:
                    attributions = shap_values[0, :, 1]  # First instance, all features, positive class
                else:
                    attributions = shap_values[0, :, 0]  # First instance, all features, single class
            elif len(shap_values.shape) == 2:
                # Shape is (n_instances, n_features) - regression or single class
                attributions = shap_values[0]
            else:
                # Single instance case
                attributions = shap_values
            
            # Ensure attributions is 1D array matching number of features
            attributions = np.array(attributions).flatten()
            
            # Final validation
            if len(attributions) != len(feature_names):
                raise ValueError(f"SHAP attributions length {len(attributions)} doesn't match number of features {len(feature_names)}")
            
            # Get model prediction
            model_output = self.model.predict(instance_array)[0]
            
            # Get baseline (expected value)
            baseline = getattr(self.shap_explainer, 'expected_value', None)
            if isinstance(baseline, (list, np.ndarray)) and len(baseline) > 0:
                baseline = baseline[0]
            
            return ExplanationResult(
                attributions=attributions,
                feature_names=feature_names,
                feature_values=feature_values,
                method='SHAP',
                model_output=model_output,
                baseline=baseline,
                explanation_type='local',
                metadata={
                    'explainer_type': self.explainer_type,
                    'shap_explainer': type(self.shap_explainer).__name__
                }
            )
            
        except Exception as e:
            raise ExplainerError(
                f"Failed to generate SHAP explanation: {str(e)}",
                explainer="SHAPExplainer"
            )
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Optional[np.ndarray] = None
    ) -> 'SHAPExplainer':
        """
        Fit the SHAP explainer (mainly for setting background data).
        
        Args:
            X: Training data to use as background
            y: Target values (not used)
            
        Returns:
            Self for method chaining
        """
        if self.background_data is None:
            # Use a sample of the training data as background
            sample_size = min(100, len(X))
            if isinstance(X, pd.DataFrame):
                self.background_data = X.sample(n=sample_size, random_state=42)
            else:
                indices = np.random.choice(len(X), size=sample_size, replace=False)
                self.background_data = X[indices]
            
            # Reinitialize explainer with background data
            self._init_shap_explainer()
        
        return super().fit(X, y)
    
    def get_feature_importance(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        importance_type: str = 'mean_abs'
    ) -> ExplanationResult:
        """
        Get global feature importance based on SHAP values.
        
        Args:
            X: Dataset to compute importance over
            importance_type: Type of importance ('mean_abs', 'mean', 'std')
            
        Returns:
            ExplanationResult containing feature importance
        """
        X = validate_data(X)
        
        # Get SHAP values for all instances
        shap_values = self.shap_explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Compute importance based on type
        if importance_type == 'mean_abs':
            importance = np.mean(np.abs(shap_values), axis=0)
        elif importance_type == 'mean':
            importance = np.mean(shap_values, axis=0)
        elif importance_type == 'std':
            importance = np.std(shap_values, axis=0)
        else:
            raise ValidationError(
                f"Unsupported importance type: {importance_type}",
                parameter="importance_type",
                expected="'mean_abs', 'mean', or 'std'"
            )
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return ExplanationResult(
            attributions=importance,
            feature_names=feature_names,
            method='SHAP',
            explanation_type='global',
            metadata={
                'importance_type': importance_type,
                'explainer_type': self.explainer_type,
                'sample_size': len(X)
            }
        )
