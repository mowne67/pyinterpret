"""
LIME (Local Interpretable Model-agnostic Explanations) explainer implementation.

This module provides a LIME-based explainer that generates local explanations
by fitting linear models around the instance being explained.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from pyinterpret.core.base import LocalExplainer, ExplanationResult
from pyinterpret.core.exceptions import ModelError, ValidationError, ExplainerError
from pyinterpret.utils.validation import validate_data


class LIMEExplainer(LocalExplainer):
    """
    LIME-based explainer for generating local linear explanations.
    
    This explainer creates local explanations by perturbing the input instance
    and fitting a linear model to approximate the original model's behavior
    in the local neighborhood.
    """
    
    def __init__(
        self,
        model: Any,
        training_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        mode: str = 'auto',
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: The machine learning model to explain
            training_data: Training data for LIME background distribution
            mode: Prediction mode ('auto', 'classification', 'regression')
            feature_names: Names of the features
            categorical_features: Indices of categorical features
            **kwargs: Additional parameters for LIME explainer
        """
        if not LIME_AVAILABLE:
            raise ExplainerError(
                "LIME library is not available. Please install it with: pip install lime",
                explainer="LIMEExplainer"
            )
        
        # Set attributes first before calling super().__init__
        self.training_data = training_data
        self.mode = mode
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.lime_explainer = None
        
        super().__init__(model, **kwargs)
        
        # LIME-specific parameters
        self.num_features = kwargs.get('num_features', 10)
        self.num_samples = kwargs.get('num_samples', 5000)
        self.distance_metric = kwargs.get('distance_metric', 'euclidean')
        self.kernel_width = kwargs.get('kernel_width', None)
        
        # Initialize LIME explainer if training data is provided
        if self.training_data is not None:
            self._init_lime_explainer()
    
    def _validate_model(self) -> None:
        """Validate that the model is compatible with LIME."""
        # Check if model has predict method
        if not hasattr(self.model, 'predict'):
            raise ModelError(
                "Model must have a 'predict' method",
                model_type=type(self.model).__name__,
                required_methods=['predict']
            )
        
        # For classification, also check for predict_proba
        if self.mode == 'classification' and not hasattr(self.model, 'predict_proba'):
            raise ModelError(
                "Classification model must have a 'predict_proba' method",
                model_type=type(self.model).__name__,
                required_methods=['predict', 'predict_proba']
            )
    
    def _detect_mode(self) -> str:
        """Automatically detect if the model is for classification or regression."""
        if hasattr(self.model, 'predict_proba'):
            return 'classification'
        elif hasattr(self.model, '_estimator_type'):
            return self.model._estimator_type
        else:
            # Default to regression
            return 'regression'
    
    def _init_lime_explainer(self) -> None:
        """Initialize the LIME tabular explainer."""
        if self.training_data is None:
            raise ExplainerError(
                "Training data is required to initialize LIME explainer",
                explainer="LIMEExplainer"
            )
        
        # Validate training data
        self.training_data = validate_data(self.training_data)
        
        # Auto-detect mode if needed
        if self.mode == 'auto':
            self.mode = self._detect_mode()
        
        # Prepare feature names
        if isinstance(self.training_data, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = self.training_data.columns.tolist()
            training_array = self.training_data.values
        else:
            training_array = self.training_data
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(training_array.shape[1])]
        
        try:
            # Initialize LIME explainer
            self.lime_explainer = LimeTabularExplainer(
                training_array,
                feature_names=self.feature_names,
                categorical_features=self.categorical_features,
                mode=self.mode,
                kernel_width=self.kernel_width,
                verbose=False
            )
        except Exception as e:
            raise ExplainerError(
                f"Failed to initialize LIME explainer: {str(e)}",
                explainer="LIMEExplainer"
            )
    
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Optional[np.ndarray] = None
    ) -> 'LIMEExplainer':
        """
        Fit the LIME explainer with training data.
        
        Args:
            X: Training data
            y: Target values (not used)
            
        Returns:
            Self for method chaining
        """
        self.training_data = X
        self._init_lime_explainer()
        return super().fit(X, y)
    
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single instance using LIME.
        
        Args:
            instance: Single instance to explain
            **kwargs: Additional parameters for LIME explanation
            
        Returns:
            ExplanationResult containing LIME explanations
        """
        if self.lime_explainer is None:
            raise ExplainerError(
                "LIME explainer not initialized. Call fit() first or provide training_data.",
                explainer="LIMEExplainer"
            )
        
        # Validate and prepare instance
        instance = validate_data(instance, expected_dims=1)
        
        if isinstance(instance, pd.Series):
            feature_names = instance.index.tolist()
            feature_values = instance.values
            instance_array = instance.values
        else:
            feature_names = self.feature_names or [f"feature_{i}" for i in range(len(instance))]
            feature_values = instance
            instance_array = instance
        
        # Extract parameters
        num_features = kwargs.get('num_features', self.num_features)
        num_samples = kwargs.get('num_samples', self.num_samples)
        
        try:
            # Create prediction function
            if self.mode == 'classification':
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                instance_array,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                **kwargs
            )
            
            # Extract explanation data
            lime_list = explanation.as_list()
            
            # Initialize attribution arrays
            attributions = np.zeros(len(feature_names))
            
            # Map LIME explanations to feature indices
            for feature_desc, attribution in lime_list:
                # LIME returns feature descriptions like "feature_name <= value"
                # We need to extract the feature name/index
                feature_idx = self._parse_feature_description(feature_desc, feature_names)
                if feature_idx is not None:
                    attributions[feature_idx] = attribution
            
            # Get model prediction
            model_output = self.model.predict(instance_array.reshape(1, -1))[0]
            
            # Get intercept as baseline
            baseline = None
            if hasattr(explanation, 'intercept'):
                try:
                    if self.mode == 'classification':
                        # For classification, intercept might be per class
                        baseline = explanation.intercept.get(1, explanation.intercept.get(0, None))
                    else:
                        # For regression, it's usually a single value
                        baseline = explanation.intercept[0] if isinstance(explanation.intercept, dict) else explanation.intercept
                except (KeyError, IndexError, TypeError):
                    baseline = None
            
            return ExplanationResult(
                attributions=attributions,
                feature_names=feature_names,
                feature_values=feature_values,
                method='LIME',
                model_output=model_output,
                baseline=baseline,
                explanation_type='local',
                metadata={
                    'num_features': num_features,
                    'num_samples': num_samples,
                    'score': explanation.score,
                    'local_pred': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None
                }
            )
            
        except Exception as e:
            raise ExplainerError(
                f"Failed to generate LIME explanation: {str(e)}",
                explainer="LIMEExplainer"
            )
    
    def _parse_feature_description(self, feature_desc: str, feature_names: List[str]) -> Optional[int]:
        """
        Parse LIME feature description to get feature index.
        
        Args:
            feature_desc: LIME feature description (e.g., "feature_1 <= 0.5")
            feature_names: List of feature names
            
        Returns:
            Feature index or None if not found
        """
        # Try to extract feature name from description
        for i, name in enumerate(feature_names):
            if feature_desc.startswith(name):
                return i
        
        # Try to extract feature index from description like "feature_1"
        if feature_desc.startswith('feature_'):
            try:
                idx = int(feature_desc.split('_')[1].split()[0])
                if 0 <= idx < len(feature_names):
                    return idx
            except (ValueError, IndexError):
                pass
        
        return None
    
    def get_local_surrogate_model(
        self,
        instance: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get the local surrogate model for an instance.
        
        Args:
            instance: Instance to explain
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing surrogate model information
        """
        explanation_result = self.explain_instance(instance, **kwargs)
        
        # Create a simple linear model with the coefficients
        surrogate_model = Ridge(alpha=0.0)
        
        # Mock fitting - we already have the coefficients from LIME
        surrogate_model.coef_ = explanation_result.attributions
        surrogate_model.intercept_ = explanation_result.baseline or 0.0
        
        return {
            'model': surrogate_model,
            'coefficients': explanation_result.attributions,
            'intercept': explanation_result.baseline,
            'feature_names': explanation_result.feature_names,
            'score': explanation_result.metadata.get('score', None),
            'local_prediction': explanation_result.metadata.get('local_pred', None)
        }
