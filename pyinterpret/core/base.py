"""
Base classes and interfaces for PyInterpret explainers.

This module defines the core abstractions that all explainer implementations
must follow to ensure consistency across the library.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ExplanationResult:
    """
    Container for explanation results from any explainer.
    
    This class provides a standardized format for returning explanation
    results regardless of the underlying explanation method.
    """
    
    # Core explanation data
    attributions: Optional[Union[np.ndarray, pd.DataFrame]] = None
    feature_names: Optional[List[str]] = None
    feature_values: Optional[Union[np.ndarray, pd.Series]] = None
    
    # Method-specific metadata
    method: str = ""
    model_output: Optional[Union[float, np.ndarray]] = None
    baseline: Optional[Union[float, np.ndarray]] = None
    
    # Additional information
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation_type: str = ""  # 'local' or 'global'
    
    def __post_init__(self):
        """Validate the explanation result after initialization."""
        if self.attributions is not None and self.feature_names is not None:
            if hasattr(self.attributions, 'shape'):
                if len(self.attributions.shape) == 1:
                    expected_length = len(self.attributions)
                else:
                    expected_length = self.attributions.shape[1]
                
                if len(self.feature_names) != expected_length:
                    raise ValueError(
                        f"Feature names length ({len(self.feature_names)}) "
                        f"doesn't match attributions shape ({expected_length})"
                    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation result to dictionary format."""
        result = {
            'method': self.method,
            'explanation_type': self.explanation_type,
            'metadata': self.metadata
        }
        
        if self.attributions is not None:
            if isinstance(self.attributions, pd.DataFrame):
                result['attributions'] = self.attributions.to_dict()
            else:
                result['attributions'] = self.attributions.tolist()
        
        if self.feature_names is not None:
            result['feature_names'] = self.feature_names
            
        if self.feature_values is not None:
            if isinstance(self.feature_values, pd.Series):
                result['feature_values'] = self.feature_values.to_dict()
            else:
                result['feature_values'] = self.feature_values.tolist()
        
        if self.model_output is not None:
            result['model_output'] = (
                self.model_output.tolist() 
                if isinstance(self.model_output, np.ndarray) 
                else self.model_output
            )
        
        if self.baseline is not None:
            result['baseline'] = (
                self.baseline.tolist() 
                if isinstance(self.baseline, np.ndarray) 
                else self.baseline
            )
        
        return result


class BaseExplainer(ABC):
    """
    Abstract base class for all explainer implementations.
    
    This class defines the common interface that all explainers must implement,
    ensuring consistency across different explanation methods.
    """
    
    def __init__(self, model: Any, **kwargs):
        """
        Initialize the explainer with a model.
        
        Args:
            model: The machine learning model to explain
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.config = kwargs
        self._is_fitted = False
        
        # Validate model during initialization
        self._validate_model()
    
    @abstractmethod
    def explain(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        **kwargs
    ) -> ExplanationResult:
        """
        Generate explanations for the given input data.
        
        Args:
            X: Input data to explain
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult containing the explanations
        """
        pass
    
    @abstractmethod
    def _validate_model(self) -> None:
        """
        Validate that the provided model is compatible with this explainer.
        
        Raises:
            ModelError: If the model is not compatible
        """
        pass
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> 'BaseExplainer':
        """
        Fit the explainer to the data (if required by the method).
        
        Args:
            X: Training data
            y: Target values (optional)
            
        Returns:
            Self for method chaining
        """
        self._is_fitted = True
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get explainer parameters."""
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseExplainer':
        """Set explainer parameters."""
        self.config.update(params)
        return self
    
    @property
    def explanation_type(self) -> str:
        """Return the type of explanations this explainer provides."""
        return getattr(self, '_explanation_type', 'unknown')
    
    def __repr__(self) -> str:
        """String representation of the explainer."""
        class_name = self.__class__.__name__
        return f"{class_name}(model={type(self.model).__name__})"


class LocalExplainer(BaseExplainer):
    """Base class for local (instance-level) explainers."""
    
    _explanation_type = 'local'
    
    @abstractmethod
    def explain_instance(
        self, 
        instance: Union[np.ndarray, pd.Series], 
        **kwargs
    ) -> ExplanationResult:
        """
        Explain a single instance.
        
        Args:
            instance: Single instance to explain
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult for the instance
        """
        pass
    
    def explain(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        **kwargs
    ) -> Union[ExplanationResult, List[ExplanationResult]]:
        """
        Explain one or more instances.
        
        Args:
            X: Input data to explain
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult or list of ExplanationResults
        """
        if hasattr(X, 'shape') and len(X.shape) == 1:
            # Single instance
            return self.explain_instance(X, **kwargs)
        else:
            # Multiple instances
            results = []
            for i in range(len(X)):
                instance = X.iloc[i] if isinstance(X, pd.DataFrame) else X[i]
                results.append(self.explain_instance(instance, **kwargs))
            return results


class GlobalExplainer(BaseExplainer):
    """Base class for global (model-level) explainers."""
    
    _explanation_type = 'global'
    
    @abstractmethod
    def explain_global(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        **kwargs
    ) -> ExplanationResult:
        """
        Generate global explanations for the model.
        
        Args:
            X: Dataset to use for global explanation
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult containing global explanations
        """
        pass
    
    def explain(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        **kwargs
    ) -> ExplanationResult:
        """
        Generate explanations (delegates to explain_global).
        
        Args:
            X: Input data to explain
            **kwargs: Method-specific parameters
            
        Returns:
            ExplanationResult containing the explanations
        """
        return self.explain_global(X, **kwargs)
