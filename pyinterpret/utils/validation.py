"""
Input validation utilities for PyInterpret.

This module provides comprehensive validation functions to ensure
inputs are properly formatted and compatible with explainer methods.
"""

from typing import Any, List, Optional, Union, Tuple, Dict
import numpy as np
import pandas as pd

from pyinterpret.core.exceptions import ValidationError, ModelError, DataError


def validate_model(model: Any, required_methods: Optional[List[str]] = None) -> None:
    """
    Validate that a model has the required methods.
    
    Args:
        model: Model object to validate
        required_methods: List of required method names
        
    Raises:
        ModelError: If model doesn't have required methods
    """
    if required_methods is None:
        required_methods = ['predict']
    
    missing_methods = []
    for method in required_methods:
        if not hasattr(model, method):
            missing_methods.append(method)
    
    if missing_methods:
        raise ModelError(
            f"Model is missing required methods: {missing_methods}",
            model_type=type(model).__name__,
            required_methods=required_methods
        )


def validate_data(
    data: Union[np.ndarray, pd.DataFrame, pd.Series],
    expected_dims: Optional[int] = None,
    min_samples: int = 1,
    feature_names: Optional[List[str]] = None
) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Validate and potentially transform input data.
    
    Args:
        data: Input data to validate
        expected_dims: Expected number of dimensions (1 or 2)
        min_samples: Minimum number of samples required
        feature_names: Expected feature names for DataFrames
        
    Returns:
        Validated data (potentially transformed)
        
    Raises:
        DataError: If data doesn't meet requirements
        ValidationError: If validation parameters are invalid
    """
    if data is None:
        raise DataError("Data cannot be None")
    
    # Convert to numpy/pandas if needed
    if isinstance(data, list):
        data = np.array(data)
    
    # Check data type
    if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
        raise DataError(
            f"Data must be numpy array, pandas DataFrame, or pandas Series. Got {type(data)}"
        )
    
    # Check for empty data
    if len(data) == 0:
        raise DataError("Data cannot be empty")
    
    # Check minimum samples
    if len(data) < min_samples:
        raise DataError(
            f"Data must have at least {min_samples} samples. Got {len(data)}",
            data_shape=data.shape if hasattr(data, 'shape') else (len(data),)
        )
    
    # Check dimensions
    if expected_dims is not None:
        if isinstance(data, pd.Series):
            actual_dims = 1
        elif isinstance(data, (np.ndarray, pd.DataFrame)):
            actual_dims = len(data.shape)
        else:
            actual_dims = 1
        
        if actual_dims != expected_dims:
            raise DataError(
                f"Data must have {expected_dims} dimensions. Got {actual_dims}",
                data_shape=data.shape if hasattr(data, 'shape') else (len(data),),
                expected_shape=f"{expected_dims}D"
            )
    
    # Check for NaN values
    if isinstance(data, pd.DataFrame):
        if data.isnull().any().any():
            raise DataError("Data contains NaN values")
    elif isinstance(data, pd.Series):
        if data.isnull().any():
            raise DataError("Data contains NaN values")
    elif isinstance(data, np.ndarray):
        if np.isnan(data).any():
            raise DataError("Data contains NaN values")
    
    # Check for infinite values
    if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
        if isinstance(data, pd.DataFrame):
            has_inf = np.isinf(data.select_dtypes(include=[np.number])).any().any()
        elif isinstance(data, pd.Series):
            has_inf = np.isinf(data).any() if pd.api.types.is_numeric_dtype(data) else False
        else:
            has_inf = np.isinf(data).any()
        
        if has_inf:
            raise DataError("Data contains infinite values")
    
    # Validate feature names if provided
    if feature_names is not None and isinstance(data, pd.DataFrame):
        if not all(name in data.columns for name in feature_names):
            missing_features = [name for name in feature_names if name not in data.columns]
            raise ValidationError(
                f"Features not found in data: {missing_features}",
                parameter="feature_names"
            )
    
    return data


def validate_features(
    features: Union[int, str, List[Union[int, str]]],
    feature_names: List[str],
    max_features: Optional[int] = None
) -> Tuple[List[int], List[str]]:
    """
    Validate and convert feature specification to indices and names.
    
    Args:
        features: Feature specification (indices or names)
        feature_names: Available feature names
        max_features: Maximum number of features allowed
        
    Returns:
        Tuple of (feature_indices, selected_feature_names)
        
    Raises:
        ValidationError: If feature specification is invalid
    """
    if not isinstance(features, list):
        features = [features]
    
    if max_features is not None and len(features) > max_features:
        raise ValidationError(
            f"Too many features specified. Maximum allowed: {max_features}, got: {len(features)}",
            parameter="features"
        )
    
    feature_indices = []
    selected_names = []
    
    for feature in features:
        if isinstance(feature, int):
            if 0 <= feature < len(feature_names):
                feature_indices.append(feature)
                selected_names.append(feature_names[feature])
            else:
                raise ValidationError(
                    f"Feature index {feature} out of range [0, {len(feature_names)-1}]",
                    parameter="features"
                )
        elif isinstance(feature, str):
            if feature in feature_names:
                idx = feature_names.index(feature)
                feature_indices.append(idx)
                selected_names.append(feature)
            else:
                raise ValidationError(
                    f"Feature '{feature}' not found in available features",
                    parameter="features",
                    expected=f"One of {feature_names}"
                )
        else:
            raise ValidationError(
                f"Feature must be int or str, got {type(feature)}",
                parameter="features"
            )
    
    return feature_indices, selected_names


def validate_target(
    y: Union[np.ndarray, pd.Series],
    X_length: int,
    task_type: str = 'auto'
) -> np.ndarray:
    """
    Validate target values.
    
    Args:
        y: Target values
        X_length: Expected length (should match X)
        task_type: Task type ('classification', 'regression', 'auto')
        
    Returns:
        Validated target array
        
    Raises:
        DataError: If target validation fails
    """
    y = validate_data(y, expected_dims=1, min_samples=1)
    
    if len(y) != X_length:
        raise DataError(
            f"Target length ({len(y)}) doesn't match feature length ({X_length})",
            data_shape=(len(y),),
            expected_shape=(X_length,)
        )
    
    # Convert to numpy array
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = y
    
    # Validate based on task type
    if task_type == 'classification':
        # Check if target values are discrete
        unique_values = np.unique(y_array)
        if len(unique_values) > 100:
            raise DataError(
                f"Too many unique values for classification: {len(unique_values)}"
            )
    elif task_type == 'regression':
        # Check if target values are numeric
        if not np.issubdtype(y_array.dtype, np.number):
            raise DataError(
                f"Target values must be numeric for regression. Got dtype: {y_array.dtype}"
            )
    
    return y_array


def validate_explanation_result(result: 'ExplanationResult') -> None:
    """
    Validate an ExplanationResult object.
    
    Args:
        result: ExplanationResult to validate
        
    Raises:
        ValidationError: If result is invalid
    """
    from pyinterpret.core.base import ExplanationResult
    
    if not isinstance(result, ExplanationResult):
        raise ValidationError(
            f"Expected ExplanationResult, got {type(result)}",
            parameter="result"
        )
    
    # Check required fields
    if result.method is None or result.method == "":
        raise ValidationError(
            "ExplanationResult must have a method specified",
            parameter="method"
        )
    
    if result.explanation_type not in ['local', 'global', '']:
        raise ValidationError(
            f"Invalid explanation type: {result.explanation_type}",
            parameter="explanation_type",
            expected="'local' or 'global'"
        )
    
    # Check consistency between attributions and feature names
    if result.attributions is not None and result.feature_names is not None:
        if hasattr(result.attributions, 'shape'):
            if len(result.attributions.shape) == 1:
                expected_length = len(result.attributions)
            else:
                expected_length = result.attributions.shape[-1]
        else:
            expected_length = len(result.attributions)
        
        if len(result.feature_names) != expected_length:
            raise ValidationError(
                f"Feature names length ({len(result.feature_names)}) doesn't match attributions length ({expected_length})",
                parameter="feature_names"
            )


def check_sklearn_compatibility(model: Any) -> Dict[str, bool]:
    """
    Check sklearn compatibility of a model.
    
    Args:
        model: Model to check
        
    Returns:
        Dictionary with compatibility flags
    """
    compatibility = {
        'has_predict': hasattr(model, 'predict'),
        'has_predict_proba': hasattr(model, 'predict_proba'),
        'has_decision_function': hasattr(model, 'decision_function'),
        'has_fit': hasattr(model, 'fit'),
        'has_get_params': hasattr(model, 'get_params'),
        'has_set_params': hasattr(model, 'set_params'),
        'is_classifier': hasattr(model, '_estimator_type') and model._estimator_type == 'classifier',
        'is_regressor': hasattr(model, '_estimator_type') and model._estimator_type == 'regressor'
    }
    
    return compatibility
