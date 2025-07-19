"""
Tests for core PyInterpret functionality.

This module tests the base classes and core abstractions
used throughout the library.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from pyinterpret.core.base import BaseExplainer, ExplanationResult, LocalExplainer, GlobalExplainer
from pyinterpret.core.exceptions import PyInterpretError, ValidationError, ModelError


class TestExplanationResult:
    """Test ExplanationResult class."""
    
    def test_basic_creation(self):
        """Test basic creation of ExplanationResult."""
        result = ExplanationResult(
            attributions=np.array([0.1, 0.2, -0.3]),
            feature_names=['a', 'b', 'c'],
            method='test'
        )
        
        assert result.method == 'test'
        assert len(result.attributions) == 3
        assert len(result.feature_names) == 3
    
    def test_validation_error(self):
        """Test validation error on mismatched lengths."""
        with pytest.raises(ValueError):
            ExplanationResult(
                attributions=np.array([0.1, 0.2]),
                feature_names=['a', 'b', 'c'],
                method='test'
            )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ExplanationResult(
            attributions=np.array([0.1, 0.2, -0.3]),
            feature_names=['a', 'b', 'c'],
            method='test',
            explanation_type='local'
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['method'] == 'test'
        assert result_dict['explanation_type'] == 'local'
        assert 'attributions' in result_dict
        assert 'feature_names' in result_dict


class MockExplainer(BaseExplainer):
    """Mock explainer for testing base functionality."""
    
    def _validate_model(self):
        if not hasattr(self.model, 'predict'):
            raise ModelError("Model needs predict method")
    
    def explain(self, X, **kwargs):
        return ExplanationResult(
            attributions=np.random.random(X.shape[1]),
            feature_names=[f'feature_{i}' for i in range(X.shape[1])],
            method='mock'
        )


class MockLocalExplainer(LocalExplainer):
    """Mock local explainer for testing."""
    
    def _validate_model(self):
        if not hasattr(self.model, 'predict'):
            raise ModelError("Model needs predict method")
    
    def explain_instance(self, instance, **kwargs):
        return ExplanationResult(
            attributions=np.random.random(len(instance)),
            feature_names=[f'feature_{i}' for i in range(len(instance))],
            method='mock_local',
            explanation_type='local'
        )


class MockGlobalExplainer(GlobalExplainer):
    """Mock global explainer for testing."""
    
    def _validate_model(self):
        if not hasattr(self.model, 'predict'):
            raise ModelError("Model needs predict method")
    
    def explain_global(self, X, **kwargs):
        return ExplanationResult(
            attributions=np.random.random(X.shape[1]),
            feature_names=[f'feature_{i}' for i in range(X.shape[1])],
            method='mock_global',
            explanation_type='global'
        )


class TestBaseExplainer:
    """Test BaseExplainer functionality."""
    
    def test_initialization_with_valid_model(self):
        """Test initialization with valid model."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        explainer = MockExplainer(model)
        assert explainer.model == model
        assert not explainer._is_fitted
    
    def test_initialization_with_invalid_model(self):
        """Test initialization with invalid model."""
        model = Mock()
        # No predict method
        
        with pytest.raises(ModelError):
            MockExplainer(model)
    
    def test_fit(self):
        """Test fitting explainer."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        explainer = MockExplainer(model)
        X = np.random.random((10, 3))
        
        result = explainer.fit(X)
        assert result == explainer
        assert explainer._is_fitted
    
    def test_get_set_params(self):
        """Test parameter getting and setting."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        explainer = MockExplainer(model, param1='value1', param2=42)
        
        params = explainer.get_params()
        assert params['param1'] == 'value1'
        assert params['param2'] == 42
        
        explainer.set_params(param1='new_value', param3='value3')
        new_params = explainer.get_params()
        assert new_params['param1'] == 'new_value'
        assert new_params['param3'] == 'value3'
    
    def test_string_representation(self):
        """Test string representation."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        explainer = MockExplainer(model)
        repr_str = repr(explainer)
        
        assert 'MockExplainer' in repr_str
        assert 'Mock' in repr_str


class TestLocalExplainer:
    """Test LocalExplainer functionality."""
    
    def test_explain_single_instance(self):
        """Test explaining single instance."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1]))
        
        explainer = MockLocalExplainer(model)
        instance = np.array([0.1, 0.2, 0.3])
        
        result = explainer.explain(instance)
        assert isinstance(result, ExplanationResult)
        assert result.explanation_type == 'local'
    
    def test_explain_multiple_instances(self):
        """Test explaining multiple instances."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 2]))
        
        explainer = MockLocalExplainer(model)
        X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        results = explainer.explain(X)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ExplanationResult) for r in results)
    
    def test_explanation_type(self):
        """Test explanation type property."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1]))
        
        explainer = MockLocalExplainer(model)
        assert explainer.explanation_type == 'local'


class TestGlobalExplainer:
    """Test GlobalExplainer functionality."""
    
    def test_explain_global(self):
        """Test global explanation."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 2, 3]))
        
        explainer = MockGlobalExplainer(model)
        X = np.random.random((10, 3))
        
        result = explainer.explain(X)
        assert isinstance(result, ExplanationResult)
        assert result.explanation_type == 'global'
    
    def test_explanation_type(self):
        """Test explanation type property."""
        model = Mock()
        model.predict = Mock(return_value=np.array([1]))
        
        explainer = MockGlobalExplainer(model)
        assert explainer.explanation_type == 'global'


class TestExceptions:
    """Test custom exceptions."""
    
    def test_pyinterpret_error(self):
        """Test PyInterpretError."""
        error = PyInterpretError("Test message", "Additional details")
        assert str(error) == "Test message\nDetails: Additional details"
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Invalid parameter",
            parameter="test_param",
            expected="string",
            received="int"
        )
        
        assert "Invalid parameter" in str(error)
        assert "test_param" in str(error)
    
    def test_model_error(self):
        """Test ModelError."""
        error = ModelError(
            "Model missing methods",
            model_type="TestModel",
            required_methods=["predict", "fit"]
        )
        
        assert "Model missing methods" in str(error)
        assert "TestModel" in str(error)
        assert "predict" in str(error)


if __name__ == "__main__":
    pytest.main([__file__])
