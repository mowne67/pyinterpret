"""
Tests for local explainers in PyInterpret.

This module tests SHAP and LIME explainer implementations
to ensure they work correctly with different model types.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression

from pyinterpret.local.shap_explainer import SHAPExplainer
from pyinterpret.local.lime_explainer import LIMEExplainer
from pyinterpret.core.base import ExplanationResult
from pyinterpret.core.exceptions import ModelError, ExplainerError, ValidationError


class TestSHAPExplainer:
    """Test SHAP explainer functionality."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    @pytest.fixture
    def rf_classifier(self, classification_data):
        """Create fitted random forest classifier."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def rf_regressor(self, regression_data):
        """Create fitted random forest regressor."""
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def lr_classifier(self, classification_data):
        """Create fitted logistic regression."""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model
    
    def test_initialization_with_tree_model(self, rf_classifier):
        """Test initialization with tree-based model."""
        explainer = SHAPExplainer(rf_classifier, explainer_type='auto')
        assert explainer.explainer_type == 'auto'
        assert explainer.shap_explainer is not None
    
    def test_initialization_without_shap(self):
        """Test initialization when SHAP is not available."""
        with patch('pyinterpret.local.shap_explainer.SHAP_AVAILABLE', False):
            model = Mock()
            model.predict = Mock()
            
            with pytest.raises(ExplainerError, match="SHAP library is not available"):
                SHAPExplainer(model)
    
    def test_model_validation_no_predict(self):
        """Test model validation fails without predict method."""
        model = Mock(spec=[])
        # No predict method

        with pytest.raises(ModelError, match="predict"):
            SHAPExplainer(model)
    
    def test_auto_explainer_selection_tree(self, rf_classifier):
        """Test automatic selection of tree explainer."""
        explainer = SHAPExplainer(rf_classifier, explainer_type='auto')
        # Should automatically select tree explainer for RandomForest
        assert explainer.shap_explainer is not None
    
    def test_auto_explainer_selection_linear(self, lr_classifier):
        """Test automatic selection of linear explainer."""
        explainer = SHAPExplainer(lr_classifier, explainer_type='auto')
        # Should automatically select linear explainer for LogisticRegression
        assert explainer.shap_explainer is not None
    
    def test_kernel_explainer_needs_background(self, lr_classifier):
        """Test that kernel explainer requires background data."""
        with pytest.raises(ExplainerError, match="Background data is required"):
            SHAPExplainer(lr_classifier, explainer_type='kernel')
    
    def test_kernel_explainer_with_background(self, lr_classifier, classification_data):
        """Test kernel explainer with background data."""
        X, _ = classification_data
        background = X[:10]  # Small background sample
        
        explainer = SHAPExplainer(
            lr_classifier, 
            explainer_type='kernel',
            background_data=background
        )
        assert explainer.shap_explainer is not None
    
    @patch('pyinterpret.local.shap_explainer.shap')
    def test_explain_instance_single(self, mock_shap, rf_classifier, classification_data):
        """Test explaining single instance."""
        X, _ = classification_data
        instance = X[0]
        
        # Mock SHAP values
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.array([[0.1, 0.2, -0.3, 0.0, 0.1, -0.2, 0.3, -0.1, 0.2, -0.1]])
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(rf_classifier)
        result = explainer.explain_instance(instance)
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'SHAP'
        assert result.explanation_type == 'local'
        assert len(result.attributions) == len(instance)
        assert len(result.feature_names) == len(instance)
    
    @patch('pyinterpret.local.shap_explainer.shap')
    def test_explain_instance_with_pandas(self, mock_shap, rf_classifier, classification_data):
        """Test explaining instance with pandas Series."""
        X, _ = classification_data
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        instance = pd.Series(X[0], index=feature_names)
        
        # Mock SHAP values
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.array([[0.1, 0.2, -0.3, 0.0, 0.1, -0.2, 0.3, -0.1, 0.2, -0.1]])
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(rf_classifier)
        result = explainer.explain_instance(instance)
        
        assert isinstance(result, ExplanationResult)
        assert result.feature_names == feature_names
        assert len(result.feature_values) == len(instance)
    
    @patch('pyinterpret.local.shap_explainer.shap')
    def test_explain_multiple_instances(self, mock_shap, rf_classifier, classification_data):
        """Test explaining multiple instances."""
        X, _ = classification_data
        instances = X[:3]
        
        # Mock SHAP values
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.random((3, X.shape[1]))
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(rf_classifier)
        results = explainer.explain(instances)
        
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, ExplanationResult) for r in results)
    
    @patch('pyinterpret.local.shap_explainer.shap')
    def test_get_feature_importance(self, mock_shap, rf_classifier, classification_data):
        """Test getting global feature importance."""
        X, _ = classification_data
        
        # Mock SHAP values
        mock_explainer = Mock()
        mock_explainer.shap_values.return_value = np.random.random((len(X), X.shape[1]))
        mock_explainer.expected_value = 0.5
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        explainer = SHAPExplainer(rf_classifier)
        result = explainer.get_feature_importance(X)
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'SHAP'
        assert result.explanation_type == 'global'
        assert len(result.attributions) == X.shape[1]
    
    def test_fit_sets_background_data(self, rf_classifier, classification_data):
        """Test that fit method sets background data."""
        X, _ = classification_data
        
        explainer = SHAPExplainer(rf_classifier)
        explainer.fit(X)
        
        assert explainer.background_data is not None
        assert len(explainer.background_data) <= 100  # Should sample at most 100 points
    
    def test_invalid_explainer_type(self, rf_classifier):
        """Test invalid explainer type."""
        with pytest.raises(ExplainerError, match="Unsupported explainer type"):
            SHAPExplainer(rf_classifier, explainer_type='invalid')


class TestLIMEExplainer:
    """Test LIME explainer functionality."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        return X, y
    
    @pytest.fixture
    def rf_classifier(self, classification_data):
        """Create fitted random forest classifier."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def rf_regressor(self, regression_data):
        """Create fitted random forest regressor."""
        X, y = regression_data
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    def test_initialization_without_lime(self):
        """Test initialization when LIME is not available."""
        with patch('pyinterpret.local.lime_explainer.LIME_AVAILABLE', False):
            model = Mock()
            model.predict = Mock()
            
            with pytest.raises(ExplainerError, match="LIME library is not available"):
                LIMEExplainer(model)
    
    def test_model_validation_no_predict(self):
        """Test model validation fails without predict method."""
        model = Mock(spec=[])
        # No predict method

        with pytest.raises(ModelError, match="predict"):
            LIMEExplainer(model)

    def test_classification_model_validation_no_predict_proba(self):
        """Test classification model validation without predict_proba."""
        model = Mock(spec=['predict'])
        # No predict_proba method
        
        with pytest.raises(ModelError, match="predict_proba"):
            LIMEExplainer(model, mode='classification')
    
    def test_initialization_with_training_data(self, rf_classifier, classification_data):
        """Test initialization with training data."""
        X, _ = classification_data
        
        explainer = LIMEExplainer(rf_classifier, training_data=X, mode='classification')
        assert explainer.lime_explainer is not None
        assert explainer.training_data is not None
    
    def test_initialization_without_training_data(self, rf_classifier):
        """Test initialization without training data."""
        explainer = LIMEExplainer(rf_classifier, mode='classification')
        assert explainer.lime_explainer is None
        assert explainer.training_data is None
    
    def test_fit_initializes_lime(self, rf_classifier, classification_data):
        """Test that fit initializes LIME explainer."""
        X, _ = classification_data
        
        explainer = LIMEExplainer(rf_classifier, mode='classification')
        explainer.fit(X)
        
        assert explainer.lime_explainer is not None
        assert explainer.training_data is not None
    
    def test_mode_detection_classifier(self, rf_classifier):
        """Test automatic mode detection for classifier."""
        explainer = LIMEExplainer(rf_classifier, mode='auto')
        explainer._detect_mode()
        # RandomForest has predict_proba, so should be classification
    
    def test_mode_detection_regressor(self, rf_regressor):
        """Test automatic mode detection for regressor."""
        explainer = LIMEExplainer(rf_regressor, mode='auto')
        mode = explainer._detect_mode()
        # RandomForestRegressor has _estimator_type
        assert mode in ['regression', 'regressor']
    
    @patch('pyinterpret.local.lime_explainer.LimeTabularExplainer')
    def test_explain_instance_classification(self, mock_lime_class, rf_classifier, classification_data):
        """Test explaining instance for classification."""
        X, _ = classification_data
        instance = X[0]
        
        # Mock LIME explanation
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [
            ('feature_0 <= 0.5', 0.1),
            ('feature_1 > 0.3', -0.2),
            ('feature_2 <= 1.0', 0.3)
        ]
        mock_explanation.intercept = [0.5]
        mock_explanation.score = 0.95
        mock_explanation.local_pred = [0.8]
        
        mock_lime_explainer = Mock()
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        mock_lime_class.return_value = mock_lime_explainer
        
        explainer = LIMEExplainer(rf_classifier, training_data=X, mode='classification')
        result = explainer.explain_instance(instance)
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'LIME'
        assert result.explanation_type == 'local'
        assert len(result.attributions) == len(instance)
    
    @patch('pyinterpret.local.lime_explainer.LimeTabularExplainer')
    def test_explain_instance_pandas(self, mock_lime_class, rf_classifier, classification_data):
        """Test explaining pandas Series instance."""
        X, _ = classification_data
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        instance = pd.Series(X[0], index=feature_names)
        
        # Mock LIME explanation
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [
            ('feature_0 <= 0.5', 0.1),
            ('feature_1 > 0.3', -0.2)
        ]
        mock_explanation.intercept = [0.5]
        mock_explanation.score = 0.95
        mock_explanation.local_pred = [0.8]

        mock_lime_explainer = Mock()
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        mock_lime_class.return_value = mock_lime_explainer

        explainer = LIMEExplainer(rf_classifier, training_data=X.copy(), mode='classification')
        result = explainer.explain_instance(instance)
        
        assert result.feature_names == feature_names
        assert len(result.feature_values) == len(instance)
    
    def test_explain_without_lime_explainer(self, rf_classifier):
        """Test explaining without initialized LIME explainer."""
        explainer = LIMEExplainer(rf_classifier, mode='classification')
        instance = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        with pytest.raises(ExplainerError, match="LIME explainer not initialized"):
            explainer.explain_instance(instance)
    
    @patch('pyinterpret.local.lime_explainer.LimeTabularExplainer')
    def test_get_local_surrogate_model(self, mock_lime_class, rf_classifier, classification_data):
        """Test getting local surrogate model."""
        X, _ = classification_data
        instance = X[0]
        
        # Mock LIME explanation
        mock_explanation = Mock()
        mock_explanation.as_list.return_value = [
            ('feature_0 <= 0.5', 0.1),
            ('feature_1 > 0.3', -0.2)
        ]
        mock_explanation.intercept = [0.5]
        mock_explanation.score = 0.95
        mock_explanation.local_pred = [0.8]
        
        mock_lime_explainer = Mock()
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        mock_lime_class.return_value = mock_lime_explainer
        
        explainer = LIMEExplainer(rf_classifier, training_data=X, mode='classification')
        surrogate_info = explainer.get_local_surrogate_model(instance)
        
        assert 'model' in surrogate_info
        assert 'coefficients' in surrogate_info
        assert 'intercept' in surrogate_info
        assert 'feature_names' in surrogate_info
    
    def test_parse_feature_description(self, rf_classifier):
        """Test parsing LIME feature descriptions."""
        explainer = LIMEExplainer(rf_classifier, mode='classification')
        feature_names = ['age', 'income', 'score']
        
        # Test exact feature name match
        idx = explainer._parse_feature_description('age <= 25', feature_names)
        assert idx == 0
        
        # Test feature index format
        idx = explainer._parse_feature_description('feature_1 > 0.5', feature_names)
        assert idx == 1
        
        # Test no match
        idx = explainer._parse_feature_description('unknown_feature', feature_names)
        assert idx is None


if __name__ == "__main__":
    pytest.main([__file__])
