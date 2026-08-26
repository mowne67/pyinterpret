"""
Tests for global explainers in PyInterpret.

This module tests permutation importance and partial dependence
explainer implementations.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error

from pyinterpret.global_.permutation_importance import PermutationImportanceExplainer
from pyinterpret.global_.partial_dependence import PartialDependenceExplainer
from pyinterpret.core.base import ExplanationResult
from pyinterpret.core.exceptions import ModelError, ExplainerError, ValidationError


class TestPermutationImportanceExplainer:
    """Test permutation importance explainer functionality."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, 
                                 n_informative=5, random_state=42)
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(n_samples=200, n_features=10, n_informative=5, 
                             random_state=42)
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
    
    def test_initialization_auto_scoring(self, rf_classifier):
        """Test initialization with auto scoring."""
        explainer = PermutationImportanceExplainer(rf_classifier, scoring='auto')
        assert explainer.scoring == 'auto'
        assert explainer.n_repeats == 5
    
    def test_initialization_custom_scoring(self, rf_regressor):
        """Test initialization with custom scoring."""
        explainer = PermutationImportanceExplainer(
            rf_regressor, 
            scoring='mse',
            n_repeats=10,
            random_state=42
        )
        assert explainer.scoring == 'mse'
        assert explainer.n_repeats == 10
        assert explainer.random_state == 42
    
    def test_model_validation_no_predict(self):
        """Test model validation fails without predict method."""
        model = Mock(spec=[])
        # No predict method

        with pytest.raises(ModelError, match="predict"):
            PermutationImportanceExplainer(model)
    
    def test_scoring_function_setup_auto_classifier(self, rf_classifier):
        """Test auto scoring setup for classifier."""
        explainer = PermutationImportanceExplainer(rf_classifier, scoring='auto')
        explainer._setup_scoring()
        assert explainer.scoring_name == 'accuracy'
    
    def test_scoring_function_setup_auto_regressor(self, rf_regressor):
        """Test auto scoring setup for regressor."""
        explainer = PermutationImportanceExplainer(rf_regressor, scoring='auto')
        explainer._setup_scoring()
        assert explainer.scoring_name == 'neg_mse'
    
    def test_scoring_function_setup_custom(self, rf_classifier):
        """Test custom scoring function setup."""
        explainer = PermutationImportanceExplainer(rf_classifier, scoring='accuracy')
        explainer._setup_scoring()
        assert explainer.scoring_name == 'accuracy'
    
    def test_invalid_scoring_function(self, rf_classifier):
        """Test invalid scoring function."""
        with pytest.raises(ValidationError, match="Unsupported scoring function"):
            PermutationImportanceExplainer(rf_classifier, scoring='invalid_scoring')
    
    def test_explain_global_classification(self, rf_classifier, classification_data):
        """Test global explanation for classification."""
        X, y = classification_data
        
        explainer = PermutationImportanceExplainer(
            rf_classifier, 
            scoring='accuracy',
            n_repeats=3,
            random_state=42
        )
        result = explainer.explain_global(X, y)
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'PermutationImportance'
        assert result.explanation_type == 'global'
        assert len(result.attributions) == X.shape[1]
        assert len(result.feature_names) == X.shape[1]
        
        # Check metadata
        assert 'baseline_score' in result.metadata
        assert 'scoring' in result.metadata
        assert 'std_importance' in result.metadata
        assert result.metadata['n_repeats'] == 3
    
    def test_explain_global_regression(self, rf_regressor, regression_data):
        """Test global explanation for regression."""
        X, y = regression_data
        
        explainer = PermutationImportanceExplainer(
            rf_regressor,
            scoring='neg_mse',
            n_repeats=2,
            random_state=42
        )
        result = explainer.explain_global(X, y)
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'PermutationImportance'
        assert len(result.attributions) == X.shape[1]
        assert result.metadata['scoring'] == 'neg_mse'
    
    def test_explain_global_with_pandas(self, rf_classifier, classification_data):
        """Test global explanation with pandas DataFrame."""
        X, y = classification_data
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        explainer = PermutationImportanceExplainer(rf_classifier, scoring='accuracy')
        result = explainer.explain_global(X_df, y_series)
        
        assert result.feature_names == feature_names
        assert len(result.attributions) == len(feature_names)
    
    def test_explain_global_mismatched_lengths(self, rf_classifier, classification_data):
        """Test error with mismatched X and y lengths."""
        X, y = classification_data
        y_short = y[:len(y)//2]
        
        explainer = PermutationImportanceExplainer(rf_classifier)
        
        with pytest.raises(ValidationError, match="same length"):
            explainer.explain_global(X, y_short)
    
    def test_get_feature_ranking(self, rf_classifier, classification_data):
        """Test getting feature ranking."""
        X, y = classification_data
        
        explainer = PermutationImportanceExplainer(
            rf_classifier,
            scoring='accuracy',
            n_repeats=2,
            random_state=42
        )
        ranking = explainer.get_feature_ranking(X, y, top_k=5)
        
        assert 'ranking' in ranking
        assert 'baseline_score' in ranking
        assert 'total_features' in ranking
        assert len(ranking['ranking']) <= 5
        
        # Check ranking structure
        for i, item in enumerate(ranking['ranking']):
            assert 'feature' in item
            assert 'importance' in item
            assert 'std' in item
            assert 'rank' in item
            assert item['rank'] == i + 1
    
    def test_get_feature_ranking_all_features(self, rf_classifier, classification_data):
        """Test getting ranking for all features."""
        X, y = classification_data
        
        explainer = PermutationImportanceExplainer(rf_classifier, n_repeats=2)
        ranking = explainer.get_feature_ranking(X, y, top_k=None)
        
        assert len(ranking['ranking']) == X.shape[1]
    
    @patch('matplotlib.pyplot')
    def test_plot_importance(self, mock_plt, rf_classifier, classification_data):
        """Test plotting importance."""
        X, y = classification_data
        
        explainer = PermutationImportanceExplainer(rf_classifier, n_repeats=2)
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig = explainer.plot_importance(X, y, top_k=5)
        
        assert fig == mock_fig
        mock_plt.subplots.assert_called_once()
        mock_ax.barh.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
    
    def test_plot_importance_without_matplotlib(self, rf_classifier, classification_data):
        """Test plotting without matplotlib."""
        X, y = classification_data
        
        with patch('pyinterpret.global_.permutation_importance.plt', None):
            explainer = PermutationImportanceExplainer(rf_classifier)
            
            with pytest.raises(ExplainerError, match="Matplotlib is required"):
                explainer.plot_importance(X, y)
    
    def test_custom_scoring_function(self, rf_classifier, classification_data):
        """Test with custom scoring function."""
        X, y = classification_data
        
        def custom_scorer(y_true, y_pred):
            return accuracy_score(y_true, y_pred)
        
        explainer = PermutationImportanceExplainer(rf_classifier, scoring=custom_scorer)
        result = explainer.explain_global(X, y)
        
        assert result.metadata['scoring'] == 'custom'


class TestPartialDependenceExplainer:
    """Test partial dependence explainer functionality."""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        X, y = make_classification(n_samples=200, n_features=8, n_classes=2,
                                 n_informative=4, random_state=42)
        return X, y
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(n_samples=200, n_features=8, n_informative=4,
                             random_state=42)
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
    
    def test_initialization(self, rf_classifier):
        """Test initialization."""
        explainer = PartialDependenceExplainer(
            rf_classifier,
            grid_resolution=50,
            percentile_range=(0.1, 0.9)
        )
        assert explainer.grid_resolution == 50
        assert explainer.percentile_range == (0.1, 0.9)
    
    def test_invalid_percentile_range(self, rf_classifier):
        """Test invalid percentile range."""
        with pytest.raises(ValidationError, match="Percentile range"):
            PartialDependenceExplainer(rf_classifier, percentile_range=(0.9, 0.1))
    
    def test_model_validation_no_predict(self):
        """Test model validation fails without predict method."""
        model = Mock(spec=[])
        # No predict method

        with pytest.raises(ModelError, match="predict"):
            PartialDependenceExplainer(model)
    
    def test_explain_global_single_feature_default(self, rf_classifier, classification_data):
        """Test explaining single feature (default to first)."""
        X, y = classification_data
        
        explainer = PartialDependenceExplainer(rf_classifier, grid_resolution=20)
        result = explainer.explain_global(X)
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'PartialDependence'
        assert result.explanation_type == 'global'
        assert len(result.feature_names) == 1
        assert result.feature_names[0] == 'feature_0'
        
        # Check metadata
        assert 'grid' in result.metadata
        assert 'feature_indices' in result.metadata
        assert 'is_2d' in result.metadata
        assert not result.metadata['is_2d']
    
    def test_explain_global_single_feature_by_index(self, rf_classifier, classification_data):
        """Test explaining single feature by index."""
        X, y = classification_data
        
        explainer = PartialDependenceExplainer(rf_classifier)
        result = explainer.explain_global(X, features=3)
        
        assert len(result.feature_names) == 1
        assert result.feature_names[0] == 'feature_3'
        assert result.metadata['feature_indices'] == [3]
    
    def test_explain_global_single_feature_by_name(self, rf_classifier, classification_data):
        """Test explaining single feature by name with pandas."""
        X, y = classification_data
        feature_names = [f'col_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        explainer = PartialDependenceExplainer(rf_classifier)
        result = explainer.explain_global(X_df, features='col_2')
        
        assert result.feature_names == ['col_2']
        assert result.metadata['feature_indices'] == [2]
    
    def test_explain_global_two_features(self, rf_classifier, classification_data):
        """Test explaining two features."""
        X, y = classification_data
        
        explainer = PartialDependenceExplainer(rf_classifier, grid_resolution=10)
        result = explainer.explain_global(X, features=[0, 1])
        
        assert len(result.feature_names) == 2
        assert result.metadata['is_2d'] == True
        
        # Check 2D grid shape
        grid_0, grid_1 = result.metadata['grid']
        expected_shape = (len(grid_0), len(grid_1))
        assert result.attributions.shape == expected_shape
    
    def test_explain_global_too_many_features(self, rf_classifier, classification_data):
        """Test error with more than 2 features."""
        X, y = classification_data
        
        explainer = PartialDependenceExplainer(rf_classifier)
        
        with pytest.raises(ValidationError, match="at most 2 features"):
            explainer.explain_global(X, features=[0, 1, 2])
    
    def test_invalid_feature_index(self, rf_classifier, classification_data):
        """Test invalid feature index."""
        X, y = classification_data
        
        explainer = PartialDependenceExplainer(rf_classifier)
        
        with pytest.raises(ValidationError, match="Feature index .* out of range"):
            explainer.explain_global(X, features=999)
    
    def test_invalid_feature_name(self, rf_classifier, classification_data):
        """Test invalid feature name."""
        X, y = classification_data
        X_df = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        explainer = PartialDependenceExplainer(rf_classifier)
        
        with pytest.raises(ValidationError, match="Feature .* not found"):
            explainer.explain_global(X_df, features='nonexistent_col')
    
    def test_create_grid_continuous(self, rf_classifier):
        """Test grid creation for continuous feature."""
        X = np.random.randn(100, 3)
        
        explainer = PartialDependenceExplainer(rf_classifier, grid_resolution=20)
        grid = explainer._create_grid(X, feature_idx=0)
        
        assert len(grid) == 20
        assert grid[0] >= np.percentile(X[:, 0], 5)
        assert grid[-1] <= np.percentile(X[:, 0], 95)
    
    def test_create_grid_categorical(self, rf_classifier):
        """Test grid creation for categorical-like feature."""
        X = np.random.choice([0, 1, 2, 3, 4], size=(100, 3))
        
        explainer = PartialDependenceExplainer(rf_classifier)
        grid = explainer._create_grid(X, feature_idx=0)
        
        # Should use unique values for categorical
        assert len(grid) <= 5  # At most 5 unique values
        assert all(val in [0, 1, 2, 3, 4] for val in grid)
    
    @patch('matplotlib.pyplot')
    def test_plot_partial_dependence_1d(self, mock_plt, rf_classifier, classification_data):
        """Test plotting 1D partial dependence."""
        X, y = classification_data
        
        explainer = PartialDependenceExplainer(rf_classifier, grid_resolution=10)
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig = explainer.plot_partial_dependence(X, features=0)
        
        assert fig == mock_fig
        mock_plt.subplots.assert_called_once()
        mock_ax.plot.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
    
    @patch('matplotlib.pyplot')
    def test_plot_partial_dependence_2d(self, mock_plt, rf_classifier, classification_data):
        """Test plotting 2D partial dependence."""
        X, y = classification_data
        
        explainer = PartialDependenceExplainer(rf_classifier, grid_resolution=5)
        
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_contour = Mock()
        mock_ax.contourf.return_value = mock_contour
        
        fig = explainer.plot_partial_dependence(X, features=[0, 1])
        
        assert fig == mock_fig
        mock_ax.contourf.assert_called_once()
        mock_ax.contour.assert_called_once()
        mock_plt.colorbar.assert_called_once()
    
    def test_plot_without_matplotlib(self, rf_classifier, classification_data):
        """Test plotting without matplotlib."""
        X, y = classification_data
        
        with patch('pyinterpret.global_.partial_dependence.plt', None):
            explainer = PartialDependenceExplainer(rf_classifier)
            
            with pytest.raises(ExplainerError, match="Matplotlib is required"):
                explainer.plot_partial_dependence(X, features=0)
    
    def test_regression_model(self, rf_regressor, regression_data):
        """Test with regression model."""
        X, y = regression_data
        
        explainer = PartialDependenceExplainer(rf_regressor, grid_resolution=15)
        result = explainer.explain_global(X, features=0)
        
        assert isinstance(result, ExplanationResult)
        assert len(result.attributions) == 15  # grid_resolution
    
    def test_parse_features_mixed_types(self, rf_classifier, classification_data):
        """Test parsing mixed feature types."""
        X, y = classification_data
        X_df = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
        
        explainer = PartialDependenceExplainer(rf_classifier)
        
        # Mix of index and name
        indices, names = explainer._parse_features([0, 'col_1'], X_df.columns.tolist())
        
        assert indices == [0, 1]
        assert names == ['col_0', 'col_1']


if __name__ == "__main__":
    pytest.main([__file__])
