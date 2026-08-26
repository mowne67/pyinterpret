"""
Partial dependence explainer implementation.

This module provides partial dependence plot functionality to show
the marginal effect of features on the predicted outcome.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from itertools import product

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from pyinterpret.core.base import GlobalExplainer, ExplanationResult
from pyinterpret.core.exceptions import ModelError, ValidationError, ExplainerError
from pyinterpret.utils.validation import validate_data


class PartialDependenceExplainer(GlobalExplainer):
    """
    Partial dependence explainer for understanding feature effects.
    
    This explainer calculates partial dependence plots (PDPs) to show
    the marginal effect of one or more features on the predicted outcome
    of a machine learning model.
    """
    
    def __init__(
        self,
        model: Any,
        grid_resolution: int = 100,
        percentile_range: Tuple[float, float] = (0.05, 0.95),
        **kwargs
    ):
        """
        Initialize partial dependence explainer.
        
        Args:
            model: The machine learning model to explain
            grid_resolution: Number of points in the grid for each feature
            percentile_range: Range of percentiles to use for grid creation
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        
        self.grid_resolution = grid_resolution
        self.percentile_range = percentile_range
        
        if not (0 <= percentile_range[0] < percentile_range[1] <= 1):
            raise ValidationError(
                "Percentile range must be between 0 and 1 with min < max",
                parameter="percentile_range"
            )
    
    def _validate_model(self) -> None:
        """Validate that the model is compatible with partial dependence."""
        if not hasattr(self.model, 'predict'):
            raise ModelError(
                "Model must have a 'predict' method",
                model_type=type(self.model).__name__,
                required_methods=['predict']
            )
    
    def explain_global(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        features: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **kwargs
    ) -> ExplanationResult:
        """
        Calculate partial dependence for specified features.
        
        Args:
            X: Input dataset
            features: Feature(s) to calculate partial dependence for
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult containing partial dependence data
        """
        # Validate inputs
        X = validate_data(X)
        
        # Get feature names and indices
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        # Handle feature specification
        if features is None:
            # Default to first feature
            feature_indices = [0]
            selected_features = [feature_names[0]]
        else:
            feature_indices, selected_features = self._parse_features(features, feature_names)
        
        # Validate number of features
        if len(feature_indices) > 2:
            raise ValidationError(
                "Partial dependence supports at most 2 features at once",
                parameter="features"
            )
        
        # Calculate partial dependence
        if len(feature_indices) == 1:
            pd_result = self._calculate_1d_partial_dependence(
                X_array, feature_indices[0], **kwargs
            )
        else:
            pd_result = self._calculate_2d_partial_dependence(
                X_array, feature_indices, **kwargs
            )
        
        return ExplanationResult(
            attributions=pd_result['values'],
            feature_names=selected_features,
            method='PartialDependence',
            explanation_type='global',
            metadata={
                'partial_dependence_values': pd_result['values'],
                'grid': pd_result['grid'],
                'feature_indices': feature_indices,
                'grid_resolution': self.grid_resolution,
                'percentile_range': self.percentile_range,
                'sample_size': len(X),
                'is_2d': len(feature_indices) == 2
            }
        )
    
    def _parse_features(
        self, 
        features: Union[int, str, List[Union[int, str]]], 
        feature_names: List[str]
    ) -> Tuple[List[int], List[str]]:
        """Parse feature specification into indices and names."""
        if not isinstance(features, list):
            features = [features]
        
        feature_indices = []
        selected_features = []
        
        for feature in features:
            # Handle both regular int and numpy integer types
            if isinstance(feature, (int, np.integer)):
                feature_int = int(feature)  # Convert numpy int to regular int
                if 0 <= feature_int < len(feature_names):
                    feature_indices.append(feature_int)
                    selected_features.append(feature_names[feature_int])
                else:
                    raise ValidationError(
                        f"Feature index {feature_int} out of range [0, {len(feature_names)-1}]",
                        parameter="features"
                    )
            elif isinstance(feature, str):
                if feature in feature_names:
                    idx = feature_names.index(feature)
                    feature_indices.append(idx)
                    selected_features.append(feature)
                else:
                    raise ValidationError(
                        f"Feature '{feature}' not found in feature names",
                        parameter="features"
                    )
            else:
                raise ValidationError(
                    f"Feature must be int or str, got {type(feature)}",
                    parameter="features"
                )
        
        return feature_indices, selected_features
    
    def _create_grid(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        """Create grid values for a single feature."""
        feature_values = X[:, feature_idx]
        
        # Calculate percentile-based range
        min_val = np.percentile(feature_values, self.percentile_range[0] * 100)
        max_val = np.percentile(feature_values, self.percentile_range[1] * 100)
        
        # Create grid
        if len(np.unique(feature_values)) <= 10:
            # For categorical-like features, use unique values
            unique_vals = np.unique(feature_values)
            grid = unique_vals[(unique_vals >= min_val) & (unique_vals <= max_val)]
        else:
            # For continuous features, use linear grid
            grid = np.linspace(min_val, max_val, self.grid_resolution)
        
        return grid
    
    def _calculate_1d_partial_dependence(
        self, 
        X: np.ndarray, 
        feature_idx: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate 1D partial dependence for a single feature."""
        grid = self._create_grid(X, feature_idx)
        pd_values = np.zeros(len(grid))
        
        for i, grid_value in enumerate(grid):
            # Create modified dataset
            X_modified = X.copy()
            X_modified[:, feature_idx] = grid_value
            
            try:
                # Calculate predictions and average
                predictions = self.model.predict(X_modified)
                pd_values[i] = np.mean(predictions)
            except Exception as e:
                raise ExplainerError(
                    f"Failed to calculate predictions for grid value {grid_value}: {str(e)}",
                    explainer="PartialDependenceExplainer"
                )
        
        return {
            'values': pd_values,
            'grid': grid
        }
    
    def _calculate_2d_partial_dependence(
        self, 
        X: np.ndarray, 
        feature_indices: List[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate 2D partial dependence for two features."""
        grid_0 = self._create_grid(X, feature_indices[0])
        grid_1 = self._create_grid(X, feature_indices[1])
        
        pd_values = np.zeros((len(grid_0), len(grid_1)))
        
        for i, value_0 in enumerate(grid_0):
            for j, value_1 in enumerate(grid_1):
                # Create modified dataset
                X_modified = X.copy()
                X_modified[:, feature_indices[0]] = value_0
                X_modified[:, feature_indices[1]] = value_1
                
                try:
                    # Calculate predictions and average
                    predictions = self.model.predict(X_modified)
                    pd_values[i, j] = np.mean(predictions)
                except Exception as e:
                    raise ExplainerError(
                        f"Failed to calculate predictions for grid values ({value_0}, {value_1}): {str(e)}",
                        explainer="PartialDependenceExplainer"
                    )
        
        return {
            'values': pd_values,
            'grid': [grid_0, grid_1]
        }
    
    def plot_partial_dependence(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        features: Optional[Union[int, str, List[Union[int, str]]]] = None,
        **plot_kwargs
    ) -> Any:
        """
        Plot partial dependence.
        
        Args:
            X: Input dataset
            features: Feature(s) to plot
            **plot_kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure object
        """
        if plt is None:
            raise ExplainerError(
                "Matplotlib is required for plotting. Install with: pip install matplotlib",
                explainer="PartialDependenceExplainer"
            )

        result = self.explain_global(X, features)
        
        if result.metadata['is_2d']:
            return self._plot_2d_partial_dependence(result, **plot_kwargs)
        else:
            return self._plot_1d_partial_dependence(result, **plot_kwargs)
    
    def _plot_1d_partial_dependence(self, result: ExplanationResult, **plot_kwargs) -> Any:
        """Plot 1D partial dependence."""
        fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (8, 6)))
        
        grid = result.metadata['grid']
        values = result.attributions
        
        ax.plot(grid, values, linewidth=2, **{k: v for k, v in plot_kwargs.items() 
                                              if k not in ['figsize', 'title']})
        ax.set_xlabel(result.feature_names[0])
        ax.set_ylabel('Partial Dependence')
        ax.set_title(plot_kwargs.get('title', f'Partial Dependence: {result.feature_names[0]}'))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_2d_partial_dependence(self, result: ExplanationResult, **plot_kwargs) -> Any:
        """Plot 2D partial dependence."""
        fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (10, 8)))
        
        grid_0, grid_1 = result.metadata['grid']
        values = result.attributions
        
        # Create meshgrid for contour plot
        X_grid, Y_grid = np.meshgrid(grid_1, grid_0)
        
        contour = ax.contourf(X_grid, Y_grid, values, levels=20, alpha=0.8)
        ax.contour(X_grid, Y_grid, values, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        
        ax.set_xlabel(result.feature_names[1])
        ax.set_ylabel(result.feature_names[0])
        ax.set_title(plot_kwargs.get('title', 
                    f'Partial Dependence: {result.feature_names[0]} vs {result.feature_names[1]}'))
        
        # Add colorbar
        plt.colorbar(contour, ax=ax, label='Partial Dependence')
        
        plt.tight_layout()
        return fig
