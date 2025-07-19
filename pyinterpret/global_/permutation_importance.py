"""
Permutation importance explainer implementation.

This module provides permutation-based feature importance calculation,
which measures the decrease in model performance when a feature's values
are randomly shuffled.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from sklearn.model_selection import cross_val_score

from pyinterpret.core.base import GlobalExplainer, ExplanationResult
from pyinterpret.core.exceptions import ModelError, ValidationError, ExplainerError
from pyinterpret.utils.validation import validate_data


class PermutationImportanceExplainer(GlobalExplainer):
    """
    Permutation importance explainer for measuring feature importance.
    
    This explainer calculates feature importance by measuring how much
    the model performance decreases when each feature is randomly shuffled,
    breaking the relationship between the feature and the target.
    """
    
    def __init__(
        self,
        model: Any,
        scoring: Union[str, Callable] = 'auto',
        n_repeats: int = 5,
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize permutation importance explainer.
        
        Args:
            model: The machine learning model to explain
            scoring: Scoring function or string ('auto', 'accuracy', 'mse', 'rmse', 'mae', 'r2')
            n_repeats: Number of times to permute each feature
            random_state: Random state for reproducibility
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Determine scoring function
        self._setup_scoring()
    
    def _validate_model(self) -> None:
        """Validate that the model is compatible with permutation importance."""
        if not hasattr(self.model, 'predict'):
            raise ModelError(
                "Model must have a 'predict' method",
                model_type=type(self.model).__name__,
                required_methods=['predict']
            )
    
    def _setup_scoring(self) -> None:
        """Setup the scoring function based on the model type."""
        if self.scoring == 'auto':
            # Auto-detect based on model type
            if hasattr(self.model, 'predict_proba'):
                self.scoring_func = self._accuracy_score
                self.scoring_name = 'accuracy'
            else:
                self.scoring_func = self._neg_mse_score
                self.scoring_name = 'neg_mse'
        elif isinstance(self.scoring, str):
            self.scoring_func = self._get_scoring_function(self.scoring)
            self.scoring_name = self.scoring
        elif callable(self.scoring):
            self.scoring_func = self.scoring
            self.scoring_name = 'custom'
        else:
            raise ValidationError(
                f"Invalid scoring parameter: {self.scoring}",
                parameter="scoring",
                expected="string or callable"
            )
    
    def _get_scoring_function(self, scoring: str) -> Callable:
        """Get scoring function by name."""
        scoring_functions = {
            'accuracy': self._accuracy_score,
            'mse': self._mse_score,
            'neg_mse': self._neg_mse_score,
            'rmse': self._rmse_score,
            'mae': self._mae_score,
            'r2': self._r2_score
        }
        
        if scoring not in scoring_functions:
            raise ValidationError(
                f"Unsupported scoring function: {scoring}",
                parameter="scoring",
                expected=list(scoring_functions.keys())
            )
        
        return scoring_functions[scoring]
    
    def _accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return accuracy_score(y_true, y_pred)
    
    def _mse_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error."""
        return mean_squared_error(y_true, y_pred)
    
    def _neg_mse_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate negative mean squared error (higher is better)."""
        return -mean_squared_error(y_true, y_pred)
    
    def _rmse_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate root mean squared error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _mae_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean absolute error."""
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y_true, y_pred)
    
    def _r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score."""
        from sklearn.metrics import r2_score
        return r2_score(y_true, y_pred)
    
    def explain_global(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        **kwargs
    ) -> ExplanationResult:
        """
        Calculate permutation importance for all features.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional parameters
            
        Returns:
            ExplanationResult containing permutation importance scores
        """
        # Validate inputs
        X = validate_data(X)
        y = validate_data(y, expected_dims=1)
        
        if len(X) != len(y):
            raise ValidationError(
                f"X and y must have the same length. Got {len(X)} and {len(y)}",
                parameter="X, y"
            )
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        # Calculate baseline score
        try:
            baseline_predictions = self.model.predict(X_array)
            baseline_score = self.scoring_func(y, baseline_predictions)
        except Exception as e:
            raise ExplainerError(
                f"Failed to calculate baseline score: {str(e)}",
                explainer="PermutationImportanceExplainer"
            )
        
        # Calculate permutation importance for each feature
        n_features = X_array.shape[1]
        importance_scores = np.zeros((n_features, self.n_repeats))
        
        for feature_idx in range(n_features):
            for repeat in range(self.n_repeats):
                # Create permuted copy of X
                X_permuted = X_array.copy()
                
                # Permute the current feature
                permuted_indices = self.rng.permutation(len(X_array))
                X_permuted[:, feature_idx] = X_array[permuted_indices, feature_idx]
                
                try:
                    # Calculate score with permuted feature
                    permuted_predictions = self.model.predict(X_permuted)
                    permuted_score = self.scoring_func(y, permuted_predictions)
                    
                    # Importance is the decrease in score
                    importance_scores[feature_idx, repeat] = baseline_score - permuted_score
                    
                except Exception as e:
                    raise ExplainerError(
                        f"Failed to calculate permuted score for feature {feature_idx}: {str(e)}",
                        explainer="PermutationImportanceExplainer"
                    )
        
        # Calculate statistics
        mean_importance = np.mean(importance_scores, axis=1)
        std_importance = np.std(importance_scores, axis=1)
        
        return ExplanationResult(
            attributions=mean_importance,
            feature_names=feature_names,
            method='PermutationImportance',
            explanation_type='global',
            metadata={
                'baseline_score': baseline_score,
                'scoring': self.scoring_name,
                'n_repeats': self.n_repeats,
                'std_importance': std_importance,
                'raw_scores': importance_scores,
                'sample_size': len(X)
            }
        )
    
    def get_feature_ranking(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get feature ranking based on permutation importance.
        
        Args:
            X: Input features
            y: Target values
            top_k: Number of top features to return (None for all)
            
        Returns:
            Dictionary containing ranked features and their importance scores
        """
        result = self.explain_global(X, y)
        
        # Create ranking
        importance_indices = np.argsort(result.attributions)[::-1]
        
        if top_k is not None:
            importance_indices = importance_indices[:top_k]
        
        ranked_features = []
        for idx in importance_indices:
            ranked_features.append({
                'feature': result.feature_names[idx],
                'importance': result.attributions[idx],
                'std': result.metadata['std_importance'][idx],
                'rank': len(ranked_features) + 1
            })
        
        return {
            'ranking': ranked_features,
            'baseline_score': result.metadata['baseline_score'],
            'scoring': result.metadata['scoring'],
            'total_features': len(result.feature_names)
        }
    
    def plot_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        top_k: int = 10,
        **plot_kwargs
    ) -> Any:
        """
        Plot permutation importance scores.
        
        Args:
            X: Input features
            y: Target values
            top_k: Number of top features to plot
            **plot_kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ExplainerError(
                "Matplotlib is required for plotting. Install with: pip install matplotlib",
                explainer="PermutationImportanceExplainer"
            )
        
        result = self.explain_global(X, y)
        
        # Get top k features
        top_indices = np.argsort(result.attributions)[::-1][:top_k]
        top_features = [result.feature_names[i] for i in top_indices]
        top_importance = result.attributions[top_indices]
        top_std = result.metadata['std_importance'][top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=plot_kwargs.get('figsize', (10, 6)))
        
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_importance, xerr=top_std, 
                      capsize=3, **{k: v for k, v in plot_kwargs.items() if k != 'figsize'})
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel(f'Permutation Importance ({result.metadata["scoring"]})')
        ax.set_title('Feature Importance (Permutation)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
