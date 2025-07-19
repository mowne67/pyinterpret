"""
Visualization utilities for PyInterpret.

This module provides common plotting functions for displaying
explanation results in a consistent and informative manner.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from pyinterpret.core.base import ExplanationResult
from pyinterpret.core.exceptions import ExplainerError


def plot_attributions(
    result: ExplanationResult,
    top_k: Optional[int] = None,
    show_values: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Plot feature attributions from an explanation result.
    
    Args:
        result: ExplanationResult containing attributions
        top_k: Number of top features to show (None for all)
        show_values: Whether to show attribution values
        title: Plot title
        **kwargs: Additional matplotlib parameters
        
    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ExplainerError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    if result.attributions is None:
        raise ExplainerError("No attributions found in explanation result")
    
    # Prepare data
    attributions = result.attributions
    feature_names = result.feature_names or [f"feature_{i}" for i in range(len(attributions))]
    
    # Handle different attribution formats
    if isinstance(attributions, pd.DataFrame):
        attributions = attributions.values.flatten()
    elif hasattr(attributions, 'shape') and len(attributions.shape) > 1:
        attributions = attributions.flatten()
    
    # Select top k features
    if top_k is not None and top_k < len(attributions):
        top_indices = np.argsort(np.abs(attributions))[-top_k:]
        attributions = attributions[top_indices]
        feature_names = [feature_names[i] for i in top_indices]
    
    # Sort by attribution magnitude
    sorted_indices = np.argsort(attributions)
    attributions = attributions[sorted_indices]
    feature_names = [feature_names[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    # Color bars based on positive/negative values
    colors = ['red' if val < 0 else 'blue' for val in attributions]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(attributions)), attributions, color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Attribution')
    
    if title is None:
        title = f'{result.method} Attributions'
    ax.set_title(title)
    
    # Add value labels if requested
    if show_values:
        for i, (bar, val) in enumerate(zip(bars, attributions)):
            label_x = val + (0.01 * max(abs(attributions)) if val >= 0 else -0.01 * max(abs(attributions)))
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    result: ExplanationResult,
    top_k: Optional[int] = 15,
    show_std: bool = True,
    title: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Plot feature importance with optional error bars.
    
    Args:
        result: ExplanationResult containing importance scores
        top_k: Number of top features to show
        show_std: Whether to show standard deviation as error bars
        title: Plot title
        **kwargs: Additional matplotlib parameters
        
    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ExplainerError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    if result.attributions is None:
        raise ExplainerError("No attributions found in explanation result")
    
    # Prepare data
    importance = result.attributions
    feature_names = result.feature_names or [f"feature_{i}" for i in range(len(importance))]
    
    # Get standard deviation if available
    std_values = None
    if show_std and 'std_importance' in result.metadata:
        std_values = result.metadata['std_importance']
    
    # Select top k features
    if top_k is not None and top_k < len(importance):
        top_indices = np.argsort(importance)[-top_k:]
        importance = importance[top_indices]
        feature_names = [feature_names[i] for i in top_indices]
        if std_values is not None:
            std_values = std_values[top_indices]
    
    # Sort by importance
    sorted_indices = np.argsort(importance)
    importance = importance[sorted_indices]
    feature_names = [feature_names[i] for i in sorted_indices]
    if std_values is not None:
        std_values = std_values[sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    # Create horizontal bar plot with error bars
    y_pos = np.arange(len(feature_names))
    bars = ax.barh(y_pos, importance, xerr=std_values if show_std else None,
                   capsize=3, alpha=0.7, color='steelblue')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Importance')
    
    if title is None:
        title = f'{result.method} Feature Importance'
    ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_local_explanation(
    result: ExplanationResult,
    instance_values: Optional[np.ndarray] = None,
    reference_values: Optional[np.ndarray] = None,
    max_features: int = 10,
    **kwargs
) -> Any:
    """
    Plot local explanation with instance values and reference comparison.
    
    Args:
        result: ExplanationResult containing local explanations
        instance_values: Values of the explained instance
        reference_values: Reference values for comparison
        max_features: Maximum number of features to show
        **kwargs: Additional matplotlib parameters
        
    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ExplainerError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    if result.explanation_type != 'local':
        raise ExplainerError("This function is only for local explanations")
    
    # Use instance values from result if not provided
    if instance_values is None:
        instance_values = result.feature_values
    
    # Prepare data
    attributions = result.attributions
    feature_names = result.feature_names or [f"feature_{i}" for i in range(len(attributions))]
    
    # Select top features by attribution magnitude
    top_indices = np.argsort(np.abs(attributions))[-max_features:]
    attributions = attributions[top_indices]
    feature_names = [feature_names[i] for i in top_indices]
    
    if instance_values is not None:
        instance_values = instance_values[top_indices]
    if reference_values is not None:
        reference_values = reference_values[top_indices]
    
    # Sort by attribution value
    sorted_indices = np.argsort(attributions)
    attributions = attributions[sorted_indices]
    feature_names = [feature_names[i] for i in sorted_indices]
    if instance_values is not None:
        instance_values = instance_values[sorted_indices]
    if reference_values is not None:
        reference_values = reference_values[sorted_indices]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=kwargs.get('figsize', (15, 6)))
    
    # Plot attributions
    colors = ['red' if val < 0 else 'blue' for val in attributions]
    y_pos = np.arange(len(feature_names))
    
    ax1.barh(y_pos, attributions, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Attribution')
    ax1.set_title(f'{result.method} Attributions')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Plot feature values
    if instance_values is not None:
        ax2.barh(y_pos, instance_values, alpha=0.7, color='green', label='Instance')
        
        if reference_values is not None:
            ax2.barh(y_pos, reference_values, alpha=0.5, color='gray', label='Reference')
            ax2.legend()
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(feature_names)
        ax2.set_xlabel('Feature Value')
        ax2.set_title('Feature Values')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No feature values available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Feature Values')
    
    plt.tight_layout()
    return fig


def plot_waterfall(
    result: ExplanationResult,
    max_features: int = 10,
    **kwargs
) -> Any:
    """
    Create a waterfall plot showing how features contribute to the prediction.
    
    Args:
        result: ExplanationResult containing attributions
        max_features: Maximum number of features to show
        **kwargs: Additional matplotlib parameters
        
    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ExplainerError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    # Prepare data
    attributions = result.attributions
    feature_names = result.feature_names or [f"feature_{i}" for i in range(len(attributions))]
    baseline = result.baseline or 0.0
    
    # Select top features by magnitude
    top_indices = np.argsort(np.abs(attributions))[-max_features:]
    attributions = attributions[top_indices]
    feature_names = [feature_names[i] for i in top_indices]
    
    # Sort by attribution value
    sorted_indices = np.argsort(attributions)
    attributions = attributions[sorted_indices]
    feature_names = [feature_names[i] for i in sorted_indices]
    
    # Calculate cumulative values
    cumulative = np.cumsum(np.concatenate([[baseline], attributions]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))
    
    # Plot baseline
    ax.bar(0, baseline, color='gray', alpha=0.7, label='Baseline')
    
    # Plot waterfall
    for i, (attr, name) in enumerate(zip(attributions, feature_names)):
        color = 'blue' if attr > 0 else 'red'
        bottom = cumulative[i]
        ax.bar(i + 1, attr, bottom=bottom, color=color, alpha=0.7)
        
        # Add connection lines
        if i < len(attributions) - 1:
            ax.plot([i + 1.4, i + 1.6], [cumulative[i + 1], cumulative[i + 1]], 
                   'k--', alpha=0.5)
    
    # Plot final prediction
    final_pred = cumulative[-1]
    ax.bar(len(attributions) + 1, final_pred, color='green', alpha=0.7, label='Prediction')
    
    # Customize plot
    labels = ['Baseline'] + feature_names + ['Prediction']
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title(f'{result.method} Waterfall Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_plot(
    results: List[ExplanationResult],
    method_names: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    Create a summary plot comparing multiple explanation methods.
    
    Args:
        results: List of ExplanationResults to compare
        method_names: Names for each method (uses result.method if None)
        **kwargs: Additional matplotlib parameters
        
    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ExplainerError(
            "Matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    
    if not results:
        raise ExplainerError("No results provided for summary plot")
    
    # Prepare method names
    if method_names is None:
        method_names = [result.method for result in results]
    
    if len(method_names) != len(results):
        raise ExplainerError("Number of method names must match number of results")
    
    # Get common features
    all_features = set()
    for result in results:
        if result.feature_names:
            all_features.update(result.feature_names)
    
    if not all_features:
        raise ExplainerError("No feature names found in results")
    
    common_features = sorted(list(all_features))
    
    # Create comparison matrix
    comparison_matrix = np.zeros((len(results), len(common_features)))
    
    for i, result in enumerate(results):
        if result.attributions is not None and result.feature_names is not None:
            for j, feature in enumerate(common_features):
                if feature in result.feature_names:
                    feature_idx = result.feature_names.index(feature)
                    comparison_matrix[i, j] = result.attributions[feature_idx]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
    
    im = ax.imshow(comparison_matrix, cmap='RdBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(common_features)))
    ax.set_xticklabels(common_features, rotation=45, ha='right')
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Attribution')
    
    ax.set_title('Method Comparison Heatmap')
    plt.tight_layout()
    return fig
