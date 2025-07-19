"""
PyInterpret: A unified Python library for machine learning model interpretation.

This library provides a consistent API for various model explainability techniques,
supporting both local and global interpretation methods across different data modalities.
"""

__version__ = "0.1.0"
__author__ = "PyInterpret Team"
__license__ = "MIT"

from pyinterpret.core.base import BaseExplainer, ExplanationResult
from pyinterpret.local.shap_explainer import SHAPExplainer
from pyinterpret.local.lime_explainer import LIMEExplainer
from pyinterpret.global_.permutation_importance import PermutationImportanceExplainer
from pyinterpret.global_.partial_dependence import PartialDependenceExplainer
from pyinterpret.utils.validation import validate_model, validate_data
from pyinterpret.data.tabular import TabularData

__all__ = [
    "BaseExplainer",
    "ExplanationResult",
    "SHAPExplainer",
    "LIMEExplainer",
    "PermutationImportanceExplainer",
    "PartialDependenceExplainer",
    "TabularData",
    "validate_model",
    "validate_data"
]

# Version information
VERSION = __version__
