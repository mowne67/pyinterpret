"""
Global (model-level) explainers for PyInterpret.

This module contains explainers that provide model-level insights and
global feature importance measures.
"""

from .permutation_importance import PermutationImportanceExplainer
from .partial_dependence import PartialDependenceExplainer

__all__ = ["PermutationImportanceExplainer", "PartialDependenceExplainer"]
