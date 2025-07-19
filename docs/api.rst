API Reference
=============

This page contains the API reference for all PyInterpret modules and classes.

Core Components
---------------

Base Classes
~~~~~~~~~~~~

.. automodule:: pyinterpret.core.base
    :members:
    :undoc-members:
    :show-inheritance:

Exceptions
~~~~~~~~~~

.. automodule:: pyinterpret.core.exceptions
    :members:
    :undoc-members:
    :show-inheritance:

Local Explainers
----------------

SHAP Explainer
~~~~~~~~~~~~~~

.. automodule:: pyinterpret.local.shap_explainer
    :members:
    :undoc-members:
    :show-inheritance:

LIME Explainer
~~~~~~~~~~~~~~

.. automodule:: pyinterpret.local.lime_explainer
    :members:
    :undoc-members:
    :show-inheritance:

Global Explainers
-----------------

Permutation Importance
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pyinterpret.global_.permutation_importance
    :members:
    :undoc-members:
    :show-inheritance:

Partial Dependence
~~~~~~~~~~~~~~~~~~

.. automodule:: pyinterpret.global_.partial_dependence
    :members:
    :undoc-members:
    :show-inheritance:

Data Handling
-------------

Tabular Data
~~~~~~~~~~~~

.. automodule:: pyinterpret.data.tabular
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
---------

Validation
~~~~~~~~~~

.. automodule:: pyinterpret.utils.validation
    :members:
    :undoc-members:
    :show-inheritance:

Visualization
~~~~~~~~~~~~~

.. automodule:: pyinterpret.utils.visualization
    :members:
    :undoc-members:
    :show-inheritance:

Complete API Index
------------------

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pyinterpret.core.base.BaseExplainer
   pyinterpret.core.base.LocalExplainer
   pyinterpret.core.base.GlobalExplainer
   pyinterpret.core.base.ExplanationResult
   pyinterpret.local.shap_explainer.SHAPExplainer
   pyinterpret.local.lime_explainer.LIMEExplainer
   pyinterpret.global_.permutation_importance.PermutationImportanceExplainer
   pyinterpret.global_.partial_dependence.PartialDependenceExplainer
   pyinterpret.data.tabular.TabularData

Exceptions
~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: exception.rst

   pyinterpret.core.exceptions.PyInterpretError
   pyinterpret.core.exceptions.ValidationError
   pyinterpret.core.exceptions.ModelError
   pyinterpret.core.exceptions.ExplainerError
   pyinterpret.core.exceptions.DataError
   pyinterpret.core.exceptions.ConfigurationError

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pyinterpret.utils.validation.validate_model
   pyinterpret.utils.validation.validate_data
   pyinterpret.utils.validation.validate_features
   pyinterpret.utils.validation.validate_target
   pyinterpret.utils.validation.validate_explanation_result
   pyinterpret.utils.validation.check_sklearn_compatibility
   pyinterpret.utils.visualization.plot_attributions
   pyinterpret.utils.visualization.plot_feature_importance
   pyinterpret.utils.visualization.plot_local_explanation
   pyinterpret.utils.visualization.plot_waterfall
   pyinterpret.utils.visualization.create_summary_plot

Usage Patterns
---------------

Basic Workflow
~~~~~~~~~~~~~~

The typical PyInterpret workflow follows this pattern:

1. **Import** the required explainer classes
2. **Initialize** an explainer with your trained model
3. **Fit** the explainer (if required) with training data
4. **Explain** instances or generate global explanations
5. **Access** results through the ``ExplanationResult`` object

.. code-block:: python

   from pyinterpret import SHAPExplainer
   
   # Initialize explainer
   explainer = SHAPExplainer(model)
   
   # Fit if needed (automatic for tree models)
   explainer.fit(X_train)
   
   # Generate explanation
   result = explainer.explain_instance(x_instance)
   
   # Access results
   attributions = result.attributions
   feature_names = result.feature_names

Explanation Results
~~~~~~~~~~~~~~~~~~~

All explainers return results in a standardized ``ExplanationResult`` format:

.. code-block:: python

   result = explainer.explain_instance(instance)
   
   # Core explanation data
   result.attributions      # Feature attributions/importance scores
   result.feature_names     # Names of features
   result.feature_values    # Values of features for the instance
   
   # Method information
   result.method           # Name of explanation method
   result.explanation_type # 'local' or 'global'
   result.model_output     # Model's prediction
   result.baseline         # Baseline/reference value
   
   # Additional metadata
   result.metadata         # Method-specific additional information
   
   # Utility methods
   result.to_dict()        # Convert to dictionary

Error Handling
~~~~~~~~~~~~~~

PyInterpret provides specific exception types for clear error messages:

.. code-block:: python

   from pyinterpret.core.exceptions import ModelError, ValidationError
   
   try:
       explainer = SHAPExplainer(model)
   except ModelError as e:
       print(f"Model compatibility issue: {e}")
   except ValidationError as e:
       print(f"Input validation failed: {e}")

Extension Points
~~~~~~~~~~~~~~~~

The library is designed for easy extension:

.. code-block:: python

   from pyinterpret.core.base import LocalExplainer, ExplanationResult
   
   class CustomExplainer(LocalExplainer):
       def _validate_model(self):
           # Implement model validation
           pass
       
       def explain_instance(self, instance, **kwargs):
           # Implement explanation logic
           attributions = custom_explanation_logic(instance)
           
           return ExplanationResult(
               attributions=attributions,
               feature_names=self.get_feature_names(),
               method='CustomMethod',
               explanation_type='local'
           )

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

Most explainers accept configuration parameters during initialization:

.. code-block:: python

   # SHAP configuration
   shap_explainer = SHAPExplainer(
       model,
       explainer_type='tree',  # or 'linear', 'kernel'
       background_data=X_background
   )
   
   # LIME configuration
   lime_explainer = LIMEExplainer(
       model,
       mode='classification',
       num_features=10,
       num_samples=5000
   )
   
   # Permutation importance configuration
   perm_explainer = PermutationImportanceExplainer(
       model,
       scoring='accuracy',
       n_repeats=10,
       random_state=42
   )

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets and complex models:

* Use sampling for global explanations: ``X_sample = X[:1000]``
* Set appropriate ``num_samples`` for LIME: ``num_samples=1000``
* Use ``n_repeats=5`` for permutation importance instead of default
* Consider ``grid_resolution=20`` for partial dependence instead of 100

.. code-block:: python

   # Optimized for performance
   explainer = PermutationImportanceExplainer(
       model,
       n_repeats=5,  # Fewer repeats
       random_state=42
   )
   
   # Use subset of data
   sample_size = min(1000, len(X_test))
   X_sample = X_test[:sample_size]
   y_sample = y_test[:sample_size]
   
   result = explainer.explain_global(X_sample, y_sample)
