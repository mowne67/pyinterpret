PyInterpret Documentation
=========================

**PyInterpret** is a unified Python library for machine learning model interpretation that consolidates fragmented explainability tools under one consistent API.

The library provides comprehensive, modular coverage of global and local explanations across tabular, text, image, and time-series data, unifying capabilities found in state-of-the-art tools like SHAP, LIME, and others.

Features
--------

* **Unified API**: Consistent interface for all interpretation methods
* **Local Attribution**: SHAP, LIME, and other instance-level explanations  
* **Global Insights**: Permutation importance, partial dependence plots
* **Modular Architecture**: Easy extension and customization
* **Multiple Data Types**: Support for tabular, text, image, and time-series
* **Framework Integration**: Works with scikit-learn, pandas, and other popular ML libraries

Quick Start
-----------

Install PyInterpret:

.. code-block:: bash

   pip install pyinterpret

Basic usage example:

.. code-block:: python

   import numpy as np
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import make_classification
   from pyinterpret import SHAPExplainer, LIMEExplainer

   # Create data and train model
   X, y = make_classification(n_samples=100, n_features=10, random_state=42)
   model = RandomForestClassifier(random_state=42)
   model.fit(X, y)

   # SHAP explanation
   shap_explainer = SHAPExplainer(model)
   result = shap_explainer.explain_instance(X[0])
   
   print(f"SHAP attributions: {result.attributions}")
   print(f"Feature importance: {result.feature_names}")

   # LIME explanation
   lime_explainer = LIMEExplainer(model, training_data=X)
   lime_result = lime_explainer.explain_instance(X[0])

Core Components
---------------

**Explainer Types**

* **Local Explainers**: Explain individual predictions

  * ``SHAPExplainer``: Shapley value-based attributions
  * ``LIMEExplainer``: Local linear approximations

* **Global Explainers**: Explain overall model behavior

  * ``PermutationImportanceExplainer``: Feature importance via permutation
  * ``PartialDependenceExplainer``: Marginal feature effects

**Data Handling**

* ``TabularData``: Unified interface for tabular data with preprocessing
* ``ExplanationResult``: Standardized format for explanation outputs

**Utilities**

* Validation functions for models and data
* Visualization tools for explanation results
* Integration helpers for popular ML frameworks

Examples
--------

**Local Explanation Example**

.. code-block:: python

   from pyinterpret import SHAPExplainer
   from sklearn.ensemble import RandomForestClassifier
   
   # Train your model
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   
   # Create explainer
   explainer = SHAPExplainer(model)
   
   # Explain a single instance
   result = explainer.explain_instance(X_test[0])
   
   # Access results
   print("Feature contributions:")
   for name, attribution in zip(result.feature_names, result.attributions):
       print(f"{name}: {attribution:.3f}")

**Global Explanation Example**

.. code-block:: python

   from pyinterpret import PermutationImportanceExplainer
   
   # Create global explainer
   explainer = PermutationImportanceExplainer(model, scoring='accuracy')
   
   # Get feature importance
   result = explainer.explain_global(X_test, y_test)
   
   # Get feature ranking
   ranking = explainer.get_feature_ranking(X_test, y_test, top_k=10)
   
   for item in ranking['ranking']:
       print(f"{item['feature']}: {item['importance']:.3f}")

Architecture
------------

PyInterpret follows a modular architecture with clear separation of concerns:

**Core Layer**
  Base classes and interfaces (``BaseExplainer``, ``ExplanationResult``)

**Explainer Layer**  
  Specific implementation of interpretation methods

**Data Layer**
  Data handling and preprocessing utilities

**Utils Layer**
  Validation, visualization, and helper functions

This design ensures:

* **Consistency**: All explainers follow the same interface
* **Extensibility**: Easy to add new interpretation methods
* **Flexibility**: Support for different data types and model frameworks
* **Maintainability**: Clean separation of functionality

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples/index
   user_guide/index
   developer_guide/index

API Reference
=============

.. toctree::
   :maxdepth: 3
   
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Contributing
============

We welcome contributions! Please see our contributing guidelines for details on:

* Setting up the development environment
* Running tests
* Submitting pull requests
* Code style and documentation standards

License
=======

PyInterpret is released under the MIT License. See the LICENSE file for details.

Support
=======

* **Documentation**: https://pyinterpret.readthedocs.io/
* **Issues**: https://github.com/pyinterpret/pyinterpret/issues
* **Discussions**: https://github.com/pyinterpret/pyinterpret/discussions

Citation
========

If you use PyInterpret in your research, please cite:

.. code-block:: bibtex

   @software{pyinterpret,
     title={PyInterpret: A Unified Python Library for Machine Learning Model Interpretation},
     author={PyInterpret Team},
     year={2025},
     url={https://github.com/pyinterpret/pyinterpret}
   }
