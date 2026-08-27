# PyInterpret: A Unified Python Library for Machine Learning Model Interpretation

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyInterpret is a comprehensive Python library that unifies fragmented explainability tools under one consistent API. It provides modular coverage of both global and local explanations across different data modalities, consolidating capabilities from state-of-the-art tools like SHAP, LIME, and others.

## 🎯 Key Features

- **Unified API**: Consistent interface across all interpretation methods
- **Local Attribution**: SHAP, LIME, and other instance-level explanations
- **Global Insights**: Permutation importance, partial dependence plots
- **Modular Architecture**: Easy extension and customization
- **Multiple Data Types**: Support for tabular, text, image, and time-series data
- **Framework Integration**: Works seamlessly with scikit-learn, pandas, and other ML libraries
- **Professional Quality**: Comprehensive testing, documentation, and error handling

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install pyinterpret

# With SHAP support
pip install pyinterpret[shap]

# With LIME support  
pip install pyinterpret[lime]

# With all optional dependencies
pip install pyinterpret[all]
```

### Basic Usage

```python
from pyinterpret import SHAPExplainer, LIMEExplainer, PermutationImportanceExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd

# Create sample data and train model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

model = RandomForestClassifier(random_state=42)
model.fit(X_df, y)

# Local explanation with SHAP
shap_explainer = SHAPExplainer(model, explainer_type='tree')
shap_result = shap_explainer.explain_instance(X_df.iloc[0])

print("SHAP attributions:", shap_result.attributions)
print("Feature names:", shap_result.feature_names)

# Global explanation with Permutation Importance
perm_explainer = PermutationImportanceExplainer(model, scoring='accuracy')
perm_result = perm_explainer.explain_global(X_df, y)

print("Most important features:", perm_result.feature_names[:5])
print("Importance scores:", perm_result.attributions[:5])
```
