# PyInterpret: A Unified Python Library for Machine Learning Model Interpretation

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pyinterpret/badge/?version=latest)](https://pyinterpret.readthedocs.io/en/latest/?badge=latest)

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
pip install pyinterpret

# Or with optional dependencies
pip install pyinterpret[all]  # Includes SHAP, LIME, and all features
