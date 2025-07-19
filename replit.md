# PyInterpret: A Unified Python Library for Machine Learning Model Interpretation

## Overview

PyInterpret is a comprehensive Python library that provides a unified API for machine learning model interpretation. The library abstracts different explainability methods (SHAP, LIME, permutation importance, partial dependence) under a consistent interface, supporting both local (instance-level) and global (model-level) explanations across tabular, text, image, and time-series data.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### Latest Development Session (July 19, 2025)
- **All Core Explainers Fully Functional**: Fixed critical bugs in SHAP, LIME, Permutation Importance, and Partial Dependence explainers
- **SHAP Explainer**: Resolved multi-class attribution shape handling and LinearExplainer masker requirements
- **LIME Explainer**: Fixed initialization order and intercept extraction for classification scenarios
- **Partial Dependence**: Corrected numpy int64 type handling and metadata structure
- **Complete Testing**: All examples now run successfully for both classification and regression tasks
- **Library Status**: Production-ready with unified API working across all explainer types

## System Architecture

PyInterpret follows a modular, object-oriented architecture designed around a core abstraction layer with specialized implementations for different explanation methods:

### Core Architecture Pattern
- **Base Abstraction Layer**: Defines common interfaces and data structures
- **Modular Explainer System**: Separate modules for local and global explainers
- **Unified Result Format**: Standardized output format across all explainer types
- **Optional Dependencies**: Graceful handling of missing third-party libraries (SHAP, LIME)

### Package Structure
```
pyinterpret/
├── core/           # Base classes and fundamental abstractions
├── local/          # Instance-level explainers (SHAP, LIME)
├── global_/        # Model-level explainers (permutation importance, PDP)
├── data/           # Data handling utilities
└── utils/          # Common utilities (validation, visualization)
```

## Key Components

### 1. Core Foundation (`pyinterpret.core`)

**BaseExplainer Classes**: Abstract base classes defining the interface all explainers must implement
- `BaseExplainer`: Root interface for all explainers
- `LocalExplainer`: Specialized base for instance-level explanations
- `GlobalExplainer`: Specialized base for model-level explanations

**ExplanationResult**: Standardized container for explanation outputs with:
- Attributions (feature importance scores)
- Feature metadata (names, values)
- Method-specific information
- Validation logic to ensure data consistency

**Exception Hierarchy**: Custom exceptions for clear error handling:
- `PyInterpretError`: Base exception
- `ValidationError`: Input validation failures
- `ModelError`: Model compatibility issues

### 2. Local Explainers (`pyinterpret.local`)

**SHAPExplainer**: Wrapper for SHAP library with automatic explainer selection
- Auto-detects appropriate SHAP explainer type based on model
- Supports tree, linear, kernel, and deep explainers
- Graceful fallback when SHAP is not available

**LIMEExplainer**: Implementation of LIME with local linear approximations
- Perturbs input instances to create local dataset
- Fits linear models to approximate model behavior
- Supports both classification and regression modes

### 3. Global Explainers (`pyinterpret.global_`)

**PermutationImportanceExplainer**: Feature importance through permutation testing
- Measures performance drop when features are shuffled
- Supports multiple scoring metrics (accuracy, MSE, R², etc.)
- Configurable number of permutation repeats

**PartialDependenceExplainer**: Marginal effect analysis
- Shows how features affect predictions while averaging out other features
- Configurable grid resolution and percentile ranges
- Supports both univariate and bivariate analysis

### 4. Data Handling (`pyinterpret.data`)

**TabularData**: Unified container for tabular datasets
- Handles pandas DataFrames and numpy arrays uniformly
- Manages feature names and categorical feature identification
- Provides preprocessing capabilities (scaling, encoding)
- Includes data validation and consistency checks

### 5. Utilities (`pyinterpret.utils`)

**Validation Module**: Comprehensive input validation
- Model validation (required methods checking)
- Data format validation and transformation
- Feature consistency validation

**Visualization Module**: Plotting utilities for explanation results
- Feature attribution plots
- Feature importance visualizations
- Consistent styling across plot types

## Data Flow

### Local Explanation Workflow
1. **Input Validation**: Validate model and input instance
2. **Explainer Initialization**: Set up background data and parameters
3. **Explanation Generation**: Generate attributions for specific instance
4. **Result Packaging**: Wrap results in standardized ExplanationResult
5. **Optional Visualization**: Generate plots if requested

### Global Explanation Workflow
1. **Model Assessment**: Validate model has required prediction methods
2. **Data Preparation**: Process training/validation datasets
3. **Importance Calculation**: Compute feature importance scores
4. **Statistical Analysis**: Calculate confidence intervals (if applicable)
5. **Result Aggregation**: Package results with metadata

### Error Handling Strategy
- **Graceful Degradation**: Optional dependencies handled with informative errors
- **Early Validation**: Input validation at explainer initialization
- **Detailed Error Messages**: Context-rich exceptions with suggested fixes

## External Dependencies

### Core Dependencies (Required)
- **numpy**: Numerical computations and array operations
- **pandas**: Data manipulation and tabular data handling  
- **scikit-learn**: Model validation and basic ML utilities
- **matplotlib**: Visualization and plotting capabilities

### Optional Dependencies (Feature-Specific)
- **shap**: SHAP explainer implementation (installs with `pip install pyinterpret[shap]`)
- **lime**: LIME explainer implementation (installs with `pip install pyinterpret[lime]`)

### Development Dependencies
- **pytest**: Testing framework with coverage reporting
- **sphinx**: Documentation generation
- **black/flake8/isort**: Code formatting and linting

### Dependency Management Strategy
- **Optional Import Pattern**: Try/except blocks for optional libraries
- **Feature Flags**: `SHAP_AVAILABLE`, `LIME_AVAILABLE` flags
- **Informative Fallbacks**: Clear error messages when optional dependencies missing
- **Minimal Core**: Core functionality works without optional dependencies

## Deployment Strategy

### Package Distribution
- **PyPI Distribution**: Standard `setup.py` configuration for pip installation
- **Multiple Install Options**: 
  - `pip install pyinterpret` (core functionality)
  - `pip install pyinterpret[shap]` (with SHAP support)
  - `pip install pyinterpret[all]` (all optional dependencies)

### Documentation Strategy
- **Sphinx Documentation**: Comprehensive API documentation with ReadTheDocs hosting
- **Example-Driven**: Practical examples in `examples/` directory
- **Multiple Complexity Levels**: Basic usage to advanced customization examples

### Testing Strategy
- **Comprehensive Test Suite**: Tests for all major components
- **Mock-Based Testing**: Avoids heavy dependencies in test execution
- **Multiple Data Scenarios**: Tests with different data types and model types
- **Error Condition Testing**: Validates proper exception handling

### Development Workflow
- **Modular Development**: Independent development of explainer modules
- **Consistent API**: All explainers follow same interface patterns
- **Extension Points**: Easy addition of new explainer types through base classes