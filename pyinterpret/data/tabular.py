"""
Tabular data handling utilities for PyInterpret.

This module provides a unified interface for working with tabular data
across different explainer methods.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from pyinterpret.core.exceptions import DataError, ValidationError
from pyinterpret.utils.validation import validate_data


class TabularData:
    """
    Container and preprocessor for tabular data.
    
    This class provides a unified interface for handling tabular data,
    including preprocessing, validation, and feature management.
    """
    
    def __init__(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[Union[int, str]]] = None,
        target_name: Optional[str] = None
    ):
        """
        Initialize TabularData container.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            feature_names: Names of features
            categorical_features: Indices or names of categorical features
            target_name: Name of target variable
        """
        # Validate and store data
        self.X = validate_data(X)
        self.y = None
        if y is not None:
            self.y = validate_data(y, expected_dims=1)
            if len(self.X) != len(self.y):
                raise DataError(
                    f"X and y must have same length. Got {len(self.X)} and {len(self.y)}"
                )
        
        # Handle feature names
        if isinstance(self.X, pd.DataFrame):
            self.feature_names = list(self.X.columns)
            self._X_array = self.X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(self.X.shape[1])]
            self._X_array = self.X
        
        # Handle target name
        if isinstance(y, pd.Series):
            self.target_name = target_name or y.name or "target"
            self._y_array = y.values
        else:
            self.target_name = target_name or "target"
            self._y_array = self.y
        
        # Handle categorical features
        self.categorical_features = self._parse_categorical_features(categorical_features)
        
        # Store original data types
        self.dtypes = self._infer_dtypes()
        
        # Preprocessing utilities
        self.scalers = {}
        self.encoders = {}
        self._is_preprocessed = False
    
    def _parse_categorical_features(
        self, 
        categorical_features: Optional[List[Union[int, str]]]
    ) -> List[int]:
        """Parse categorical feature specification into indices."""
        if categorical_features is None:
            return []
        
        indices = []
        for feature in categorical_features:
            if isinstance(feature, int):
                if 0 <= feature < len(self.feature_names):
                    indices.append(feature)
                else:
                    raise ValidationError(
                        f"Categorical feature index {feature} out of range",
                        parameter="categorical_features"
                    )
            elif isinstance(feature, str):
                if feature in self.feature_names:
                    indices.append(self.feature_names.index(feature))
                else:
                    raise ValidationError(
                        f"Categorical feature '{feature}' not found",
                        parameter="categorical_features"
                    )
            else:
                raise ValidationError(
                    f"Categorical feature must be int or str, got {type(feature)}",
                    parameter="categorical_features"
                )
        
        return indices
    
    def _infer_dtypes(self) -> Dict[str, str]:
        """Infer data types for each feature."""
        dtypes = {}
        
        if isinstance(self.X, pd.DataFrame):
            for col in self.X.columns:
                if pd.api.types.is_numeric_dtype(self.X[col]):
                    dtypes[col] = 'numeric'
                else:
                    dtypes[col] = 'categorical'
        else:
            # For numpy arrays, check if feature is in categorical list
            for i, name in enumerate(self.feature_names):
                if i in self.categorical_features:
                    dtypes[name] = 'categorical'
                else:
                    dtypes[name] = 'numeric'
        
        return dtypes
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the data."""
        return self.X.shape
    
    @property
    def n_samples(self) -> int:
        """Get number of samples."""
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        """Get number of features."""
        return self.X.shape[1]
    
    def get_feature_info(self) -> pd.DataFrame:
        """Get information about features."""
        info_data = []
        
        for i, name in enumerate(self.feature_names):
            feature_info = {
                'feature_name': name,
                'index': i,
                'dtype': self.dtypes.get(name, 'unknown'),
                'is_categorical': i in self.categorical_features
            }
            
            # Add statistics
            if isinstance(self.X, pd.DataFrame):
                col_data = self.X.iloc[:, i]
            else:
                col_data = self.X[:, i]
            
            if self.dtypes.get(name) == 'numeric':
                feature_info.update({
                    'mean': np.mean(col_data),
                    'std': np.std(col_data),
                    'min': np.min(col_data),
                    'max': np.max(col_data),
                    'n_unique': len(np.unique(col_data))
                })
            else:
                feature_info.update({
                    'n_unique': len(np.unique(col_data)),
                    'most_frequent': pd.Series(col_data).mode().iloc[0] if len(col_data) > 0 else None
                })
            
            info_data.append(feature_info)
        
        return pd.DataFrame(info_data)
    
    def preprocess(
        self,
        scale_features: bool = True,
        scaling_method: str = 'standard',
        encode_categorical: bool = True,
        encoding_method: str = 'label'
    ) -> 'TabularData':
        """
        Preprocess the data with scaling and encoding.
        
        Args:
            scale_features: Whether to scale numeric features
            scaling_method: Scaling method ('standard', 'minmax')
            encode_categorical: Whether to encode categorical features
            encoding_method: Encoding method ('label', 'onehot')
            
        Returns:
            New TabularData instance with preprocessed data
        """
        X_processed = self._X_array.copy()
        
        # Scale numeric features
        if scale_features:
            numeric_indices = [i for i in range(self.n_features) 
                             if i not in self.categorical_features]
            
            if numeric_indices:
                if scaling_method == 'standard':
                    scaler = StandardScaler()
                elif scaling_method == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    raise ValidationError(
                        f"Unsupported scaling method: {scaling_method}",
                        parameter="scaling_method",
                        expected="'standard' or 'minmax'"
                    )
                
                X_processed[:, numeric_indices] = scaler.fit_transform(
                    X_processed[:, numeric_indices]
                )
                self.scalers[scaling_method] = (scaler, numeric_indices)
        
        # Encode categorical features
        if encode_categorical and self.categorical_features:
            if encoding_method == 'label':
                for cat_idx in self.categorical_features:
                    encoder = LabelEncoder()
                    X_processed[:, cat_idx] = encoder.fit_transform(X_processed[:, cat_idx])
                    self.encoders[f'label_{cat_idx}'] = encoder
            else:
                raise ValidationError(
                    f"Unsupported encoding method: {encoding_method}",
                    parameter="encoding_method",
                    expected="'label'"
                )
        
        # Create new TabularData instance
        processed_data = TabularData(
            X=X_processed,
            y=self._y_array,
            feature_names=self.feature_names.copy(),
            categorical_features=self.categorical_features.copy(),
            target_name=self.target_name
        )
        processed_data._is_preprocessed = True
        processed_data.scalers = self.scalers
        processed_data.encoders = self.encoders
        
        return processed_data
    
    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify: bool = False
    ) -> Tuple['TabularData', 'TabularData']:
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            stratify: Whether to stratify split (for classification)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.y is None:
            raise DataError("Cannot split data without target values")
        
        stratify_y = self._y_array if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            self._X_array, self._y_array,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y
        )
        
        # Create TabularData instances for train and test
        train_data = TabularData(
            X=X_train,
            y=y_train,
            feature_names=self.feature_names.copy(),
            categorical_features=self.categorical_features.copy(),
            target_name=self.target_name
        )
        
        test_data = TabularData(
            X=X_test,
            y=y_test,
            feature_names=self.feature_names.copy(),
            categorical_features=self.categorical_features.copy(),
            target_name=self.target_name
        )
        
        return train_data, test_data
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        df = pd.DataFrame(self._X_array, columns=self.feature_names)
        
        if self._y_array is not None:
            df[self.target_name] = self._y_array
        
        return df
    
    def to_arrays(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert to numpy arrays."""
        return self._X_array, self._y_array
    
    def sample(
        self, 
        n: Optional[int] = None, 
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> 'TabularData':
        """
        Sample from the data.
        
        Args:
            n: Number of samples to return
            frac: Fraction of samples to return
            random_state: Random state for reproducibility
            
        Returns:
            New TabularData instance with sampled data
        """
        if n is None and frac is None:
            raise ValidationError("Either n or frac must be specified")
        
        if n is not None and frac is not None:
            raise ValidationError("Cannot specify both n and frac")
        
        np.random.seed(random_state)
        
        if frac is not None:
            n = int(self.n_samples * frac)
        
        if n > self.n_samples:
            raise ValidationError(
                f"Cannot sample {n} items from {self.n_samples} samples"
            )
        
        indices = np.random.choice(self.n_samples, size=n, replace=False)
        
        X_sampled = self._X_array[indices]
        y_sampled = self._y_array[indices] if self._y_array is not None else None
        
        return TabularData(
            X=X_sampled,
            y=y_sampled,
            feature_names=self.feature_names.copy(),
            categorical_features=self.categorical_features.copy(),
            target_name=self.target_name
        )
    
    def describe(self) -> pd.DataFrame:
        """Get descriptive statistics for the data."""
        if isinstance(self.X, pd.DataFrame):
            return self.X.describe(include='all')
        else:
            df = pd.DataFrame(self._X_array, columns=self.feature_names)
            return df.describe(include='all')
    
    def __len__(self) -> int:
        """Get length of the data."""
        return self.n_samples
    
    def __getitem__(self, key) -> 'TabularData':
        """Get subset of the data."""
        if isinstance(key, slice):
            indices = range(*key.indices(self.n_samples))
        elif isinstance(key, (list, np.ndarray)):
            indices = key
        else:
            indices = [key]
        
        X_subset = self._X_array[indices]
        y_subset = self._y_array[indices] if self._y_array is not None else None
        
        return TabularData(
            X=X_subset,
            y=y_subset,
            feature_names=self.feature_names.copy(),
            categorical_features=self.categorical_features.copy(),
            target_name=self.target_name
        )
    
    def __repr__(self) -> str:
        """String representation of TabularData."""
        return (f"TabularData(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, "
                f"has_target={self.y is not None})")
