"""
Custom exceptions for PyInterpret library.

This module defines specific exception types to provide clear error messages
and enable proper error handling throughout the library.
"""


class PyInterpretError(Exception):
    """Base exception class for PyInterpret library."""
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ValidationError(PyInterpretError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, parameter: str = None, expected: str = None, received: str = None):
        self.parameter = parameter
        self.expected = expected
        self.received = received
        
        details = []
        if parameter:
            details.append(f"Parameter: {parameter}")
        if expected:
            details.append(f"Expected: {expected}")
        if received:
            details.append(f"Received: {received}")
        
        detail_str = ", ".join(details) if details else None
        super().__init__(message, detail_str)


class ModelError(PyInterpretError):
    """Raised when there are issues with the provided model."""
    
    def __init__(self, message: str, model_type: str = None, required_methods: list = None):
        self.model_type = model_type
        self.required_methods = required_methods
        
        details = []
        if model_type:
            details.append(f"Model type: {model_type}")
        if required_methods:
            details.append(f"Required methods: {', '.join(required_methods)}")
        
        detail_str = ", ".join(details) if details else None
        super().__init__(message, detail_str)


class ExplainerError(PyInterpretError):
    """Raised when explainer-specific errors occur."""
    
    def __init__(self, message: str, explainer: str = None, method: str = None):
        self.explainer = explainer
        self.method = method
        
        details = []
        if explainer:
            details.append(f"Explainer: {explainer}")
        if method:
            details.append(f"Method: {method}")
        
        detail_str = ", ".join(details) if details else None
        super().__init__(message, detail_str)


class DataError(PyInterpretError):
    """Raised when there are issues with input data."""
    
    def __init__(self, message: str, data_shape: tuple = None, expected_shape: tuple = None):
        self.data_shape = data_shape
        self.expected_shape = expected_shape
        
        details = []
        if data_shape:
            details.append(f"Data shape: {data_shape}")
        if expected_shape:
            details.append(f"Expected shape: {expected_shape}")
        
        detail_str = ", ".join(details) if details else None
        super().__init__(message, detail_str)


class ConfigurationError(PyInterpretError):
    """Raised when there are configuration issues."""
    
    def __init__(self, message: str, parameter: str = None, valid_options: list = None):
        self.parameter = parameter
        self.valid_options = valid_options
        
        details = []
        if parameter:
            details.append(f"Parameter: {parameter}")
        if valid_options:
            details.append(f"Valid options: {', '.join(map(str, valid_options))}")
        
        detail_str = ", ".join(details) if details else None
        super().__init__(message, detail_str)
