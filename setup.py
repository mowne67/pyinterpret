"""
Setup script for PyInterpret library.

This script configures the package for installation and distribution.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read version from package
def get_version():
    """Get version from __init__.py file."""
    version_file = os.path.join(this_directory, 'pyinterpret', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Core dependencies
install_requires = [
    'numpy>=1.19.0',
    'pandas>=1.1.0',
    'scikit-learn>=0.24.0',
    'matplotlib>=3.3.0',
]

# Optional dependencies
extras_require = {
    'shap': ['shap>=0.39.0'],
    'lime': ['lime>=0.2.0'],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'sphinx-autodoc-typehints>=1.11.0',
    ],
    'testing': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'pytest-mock>=3.3.0',
    ],
    'dev': [
        'black>=21.0.0',
        'flake8>=3.8.0',
        'isort>=5.0.0',
        'pre-commit>=2.10.0',
    ],
    'all': [
        'shap>=0.39.0',
        'lime>=0.2.0',
        'plotly>=5.0.0',
    ]
}

# Development dependencies include testing and documentation
extras_require['dev'] += extras_require['testing'] + extras_require['docs']

setup(
    name='pyinterpret',
    version=get_version(),
    author='PyInterpret Team',
    author_email='team@pyinterpret.org',
    description='A unified Python library for machine learning model interpretation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pyinterpret/pyinterpret',
    project_urls={
        'Documentation': 'https://pyinterpret.readthedocs.io/',
        'Source': 'https://github.com/pyinterpret/pyinterpret',
        'Tracker': 'https://github.com/pyinterpret/pyinterpret/issues',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
    keywords=[
        'machine learning',
        'interpretability',
        'explainability',
        'shap',
        'lime',
        'feature importance',
        'model explanation',
        'xai',
        'artificial intelligence'
    ],
    entry_points={
        'console_scripts': [
            # Future CLI tools can be added here
        ],
    },
)
