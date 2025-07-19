#!/usr/bin/env python3
"""
Test script to verify PyInterpret package works correctly before publishing.
"""

import sys
import subprocess
import tempfile
import os

def test_import():
    """Test that PyInterpret can be imported correctly."""
    print("🔍 Testing imports...")
    try:
        import pyinterpret
        from pyinterpret import SHAPExplainer, LIMEExplainer, PermutationImportanceExplainer
        print(f"✅ PyInterpret v{pyinterpret.__version__} imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality."""
    print("🧪 Testing basic functionality...")
    try:
        from pyinterpret import PermutationImportanceExplainer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        import pandas as pd
        
        # Create test data
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        # Test explainer
        explainer = PermutationImportanceExplainer(model, scoring='accuracy', n_repeats=2)
        result = explainer.explain_global(X_df, y)
        
        assert hasattr(result, 'attributions')
        assert hasattr(result, 'feature_names')
        assert len(result.attributions) == X.shape[1]
        
        print("✅ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_package_build():
    """Test that the package can be built."""
    print("📦 Testing package build...")
    try:
        # Clean previous builds
        subprocess.run("rm -rf build dist *.egg-info", shell=True)
        
        # Try to build
        result = subprocess.run("python -m build", shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Package builds successfully")
            
            # Check if files were created
            if os.path.exists('dist'):
                files = os.listdir('dist')
                print(f"📁 Created distribution files: {files}")
                return True
            else:
                print("❌ No dist directory created")
                return False
        else:
            print(f"❌ Build failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Build test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 PyInterpret Package Test Suite")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_import),
        ("Functionality Test", test_basic_functionality),
        ("Package Build Test", test_package_build)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready for publishing.")
        print("\nNext steps:")
        print("1. Run: python publish_package.py")
        print("2. Or follow PUBLISHING_GUIDE.md for manual steps")
        return True
    else:
        print("❌ Some tests failed. Fix issues before publishing.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)