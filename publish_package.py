#!/usr/bin/env python3
"""
PyInterpret Package Publisher

This script helps you build and publish the PyInterpret package to PyPI
so that anyone can install it with 'pip install pyinterpret'.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    print(f"Running: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        print(f"Error output: {result.stderr}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("\n🧹 Cleaning previous build artifacts...")
    
    artifacts = ['build', 'dist', '*.egg-info']
    for pattern in artifacts:
        if pattern.endswith('*'):
            # Handle glob patterns
            import glob
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
        else:
            if os.path.exists(pattern):
                if os.path.isdir(pattern):
                    shutil.rmtree(pattern)
                    print(f"Removed directory: {pattern}")
                else:
                    os.remove(pattern)
                    print(f"Removed file: {pattern}")

def check_requirements():
    """Check if required tools are installed."""
    print("\n🔍 Checking requirements...")
    
    required_packages = ['build', 'twine']
    missing_packages = []
    
    for package in required_packages:
        result = subprocess.run(f"python -m pip show {package}", 
                              shell=True, capture_output=True)
        if result.returncode != 0:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Installing missing packages...")
        for package in missing_packages:
            if not run_command(f"python -m pip install {package}", 
                             f"Installing {package}"):
                return False
    
    print("✅ All requirements satisfied")
    return True

def build_package():
    """Build the package."""
    return run_command("python -m build", "Building package")

def check_package():
    """Check the built package."""
    return run_command("python -m twine check dist/*", "Checking package")

def upload_to_test_pypi():
    """Upload to Test PyPI first."""
    print("\n📤 Uploading to Test PyPI...")
    print("Note: You'll need Test PyPI credentials")
    return run_command("python -m twine upload --repository testpypi dist/*", 
                      "Uploading to Test PyPI")

def upload_to_pypi():
    """Upload to real PyPI."""
    print("\n📤 Uploading to PyPI...")
    print("Note: You'll need PyPI credentials")
    return run_command("python -m twine upload dist/*", 
                      "Uploading to PyPI")

def main():
    """Main publication workflow."""
    print("🚀 PyInterpret Package Publisher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('pyinterpret') or not os.path.exists('setup.py'):
        print("❌ Error: Run this script from the PyInterpret root directory")
        sys.exit(1)
    
    print("\nThis script will help you publish PyInterpret to PyPI.")
    print("Steps:")
    print("1. Clean build artifacts")
    print("2. Check requirements") 
    print("3. Build package")
    print("4. Check package")
    print("5. Upload to Test PyPI (recommended first)")
    print("6. Upload to PyPI (final step)")
    
    choice = input("\nDo you want to continue? (y/n): ").lower().strip()
    if choice != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Step 1: Clean
    clean_build_artifacts()
    
    # Step 2: Check requirements
    if not check_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Step 3: Build
    if not build_package():
        print("❌ Build failed")
        sys.exit(1)
    
    # Step 4: Check
    if not check_package():
        print("❌ Package check failed")
        sys.exit(1)
    
    # Step 5: Test PyPI (optional)
    test_upload = input("\nUpload to Test PyPI first? (y/n): ").lower().strip()
    if test_upload == 'y':
        if upload_to_test_pypi():
            print("\n✅ Successfully uploaded to Test PyPI!")
            print("You can test install with:")
            print("pip install -i https://test.pypi.org/simple/ pyinterpret")
        else:
            print("❌ Test PyPI upload failed")
            choice = input("Continue to real PyPI anyway? (y/n): ").lower().strip()
            if choice != 'y':
                sys.exit(1)
    
    # Step 6: Real PyPI
    real_upload = input("\nUpload to real PyPI? (y/n): ").lower().strip()
    if real_upload == 'y':
        if upload_to_pypi():
            print("\n🎉 SUCCESS! PyInterpret is now published!")
            print("Anyone can install it with:")
            print("pip install pyinterpret")
        else:
            print("❌ PyPI upload failed")
            sys.exit(1)
    
    print("\n✅ Publication process completed!")

if __name__ == "__main__":
    main()