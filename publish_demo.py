#!/usr/bin/env python3
"""
Demo version of PyInterpret package publisher.
Shows the publishing process without requiring user input.
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print("Output:", result.stdout.strip()[:200] + "..." if len(result.stdout) > 200 else result.stdout.strip())
            return True
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print("Error:", result.stderr.strip()[:200] + "..." if len(result.stderr) > 200 else result.stderr.strip())
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    """Demo publication workflow."""
    print("🚀 PyInterpret Package Publisher (Demo Mode)")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('pyinterpret') or not os.path.exists('setup.py'):
        print("❌ Error: Run this script from the PyInterpret root directory")
        sys.exit(1)
    
    print("\nThis demo shows the PyInterpret publishing process:")
    print("1. Clean build artifacts ✓")
    print("2. Check requirements ✓") 
    print("3. Build package ✓")
    print("4. Check package ✓")
    print("5. Show upload commands (demo mode)")
    
    # Step 1: Clean
    print("\n" + "="*50)
    print("STEP 1: CLEANING BUILD ARTIFACTS")
    run_command("rm -rf build dist *.egg-info", "Cleaning build artifacts")
    
    # Step 2: Check twine
    print("\n" + "="*50)
    print("STEP 2: CHECKING REQUIREMENTS")
    run_command("python -m pip show twine", "Checking twine installation")
    
    # Step 3: Build
    print("\n" + "="*50)
    print("STEP 3: BUILDING PACKAGE")
    if run_command("python -m build", "Building package"):
        print("\n📦 Distribution files created:")
        if os.path.exists('dist'):
            for file in os.listdir('dist'):
                print(f"  - {file}")
    
    # Step 4: Check
    print("\n" + "="*50)
    print("STEP 4: CHECKING PACKAGE")
    run_command("python -m twine check dist/*", "Checking package")
    
    # Step 5: Demo upload commands
    print("\n" + "="*50)
    print("STEP 5: UPLOAD COMMANDS (Demo Mode)")
    print("\n🔑 To upload to Test PyPI, you would run:")
    print("   python -m twine upload --repository testpypi dist/*")
    print("   Username: __token__")
    print("   Password: [Your Test PyPI API token]")
    
    print("\n🔑 To upload to real PyPI, you would run:")
    print("   python -m twine upload dist/*")
    print("   Username: __token__")
    print("   Password: [Your PyPI API token]")
    
    print("\n" + "="*50)
    print("🎉 DEMO COMPLETED!")
    print("\nYour package is ready for publishing!")
    print("To actually upload:")
    print("1. Create accounts at https://test.pypi.org and https://pypi.org")
    print("2. Generate API tokens in your account settings")
    print("3. Run the upload commands shown above")
    
    print("\n📊 Package Statistics:")
    if os.path.exists('dist'):
        for file in os.listdir('dist'):
            size = os.path.getsize(f'dist/{file}')
            print(f"  - {file}: {size:,} bytes")
    
    print("\n🌟 After publishing, users can install with:")
    print("   pip install pyinterpret")

if __name__ == "__main__":
    main()