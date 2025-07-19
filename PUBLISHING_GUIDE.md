# PyInterpret Publishing Guide

This guide explains how to publish PyInterpret so anyone can install it with `pip install pyinterpret`.

## 📋 Prerequisites

1. **PyPI Account**: Create accounts on both:
   - [Test PyPI](https://test.pypi.org/account/register/) (for testing)
   - [PyPI](https://pypi.org/account/register/) (for real distribution)

2. **API Tokens**: Generate API tokens for upload authentication:
   - Go to Account Settings → API tokens
   - Create tokens for both Test PyPI and PyPI
   - Save these tokens securely

## 🚀 Quick Publishing (Automated)

Use the automated publisher script:

```bash
python publish_package.py
```

This script will:
- Clean old build files
- Install required tools
- Build the package
- Check for errors
- Upload to Test PyPI (optional)
- Upload to real PyPI

## 🔧 Manual Publishing Steps

If you prefer manual control:

### 1. Install Publishing Tools
```bash
pip install build twine
```

### 2. Clean Previous Builds
```bash
rm -rf build dist *.egg-info
```

### 3. Build Package
```bash
python -m build
```

### 4. Check Package
```bash
python -m twine check dist/*
```

### 5. Upload to Test PyPI (Recommended First)
```bash
python -m twine upload --repository testpypi dist/*
```

Test installation:
```bash
pip install -i https://test.pypi.org/simple/ pyinterpret
```

### 6. Upload to Real PyPI
```bash
python -m twine upload dist/*
```

## 📦 After Publishing

Once published, anyone can install PyInterpret:

```bash
# Basic installation
pip install pyinterpret

# With optional dependencies
pip install pyinterpret[shap]      # SHAP support
pip install pyinterpret[lime]      # LIME support  
pip install pyinterpret[all]       # All features
```

## 🔐 Authentication

When uploading, you'll be prompted for credentials. Use:
- Username: `__token__`
- Password: Your API token (starts with `pypi-`)

## 📝 Version Management

To release a new version:

1. Update version in `pyinterpret/__init__.py`:
   ```python
   __version__ = "0.1.1"  # Increment version
   ```

2. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.1"
   ```

3. Follow publishing steps again

## 🌐 Distribution Checklist

Before publishing, ensure:

- [ ] All tests pass: `python -m pytest tests/`
- [ ] Examples work: `python examples/basic_usage.py`
- [ ] Documentation is updated
- [ ] Version numbers match in all files
- [ ] README.md has installation instructions
- [ ] LICENSE file exists
- [ ] All dependencies are correctly specified

## 📊 Monitoring

After publishing:
- Check [PyPI page](https://pypi.org/project/pyinterpret/) for your package
- Monitor download statistics
- Watch for user feedback and issues

## 🆘 Troubleshooting

**Common Issues:**

1. **"Package already exists"**: You cannot overwrite a version. Increment the version number.

2. **Authentication failed**: Double-check your API token and ensure you're using `__token__` as username.

3. **Build errors**: Check that all files are included in `MANIFEST.in`.

4. **Import errors**: Test the package in a fresh environment after installation.

## 🔄 Continuous Publishing

For automated publishing (advanced):
- Set up GitHub Actions
- Use trusted publishing with PyPI
- Automate version bumping and releases

## 📞 Support

If you encounter issues:
1. Check the [PyPI Help](https://pypi.org/help/)
2. Review build logs for specific errors
3. Test in a clean virtual environment
4. Verify package structure matches Python standards