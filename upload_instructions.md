# PyInterpret Upload Instructions

## Step 1: Get Your API Token

1. Go to https://pypi.org/manage/account/
2. Scroll down to "API tokens" section
3. Click "Add API token"
4. Give it a name like "PyInterpret Upload"
5. Set scope to "Entire account" (or create project-specific after first upload)
6. Copy the token (starts with `pypi-`)

## Step 2: Upload to Test PyPI First (Recommended)

For safety, test on Test PyPI first:

1. Get Test PyPI token: https://test.pypi.org/manage/account/
2. Upload to test:
```bash
python -m twine upload --repository testpypi dist/*
```
- Username: `__token__`
- Password: Your Test PyPI token

3. Test install:
```bash
pip install -i https://test.pypi.org/simple/ pyinterpret
```

## Step 3: Upload to Production PyPI

Once test works:
```bash
python -m twine upload dist/*
```
- Username: `__token__`
- Password: Your PyPI token

## Ready to Upload

Your package files are ready:
- pyinterpret-0.1.0.tar.gz (350KB)
- pyinterpret-0.1.0-py3-none-any.whl (32KB)

All validation checks passed!