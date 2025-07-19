# PyInterpret Trusted Publisher Setup Guide

## What is Trusted Publishing?

Trusted Publishing uses OpenID Connect (OIDC) to securely publish packages without API tokens. GitHub Actions can authenticate directly with PyPI using cryptographic proofs.

## Setup Steps

### 1. Push Code to GitHub

First, create a GitHub repository and push your PyInterpret code:

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Initial PyInterpret release"

# Add your GitHub repository as origin
git remote add origin https://github.com/YOUR_USERNAME/pyinterpret.git
git push -u origin main
```

### 2. Configure Trusted Publisher on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Under "Pending publishers", click "Add a new pending publisher"
3. Fill in the form:
   - **PyPI project name**: `pyinterpret`
   - **Owner**: Your GitHub username
   - **Repository name**: `pyinterpret` (or whatever you named it)
   - **Workflow filename**: `publish.yml`
   - **Environment name**: `pypi`

### 3. Optional: Set up Test PyPI Publisher

Repeat the same process at https://test.pypi.org/manage/account/publishing/ with:
   - **Environment name**: `testpypi`

### 4. GitHub Repository Settings

In your GitHub repository:
1. Go to Settings → Environments
2. Create environment named `pypi`
3. Optionally create `testpypi` environment

## How to Publish

### Option 1: Create a Release (Recommended)
1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Create a tag like `v0.1.0`
4. Title: "PyInterpret v0.1.0"
5. Publish release → GitHub Actions will automatically publish to PyPI

### Option 2: Manual Workflow Trigger
1. Go to Actions tab in your repository
2. Click "Publish PyInterpret to PyPI"
3. Click "Run workflow"

## GitHub Actions Workflow

The workflow in `.github/workflows/publish.yml`:
- Builds your package automatically
- Publishes to TestPyPI on every push
- Publishes to PyPI only on tagged releases
- Uses OIDC for secure, credential-free authentication

## Advantages of Trusted Publishing

✅ **No API tokens to manage**
✅ **More secure than passwords**
✅ **Automatic authentication**
✅ **Audit trail through GitHub**
✅ **No credentials in repository**

## First Time Setup Summary

1. Push code to GitHub
2. Configure trusted publisher on PyPI
3. Create a GitHub release with tag `v0.1.0`
4. Watch GitHub Actions automatically publish your package!

Your PyInterpret library will be available at: https://pypi.org/project/pyinterpret/