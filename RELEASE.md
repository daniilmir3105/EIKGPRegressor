# Release Guide (`eikgp-regressor` 0.2.0)

This document is the release checklist for version **0.2.0**. The distribution name used by
PyPI and `pip` is `eikgp-regressor`; the Python import package is `eikg`.

The primary production publishing path is the GitHub Actions workflow
`.github/workflows/publish.yml` with PyPI Trusted Publishing. A manual Twine upload is retained
only as a fallback.

## 1. One-time account and repository setup

1. Create and verify the maintainer accounts on both
   [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/). They are separate services
   and use separate credentials.
2. Enable two-factor authentication on PyPI and TestPyPI. Store the recovery codes in a secure
   location outside the repository.
3. In the GitHub repository, create a protected environment named `pypi`. Configure required
   reviewers or other deployment protections if desired.
4. Configure a PyPI pending Trusted Publisher. This also works before the project exists on
   PyPI. Use exactly these values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `eikgp-regressor` |
   | GitHub owner | `daniilmir3105` |
   | Repository | `EIKGPolynomial` |
   | Workflow | `publish.yml` |
   | Environment | `pypi` |

Trusted Publishing exchanges GitHub's short-lived OIDC identity for a short-lived PyPI token;
no long-lived production token needs to be stored in GitHub Secrets.

## 2. Verify the release commit

Version 0.2.0 must appear in `pyproject.toml`, and the corresponding dated section must appear
in `CHANGELOG.md`. Confirm that the working tree contains only intended release changes:

```powershell
git status --short
git diff --check
git diff -- pyproject.toml CHANGELOG.md README.md RELEASE.md MANIFEST.in
```

Run every local quality gate from the repository root in a development environment:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy eikg
```

Commit and push the release preparation changes. Do not create the tag yet:

```powershell
git add .
git commit -m "Prepare release 0.2.0"
git push origin main
```

Wait for CI on that exact commit to complete successfully. If the default branch is protected,
use the normal pull-request and review process instead of pushing directly.

## 3. Build clean artifacts

Never build on top of files left by an earlier version. Remove only the repository's local
`dist` directory, verify that it is gone, and build from the release commit:

```powershell
if (Test-Path .\dist) { Remove-Item -LiteralPath .\dist -Recurse -Force }
Test-Path .\dist
python -m build
```

`Test-Path` must print `False` before the build. The new directory must contain exactly these
two files:

```text
dist/eikgp_regressor-0.2.0-py3-none-any.whl
dist/eikgp_regressor-0.2.0.tar.gz
```

Inspect the directory and validate both artifacts strictly:

```powershell
Get-ChildItem -LiteralPath .\dist
python -m twine check --strict `
  .\dist\eikgp_regressor-0.2.0-py3-none-any.whl `
  .\dist\eikgp_regressor-0.2.0.tar.gz
```

Do not use `dist/*` for uploads: an old or unrelated artifact must never be uploaded by accident.

## 4. Test the exact release on TestPyPI

Upload only the two named 0.2.0 artifacts. Use a TestPyPI project-scoped API token when the
project already exists, or an account-scoped token for its first upload:

```powershell
$env:TWINE_USERNAME = "__token__"
python -m twine upload --repository testpypi `
  .\dist\eikgp_regressor-0.2.0-py3-none-any.whl `
  .\dist\eikgp_regressor-0.2.0.tar.gz
Remove-Item Env:TWINE_USERNAME
```

Paste the complete TestPyPI token into Twine's hidden password prompt. Never commit a token,
assign its literal value in a shell command that may enter command history, place it in a
command-line argument, paste it into an issue, or reuse a TestPyPI token on production PyPI.

Create a clean virtual environment. Install runtime dependencies from production PyPI first,
then install the exact TestPyPI artifact with `--no-deps`. This avoids using TestPyPI as an
untrusted dependency source:

```powershell
python -m venv .venv-testpypi
.\.venv-testpypi\Scripts\python.exe -m pip install --upgrade pip
.\.venv-testpypi\Scripts\python.exe -m pip install "numpy>=1.24"
.\.venv-testpypi\Scripts\python.exe -m pip install `
  --index-url https://test.pypi.org/simple/ `
  --no-deps `
  eikgp-regressor==0.2.0
.\.venv-testpypi\Scripts\python.exe -c `
  "from importlib.metadata import version; from eikg import EIKGPolynomialRegressor; print(version('eikgp-regressor'))"
```

Also run a small fit/predict smoke test if desired. Delete this disposable environment after
verification. If any source change is required, make it, rerun all checks, rebuild from a clean
`dist`, and repeat the release process from the new commit.

## 5. Tag, create the GitHub Release, and publish to PyPI

After TestPyPI verification, ensure `HEAD` is still the exact green release commit. Create and
push an annotated tag whose version matches `pyproject.toml`:

```powershell
git status --short
git tag -a v0.2.0 -m "Release 0.2.0"
git push origin v0.2.0
```

Create a GitHub Release for the existing `v0.2.0` tag, use the 0.2.0 changelog section as the
release notes, and publish the release. The `release: published` event starts
`.github/workflows/publish.yml`. That workflow builds the artifacts and publishes them through
the protected `pypi` environment and Trusted Publishing.

Review the workflow logs and the resulting PyPI project page. Then verify the production package
from another clean virtual environment:

```powershell
python -m venv .venv-pypi-check
.\.venv-pypi-check\Scripts\python.exe -m pip install --upgrade pip
.\.venv-pypi-check\Scripts\python.exe -m pip install eikgp-regressor==0.2.0
.\.venv-pypi-check\Scripts\python.exe -c `
  "from importlib.metadata import version; from eikg import EIKGPolynomialRegressor; print(version('eikgp-regressor'))"
```

Do not move or recreate `v0.2.0` after publication. If GitHub release publication fails before
PyPI receives any files, diagnose the workflow and rerun the failed job. If PyPI received the
files, inspect the project page before attempting any retry.

## 6. Manual production upload (fallback only)

Use this path only when Trusted Publishing cannot be restored and the exact locally built
artifacts have passed all checks. Create a short-lived, project-scoped production PyPI API token
and keep two-factor authentication enabled:

```powershell
$env:TWINE_USERNAME = "__token__"
python -m twine upload `
  .\dist\eikgp_regressor-0.2.0-py3-none-any.whl `
  .\dist\eikgp_regressor-0.2.0.tar.gz
Remove-Item Env:TWINE_USERNAME
```

Paste the complete production token into Twine's hidden password prompt, then revoke the
temporary token after the upload. Never store it in the repository, workflow file, shell
history, or an unprotected GitHub secret.

## 7. PyPI immutability and release recovery

PyPI does not allow a distribution filename to be reused, even if the original file is deleted.
Published `eikgp_regressor-0.2.0` artifacts therefore cannot be replaced in place. If a published
artifact is incorrect:

1. Do not delete and retry version 0.2.0 expecting the filename to become available.
2. Fix the source and tests.
3. Increment the version, normally to `0.2.1` for a compatible bug fix.
4. Add a changelog entry and repeat this entire checklist with the new version and tag.

A release may be yanked on PyPI to discourage new installations, but yanking also does not make
its filenames reusable.
