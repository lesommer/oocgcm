# Updating the oocgcm Package

This document outlines the main issues that need to be addressed to update the oocgcm package for compatibility with modern Python and dependency versions.

## Current State

- **Python Version**: 3.14.4
- **Dependencies**: Updated to latest versions (e.g., dask 2026.3.0, xarray 2026.4.0, numpy 2.4.4)
- **Test Status**: Some tests are passing, but many are failing due to compatibility issues.

## Main Issues to Address

### 1. Python 3 Compatibility

#### Issues:
- **`print` Statements**: Some files still use Python 2-style `print` statements without parentheses.
- **`has_key` Method**: The `has_key` method for dictionaries is deprecated in Python 3. Use `in` operator instead.
- **String Literals**: Some string literals use `\` for escaping, which is not compatible with Python 3.

#### Files Affected:
- `/Users/lesommer/git/oocgcm/oocgcm/test/__init__.py`
- `/Users/lesommer/git/oocgcm/oocgcm/test/test_grids.py`
- `/Users/lesommer/git/oocgcm/oocgcm/core/grids.py`
- `/Users/lesommer/git/oocgcm/oocgcm/core/utils.py`

#### Fixes Applied:
- Updated `print` statements to use parentheses.
- Replaced `has_key` with `in` operator for dictionary checks.
- Fixed string literals to avoid invalid escape sequences.

### 2. Dask Compatibility

#### Issues:
- **`dask.set_options`**: This method is deprecated and replaced with `dask.config.set`.
- **`chunks` Parameter**: Passing `None` as the `chunks` parameter is deprecated. Use an empty dictionary `{}` instead.
- **`map_overlap` Dtype**: The `map_overlap` function now requires explicit `dtype` specification.

#### Files Affected:
- `/Users/lesommer/git/oocgcm/oocgcm/test/__init__.py`
- `/Users/lesommer/git/oocgcm/oocgcm/core/grids.py`
- `/Users/lesommer/git/oocgcm/oocgcm/griddeddata/grids.py`

#### Fixes Applied:
- Replaced `dask.set_options` with `dask.config.set`.
- Updated `chunks` parameter handling in grid operations.

#### Pending Fixes:
- Specify `dtype` explicitly in `map_overlap` operations.
- Update all grid operations to handle `chunks` parameter correctly.

### 3. Test Failures

#### Issues:
- **Test Errors**: Many tests are failing due to compatibility issues with newer versions of dependencies.
- **Test Coverage**: Some tests may need updates to reflect changes in the codebase.

#### Files Affected:
- `/Users/lesommer/git/oocgcm/oocgcm/test/test_grids.py`
- `/Users/lesommer/git/oocgcm/oocgcm/test/test_eos.py`

#### Fixes Applied:
- Fixed indentation and syntax errors in test files.
- Updated test files for Python 3 compatibility.

#### Pending Fixes:
- Investigate and fix test failures related to grid computations and dask array operations.
- Ensure all tests pass with the updated dependencies.

### 4. Dependency Management

#### Issues:
- **Outdated Dependencies**: Some dependencies were outdated and needed updates.
- **Missing Dependencies**: Some dependencies (e.g., `dask`) were not listed in the project's `setup.py`.

#### Files Affected:
- `/Users/lesommer/git/oocgcm/setup.py`

#### Fixes Applied:
- Updated `certifi`, `matplotlib`, `packaging`, and `wheel` to their latest versions.

#### Pending Fixes:
- Add `dask` and other necessary dependencies to the `INSTALL_REQUIRES` list in `setup.py`.
- Ensure all dependencies are compatible with the latest versions.

### 5. Code Quality and Warnings

#### Issues:
- **Syntax Warnings**: Some files have syntax warnings related to invalid escape sequences and string comparisons.
- **Deprecation Warnings**: Warnings related to deprecated methods and parameters.

#### Files Affected:
- `/Users/lesommer/git/oocgcm/oocgcm/core/grids.py`

#### Fixes Applied:
- Fixed invalid escape sequences in string literals.

#### Pending Fixes:
- Address deprecation warnings related to string comparisons (`is` vs `==`).
- Clean up syntax warnings to improve code quality.

## Next Steps

1. **Fix Dask Compatibility**:
   - Update `chunks` parameter handling in grid operations.
   - Specify `dtype` explicitly in `map_overlap` operations.

2. **Address Test Failures**:
   - Investigate and fix test failures related to grid computations and dask array operations.
   - Ensure all tests pass with the updated dependencies.

3. **Update Dependency Management**:
   - Add `dask` and other necessary dependencies to the `INSTALL_REQUIRES` list in `setup.py`.
   - Ensure all dependencies are compatible with the latest versions.

4. **Improve Code Quality**:
   - Address deprecation warnings related to string comparisons.
   - Clean up syntax warnings to improve code quality.

5. **Run Full Test Suite**:
   - Execute the full test suite to verify that all tests pass.
   - Address any remaining issues that arise during testing.

## Summary

The oocgcm package has been partially updated for compatibility with modern Python and dependency versions. Several fixes have been applied to address Python 3 compatibility and dask compatibility issues. However, there are still pending fixes related to dask operations, test failures, and dependency management. Following the outlined next steps will help ensure the package is fully updated and functional.