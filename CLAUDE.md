# Code style
- Line length: 88 characters maximum
- Use the `|` union syntax for `isinstance` checks, never a tuple:
  `isinstance(x, Foo | Bar)` ✓
  `isinstance(x, (Foo, Bar))` ✗
- Do not use sequences of lines (`─`, `-`, `=`) in comments

# Docstrings
- Use NumPy-style docstrings.
- Avoid double backticks.
- Only add Sphinx cross-references when they are genuinely useful for navigation, such as:
  - Important public functions or classes in this package
  - Key workflow entry points
  - Custom data structures or configuration objects
  - Specialized external concepts that readers may not know
- Do not include the .py filepath in the top-level docstring of scripts

# Module layer hierarchy

```
Layer 0  core
Layer 1  config, path
Layer 2  recipe
Layer 3  io/__init__ 
Layer 4  io/readers
Layer 5  geo/*
Layer 6  io/ingester, io/table_ingester, io/aggregate, io/admin, io/transform, io/harmonize
Layer 7  viz/*
Layer 8  api.py
```
Higher-numbered layers may only import from lower-numbered layers.

# Testing

The codebase needs the `openplaces` environment. Activate it with `conda activate openplaces` when testing code.
