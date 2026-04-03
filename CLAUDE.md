- Line length: 88 characters maximum
- Use the `|` union syntax for `isinstance` checks, never a tuple:
  `isinstance(x, Foo | Bar)` ✓
  `isinstance(x, (Foo, Bar))` ✗

## Module layer hierarchy

```
Layer 0  core/constants, core/schema
Layer 1  config, path
Layer 2  recipe
Layer 3  io/__init__ 
Layer 4  io/readers
Layer 5  geo/*
Layer 6  io/ingester, io/table_ingester, io/aggregate, io/admin, io/transform
Layer 7  viz/*
Layer 8  api.py
```
Higher-numbered layers may only import from lower-numbered layers.
