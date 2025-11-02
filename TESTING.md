# Testing Guidelines

## Running tests
```bash
python -m pytest  # Runs all tests with coverage
```

## Coverage target
- Minimum: 93%
- Report: `htmlcov/index.html`

## Test categories
- `test_*_error_paths.py` - error handling
- `test_*_more.py` - edge cases
- Standard tests - happy paths

