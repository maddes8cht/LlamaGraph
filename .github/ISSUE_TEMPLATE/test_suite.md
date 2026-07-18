# Add pytest test suite for core modules

## Motivation
The LlamaGraph project currently lacks automated tests. As the codebase grows, manual testing becomes insufficient to catch regressions, especially in core logic like CSV parsing and data filtering. Adding a test suite will improve code quality, facilitate refactoring, and increase confidence in releases.

## Proposed Scope
Start by adding unit tests for the following modules:
- `utils/csv_parser.py`: CSV parsing, parameter detection, and data cleaning functions.
- `model/benchmark_model.py`: Data model, filtering, and aggregation logic.

These modules are highly testable due to the MVC separation and contain the most critical business logic.

## Test Organization
- Create a `tests/` directory at the repository root.
- Follow pytest conventions: test files named `test_*.py`.
- Add `pytest` to `requirements.txt` (or `dev-requirements.txt` if we separate dependencies).
- Consider adding `pytest-tkinter` for future GUI tests, though initial focus is on non-GUI components.

## Initial Test Ideas
### csv_parser.py
- Test parsing of various llama-bench CSV formats.
- Test detection of GPU configurations and parameters.
- Test handling of malformed or missing data.

### benchmark_model.py
- Test loading of parsed CSV data into internal structures.
- Test filtering by GPU, parameter count, quantization, etc.
- Test aggregation functions (e.g., averaging, percentiles).

## Next Steps
1. Add `pytest` to dependencies.
2. Create `tests/` directory and initial test files.
3. Implement tests for the most critical functions.
4. Set up a CI workflow (e.g., GitHub Actions) to run tests on push/pull request.

## Benefits
- Early detection of bugs in core logic.
- Safer refactoring and feature additions.
- Documentation of expected behavior through test examples.

---
*Label: enhancement, testing*