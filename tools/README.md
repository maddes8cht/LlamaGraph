# tools/

Standalone helper scripts that produce (or pre-process) the benchmark data LlamaGraph visualizes.
Each tool lives in its own subdirectory so further helpers can be added the same way.

| Tool | Contents | Purpose |
|------|----------|---------|
| [`llama-optimizer/`](./llama-optimizer/) | `llama-optimizer.py`, `llama-optimizer.README.md`, `llama-optimizer.params.README.md` | Sequential/grid wrapper around `llama-bench` that finds optimal inference parameters and writes `bench_*.csv` (+ `bench_*.md`) for LlamaGraph |

## Conventions for adding a tool

1. Create `tools/<tool-name>/` and put the script plus its docs inside.
2. Keep original filenames where the script is vendored from elsewhere, so diffs against the source stay trivial.
3. Document the tool in its own directory (a `README.md` is enough for small tools).
4. Add a row to the table above.
5. Pure helper logic should be covered by tests in the top-level `tests/` directory
   (see `tests/test_llama_optimizer.py` — the hyphenated script name is loaded
   via `importlib`, since it is not a valid Python module name).
6. Mention user-facing tools in the main `README.md` (`Bundled tools` section).

## Notes

- Tools are intentionally decoupled from the `model/` / `view/` / `presenter/` packages:
  they run on their own (usually from a checkout of `llama.cpp`) and only exchange
  files (`bench_*.csv`, `bench_*.md`) with LlamaGraph.
- `tools/llama-optimizer/` is a vendored copy of the script maintained alongside
  `llama.cpp` tooling; when syncing, copy the files over and re-run
  `python -m pytest tests/test_llama_optimizer.py`.
