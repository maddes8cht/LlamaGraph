# llama-optimizer `params.txt` Reference

> **🛠️ For script usage, CLI arguments & execution details, see:**  
> [`llama-optimizer.README.md`](./llama-optimizer.README.md)

## 🔍 Overview
The `params.txt` file is the central configuration hub for `llama-optimizer.py`. It defines baseline CLI flags, optimization targets, and execution flow. The parser is deliberately simple: lines starting with `::` are directives, lines starting with `-`/`--` are CLI arguments, and `#` starts comments.

## 📜 Syntax Rules
| Rule                  | Behavior                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| `# comment`           | Ignored.                                                                 |
| `::key value`         | Parsed as configuration directive. Key is lowercased.                    |
| `-flag value` / `--flag value` | Passed `1:1` to `llama-bench`. Spaces split multi-value arguments. |
| `::optimize flag vals` | Defines optimization targets. Values are comma- **or** space-separated. |
| Whitespace            | Leading/trailing whitespace is trimmed.                                  |

## 🎛️ Directives Reference

### `::params-file <path>`
Enables recursive configuration inclusion (nesting works over multiple levels).

**Motivation:** Most of your `llama-bench` base flags (`-ctk`, `-p`, `-r`, `::output-format`, …) are identical for every model — only the model path, a few hardware flags, and the `::optimize` targets differ. Put the shared part in one base file and let each per-model file include it, so a flag change needs a single edit instead of one per model:
```ini
# base.txt — shared by all models
-ctk q8_0 -ctv q4_0
-p 2048 -n 512
-r 3
::output-format both

# qwen27b.params.txt — only what differs
::params-file base.txt
-m G:/models/qwen27b.gguf
--n-gpu-layers -1
::optimize --ubatch-size 64 128 256
```

**Merge rules (including file always wins):**
- *Directives* (`::output-dir`, `::output-format`, `::optimize-order`, …): the including file's value is used; values from the included file are inherited only when the including file doesn't set them.
- *Base args:* the included file's args come first, the including file's are appended — for flags passed twice, `llama-bench` uses the later occurrence.
- *Targets:* per flag, the including file's `::optimize` line wins; flags only defined in the included file are inherited.

Note: relative paths are resolved against the current working directory, not against the including file's directory.

### `::output-dir <path>`
*(Optional)* Default directory for result CSV/TXT files. Overridden by CLI `--output-dir`. If omitted, the current working directory is used. The directory is created if missing.

### `::output-format <both|csv|md>`
*(Optional)* Same as CLI `--output-format` (CLI wins, default `both`). `both` writes `bench_*.csv` (for LlamaGraph) + `bench_*.md` (human-readable) per run. Note: `-o` / `-oe` / `--output` / `--output-err` lines in the base args are normalized to this setting — use the directive or CLI flag instead of `-o` lines.

### `::grid`
*(Flag, no value)* Enables grid search mode, same as CLI `--grid`: all `::optimize` combinations are tested in a single `llama-bench` run instead of phase by phase. Base values of optimized flags are ignored in this mode.

### `::optimize-order <param1>,<param2>,...`
*(Optional)* Explicitly defines the execution sequence for optimization phases.  
**If omitted:** The script automatically uses the order in which `::optimize` directives appear in the file.

### `::optimize <flag> <values> [metadata...]`
Defines a parameter to optimize and the test values to run.
- **Format:** Comma-separated (`::optimize --ubatch-size 64,128,256`) **or** space-separated (`::optimize --batch-size 512 1024 2048`) — both are accepted. Tokens containing `=` (e.g. `metric=tg`) are treated as human-readable metadata and ignored.
- **Matching:** The flag must be spelled **exactly** as in your base args. The script removes that exact string (plus its value) before appending the test list — `--ubatch-size` in `::optimize` does *not* remove a `-ub` base entry, which would pass the flag twice to `llama-bench`. If the flag is absent from the base args, it is simply appended.
- **Base values are context (sequential mode):** A base value for an optimized flag is replaced during its own phase, but it stays active while *other* parameters are optimized. Example: with `--batch-size 1024` as base and `::optimize --batch-size …` in second position, the ubatch phase still runs at batch 1024. Use this to define the starting context for not-yet-optimized parameters. In `--grid` mode base values of optimized flags are always replaced and have no effect.
- **Duplicates:** If the same flag has two `::optimize` lines, the first one wins.
- **Example:**
  `::optimize --ubatch-size 64,128,256 metric=tg`
  `::optimize --batch-size 512 1024 2048`

## 🔁 Optimization Workflow
1. The script collects all `-`/`--` lines as `current_base_args`.
2. For each parameter in `optimize-order` (or implicit order):
   - Removes the existing flag/value pair from `current_base_args`.
   - Appends `<flag> <val1,val2,val3>` to the command.
   - Runs `llama-bench`.
   - Extracts the highest `t/s` for `tg` tests.
   - **Locks in** the best value back into `current_base_args` for the next phase.
3. This sequential carry-over ensures each parameter is tested under previously optimized conditions.

## 📝 Complete Example
```ini
# Base llama-bench arguments (passed 1:1)
-ctk q8_0 -ctv q4_0
-p 2048 -n 512
-fa 1 -b 1024 -ub 256
-r 3
# (no -o line needed: output format is controlled by ::output-format /
# --output-format, default both; -o/-oe lines are normalized to it)

# Optional: Explicit phase order. If missing, order of ::optimize lines is used.
# ::optimize-order dev,ngl,ubatch

# Optimization Targets (comma- or space-separated values)
::optimize ngl 22,26,30,34,38 metric=tg
::optimize ubatch 128,256,384,512
::optimize batch 512 1024 2048

# Recursive inclusion (loaded first, merged with above)
::params-file base_settings.txt
```

## 💡 Best Practices
1. **Start Broad, Then Refine:** Use wider value ranges in early phases. The sequential design prevents combinatorial explosion.
2. **Keep Metadata Readable:** You can add `# notes` or `metric=tg` after the value list for human readability. The parser safely ignores anything containing `=`.
3. **Match Flag Names Exactly:** If you use `--ngl` in your base args, the optimizer will use `--ngl`. If you use `-n`, it uses `-n`. Case and dashes must match `llama-bench`'s expected syntax.
4. **Validate Early:** Run a single phase with a small value set first. The optimizer will pass through any `llama-bench` errors verbatim, making debugging straightforward.

## ⚙️ Advanced Notes
- **No Flag Translation:** The script never converts short flags to long flags (or vice versa). What you write is what gets executed.
- **CSV Parsing:** The parser dynamically locates the header column matching your parameter name (case-insensitive, dashes ignored). It then aggregates the maximum `t/s` per value.
- **Timestamped Outputs:** Every phase creates uniquely named `bench_*` files in the output directory (`--output-dir` / `::output-dir`, else CWD): `.csv` and/or `.md` per `::output-format` / `--output-format`. This allows safe interruption and resumption without data loss.
