# LlamaGraph

**llamagraph** is a desktop visualisation tool for benchmarking results produced by
[llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
from the [llama.cpp](https://github.com/ggml-org/llama.cpp) project.

It reads the CSV output files that `llama-bench` generates and plots prompt-processing
(PP) and token-generation (TG) throughput — or latency — as interactive 2-D line
charts or 3-D surface plots.  Multiple CSV files can be compared simultaneously.

When you run an optimizer like [llama-optimus](https://github.com/BrunoArsioli/llama-optimus) or do manual parameter sweeps with `llama-bench`, you usually end up with a bunch of numbers. Sometimes an optimizer even gives you a single “best” value.

**The problem is:** you often don’t really **understand** what is happening.  
You see the final TG number is great (or not, unfortunately), but you don’t see how the parameters interact, where sudden drops occur, or why PP sometimes collapses even though TG looks fine. After staring at raw CSV files or tables for a while, it’s still hard to build a mental picture of “what is going on here”.

That’s why I made LlamaGraph.

It lets you visually explore the parameter space — any combination of `--n-gpu-layers`, `--batch-size`, `--ubatch-size` and other flags — and immediately see how PP and TG behave together.

![LlamaGraph](./media/banner800.jpg)

### What it does

- Loads one or more `llama-bench` CSV files (plus the human-readable `.md` tables)
- Shows interactive 2D and 3D plots of any parameters against performance
- Features a **right sidebar** that lets you filter any dimension not currently shown on the axes (this quickly became the most useful part, especially in 2D)
- **Comparison filter** in the left sidebar: narrow the file list to a single build number / model file before comparing runs
- Click any data point for a tooltip with both metrics (combined PP+TG view); in 3D a connector line bridges to the other surface
- Supports toggling PP/TG per file, per-series normalization, different Z-modes (including absolute PP/TG scales), surface styles, Cubic/Linear refinement with anti-overshoot clamp and gap masking, error bars, projections, level plane and more
- 3D surfaces share one depth-sorted collection so overlapping PP/TG render correctly
- Axis ticks show actually measured values instead of decimal auto-ticks

The interface is meant to be mostly self-explanatory after a bit of clicking around. A proper user guide will come later.

### Keyboard shortcuts

- `Ctrl+T`: toggle tokens/s ↔ latency · `Ctrl+R`: refresh file list · `Esc`: quit
- 3-D view, Blender-style (main digit row and numpad, 3-D mode only):
  - `1` front view (X-Z plane), `3` side view (Y-Z plane), `7` top view
    (X-Y plane) — all three switch to parallel projection automatically
  - `5` toggle perspective ↔ parallel projection from any angle
  - `0` home view (restores perspective automatically)
- Perspective is the default camera. **Ortho** (upper toolbar, `📷 Ortho`)
  selects parallel projection with no foreshortening — combined with the
  Colormap surface style, the top view (`7`) reads as a color-coded
  heightmap.
- Digit keys stay silent while typing in inputs (level field, dropdowns)
  or navigating the file/filter lists.

### Examples

This is just a short introduction by example.  
A more comprehensive, genuine guide will follow soon.

**Typical file view in 2D**  

![Multi-file 2D comparison](./media/this-is-expected.webp)
This one is what we expect when reaching the vram limit: Token gen and prompt processing remain at a plateau since more layers cannot be offloaded to GPU. With your current hardware, obviously there is nothing you can do to improve performance, here is your limit.

**Unexpected sudden performance drop of prompt processing**  
Even when TG looks reasonable, PP can collapse at certain layer counts. Visualizing both makes these problems obvious.

![PP drop at low n-gpu-layers](./media/pp-drop-at-18-ngl.webp)
Most optimisers will recommend (and have recommended) setting `-ngl 18` here, as this delivers the best tokens per second.  
However, the processing speed at this exact value is horribly slow – I spent minutes waiting for the first tokens to appear.
Am I the only one this has happened to?

Is this a mere outlier?  
**3D view of the full parameter space**  
X = n-gpu-layers, Y = n-batch, filtered by ubatch, showing normalized PP & TG.

![3D Parameter Space](./media/drop-is-real-accross-all-batch-values.webp)
The 3D view shows that this decrease at `ngl 18` is in fact real for all values of `batch`.  
Something really strange is going on here, and it's not a limitation of current hardware. This is just an obscure and unfortunate combination of parameters that you want to avoid on your hardware for this model.

### Bundled tools

The [`tools/`](./tools/) directory contains standalone helpers that produce the data LlamaGraph plots:

- **[llama-optimizer](./tools/llama-optimizer/)** — sequential/grid wrapper around `llama-bench` that finds optimal parameters for a GGUF model and writes `bench_*.csv` (+ human-readable `bench_*.md`) straight for LlamaGraph. See `tools/llama-optimizer/llama-optimizer.README.md` for usage and the `params.txt` reference.

### Startup configuration

LlamaGraph starts with built-in defaults (2D tokens/s view) when no config
is present, so existing start behavior is unchanged.

- Place a `llamagraph.config.yml` file next to `llamagraph.py` to apply it
  automatically at startup (this file is git-ignored and never committed).
- Use `python llamagraph.py --config <file>` to load any other config file,
  for example the committed `example.llamagraph.config.yml`, which documents
  every supported key (default data path, latency vs. tokens/s, `.md` usage,
  normalization, 3D mode, level plane + value, surface visibility + style,
  subdivision, interpolation, mask/wireframe/errors/projections, dolly,
  Z label mode, unify, PP/TG visibility).
- Use `python llamagraph.py --no-config` to ignore every config file.
- Explicit CLI options always win over config values (for example `--ns` /
  `--ts`, `--no-md` / `--md`, `--normalize` / `--no-normalize`,
  `--mode-3d` / `--no-mode-3d`, `--level-value`, `--surface-style`,
  `--subdiv-level`, `--interp-method`, `--z-label-mode`).

No extra dependency is needed: the flat `key: value` YAML subset is parsed
with the standard library only.

### Current status

This is an early but already very usable version (v0.1).  
It has helped me a lot when optimizers or manual tuning left me confused about why performance behaved the way it did.
This isn’t even a release yet. These are the first commits of the project.
It does contain some strange bugs that need to be fixed.

### Future ideas (maybe, no promises)

- Find smarter ways to fill data gaps with fewer benchmark runs.
- Support for other llama-bench output formats (JSON, JSONL, SQL)
- Better tools for comparing performance changes between llama.cpp PRs

Feedback and pull requests are very welcome — especially ideas on how to make the visual exploration even more useful when working with auto-optimizers.

Enjoy exploring your benchmark data!