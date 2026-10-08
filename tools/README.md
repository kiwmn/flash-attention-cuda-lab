# Build tools

CUDA extensions compile automatically on first use. To compile one explicitly:

```bash
MAX_JOBS=2 python tools/build/build.py --scenario standard
MAX_JOBS=2 python tools/build/build.py --scenario standard --version 4
```

Use `FLASH_ATTN_LAB_VERBOSE_BUILD=1` for compiler output and
`FLASH_ATTN_LAB_PTXAS_VERBOSE=1` for register and shared-memory reports. Extensions
use C++20, fast math, and SM80 code generation, which also runs on RTX 3090 (SM86).
Local build products are written under `build/` and are not committed.

The checked-in registries already contain the full configuration sets used by
the benchmark runner. `helper/search_space.py` defines the combinations for dense
steps 4 onward and the final paged implementation. After editing that search space,
regenerate the selected kernel:

```bash
python tools/build/generate_instances.py --scenario standard --version 4
python tools/build/generate_instances.py --scenario paged
```

Add `--check` to verify the existing registry against the search space without
writing files. `helper.get_configs()` reads each selected kernel's registry, so
benchmark searches only enumerate configurations that are compiled for it.
Paged regeneration removes obsolete generated `csrc/instances/p*_*.cu` units
when the search space no longer includes their configuration groups.
When changing a final kernel, apply the same change to its numbered copy under
`previous_kernels/`.

## CUDA source style

Each implementation keeps its headers local. `utils.h` contains constants,
device annotations, numeric helpers, and input checks. `kernel_traits.cuh`
defines `ForwardKernelConfig`; `forward_kernel.cuh` defines the launch arguments
and CUDA entry point. Separate layout, tensor, and paged-access headers describe
the storage and addressing required by each implementation.

Use `DEVICE_INLINE` for device helpers, retain `constexpr` where applicable,
and write loop directives as `#pragma unroll`. The `ldmatrix_x4` and
`ldmatrix_x4_transpose` helpers take their four output registers first and the
shared-memory source pointer last.

The repository's `.clang-format` specifies four-space indentation, attached
braces, an 80-column limit, and right-aligned pointer declarations. It preserves
include order and assembly string literals. From the repository root, format
CUDA sources with clang-format 23:

```bash
rg --files kernels previous_kernels -g '*.cu' -g '*.cuh' -g '*.h' -0 \
    | xargs -0 clang-format -i
```
