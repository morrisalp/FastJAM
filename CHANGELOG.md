# Changelog

Changes relative to [upstream (BGU-CS-VIL/FastJAM)](https://github.com/BGU-CS-VIL/FastJAM).

## uv / dependency management

- Added `pyproject.toml` and `uv.lock` to replace Conda-based setup with [uv](https://github.com/astral-sh/uv).
- Added `.python-version` pinning Python 3.12. (Python 3.14 causes a segfault in the HuggingFace `tokenizers` Rust extension due to a changed C API calling convention.)

## Grounded-SAM subprocess working directory fix

**`prepare_data/prepare_spair.py`, `prepare_cub_class.py`, `prepare_cub_subsets.py`**

- Added `cwd=script_path.parent` to the `subprocess.run` call that launches Grounded-SAM scripts. Without this, the scripts ran from the repo root, causing Python to import the local `transformers/` directory (containing `homography_transformer.py` etc.) instead of the installed HuggingFace `transformers` package, resulting in a segfault.

## Apple Silicon (MPS) support for Grounded-SAM

**`third_party/Grounded-Segment-Anything/grounded_sam_spair_split.py`**

- Device selection now prefers `mps` on Apple Silicon, falling back to `cpu`. Both GroundingDINO and SAM work correctly on MPS with Python 3.12.
- `Model(...)` is now passed the selected device explicitly.
- Fixed `REPO_ROOT` path: `parents[2]` → `parents[1]` (was resolving to `repos/` instead of `FastJAM/`).

**`third_party/Grounded-Segment-Anything/GroundingDINO/groundingdino/util/inference.py`**

- Changed default `device` in `load_model`, `predict`, and `Model.__init__` from `"cuda"` to `"cpu"` as a safety fallback (device is always passed explicitly by the scripts anyway).
