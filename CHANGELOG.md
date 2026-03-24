# Changelog

Changes relative to [upstream (BGU-CS-VIL/FastJAM)](https://github.com/BGU-CS-VIL/FastJAM).

## uv / dependency management

- Added `pyproject.toml` and `uv.lock` to replace Conda-based setup with [uv](https://github.com/astral-sh/uv).
- Added `.python-version` pinning Python 3.12. (Python 3.14 causes a segfault in the HuggingFace `tokenizers` Rust extension due to a changed C API calling convention.)

## RoMa: float32 on MPS (NaN fix)

**`third_party/RoMa/romatch/models/model_zoo/roma_models.py`**

- Extended the `amp_dtype = torch.float32` fallback to cover `mps` in addition to `cpu`. On MPS, `float16` causes NaN in DINOv2 transformer attention (softmax overflow), producing all-zero certainty maps and crashing `torch.multinomial`.

## MPS device selection in training code

- Added `utilities/device_utils.py` with a `get_device()` helper that selects `cuda` > `mps` > `cpu`.
- `train.py` and all `utilities/` modules (`train_utils.py`, `models.py`, `graph_utils.py`, `plot_utils.py`, `clustering_utils.py`) were falling back to CPU on Apple Silicon. All now use `get_device()`.

## DP-Means: replaced broken `pdc-dp-means` dependency

- `pdc-dp-means==0.0.8` is broken on Python 3.12 + modern scikit-learn due to Cython ABI and internal API incompatibilities (`threadpool_limits` moved, `_cython_blas` signature changed).
- Replaced with a minimal pure-numpy Lloyd-style DP-Means in `utilities/dp_means.py`, implementing the same `fit_predict` / `cluster_centers_` interface, matching the original's behaviour: delta is compared against **squared** Euclidean distance, one new cluster is spawned per iteration (the farthest point), and best run is selected by penalized inertia (`inertia + delta * n_clusters`).
- Removed `pdc-dp-means` from `pyproject.toml`.

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

## MPS: `matrix_exp` fallback

**`train.py`**

- Added `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` at module level (before any torch import) — `aten::linalg_matrix_exp` is not implemented on MPS; this enables automatic CPU fallback for that op. Must be set before torch initializes MPS.

## Bug fix: out-of-bounds edge indices in fused graph

**`utilities/graph_utils.py`**

- Added bounds check after deduplication encoding/decoding in `build_graph_from_fused_keypoints`. The `u * num_nodes + v` encoding can produce recovered indices outside `[0, num_nodes)` due to MPS integer arithmetic edge cases; stale `-1` mapped values not caught by the pre-filter caused `edge_index.min() >= 0` assertion failures in the model.

## MPS: float32 fix in clustering

**`utilities/clustering_utils.py`**

- Added `dtype=torch.float32` when converting numpy cluster centers to tensors. MPS does not support float64, and numpy arrays default to float64.

## Bug fix: `B` derived from image count, not graph batch

**`utilities/train_utils.py`**

- `B = int(graph_data.batch.max().item()) + 1` was replaced with `B = len(image_paths)`. When some images have no matched keypoints they are absent from the graph, making `batch.max() + 1 < num_images`, causing an out-of-bounds index into the keypoint matrix.

## RoMa match caching

**`utilities/matchers_utils.py`**, **`train.py`**

- Added `matches_cache_key`, `save_matches_cache`, `load_matches_cache` to `matchers_utils.py`. On subsequent runs with the same images and matching parameters, matches are loaded from `<data_folder>/.matches_cache.npz` instead of re-running RoMa (the slow step).
