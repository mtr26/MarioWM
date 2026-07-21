# Conversion Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display truthful `tqdm` progress for transition conversion and cache hashing.

**Architecture:** `world_model.conversion` will own both progress bars. The conversion loop advances by completed transitions, while `_sha256` optionally advances a shared byte-count bar used only during cache creation; validation remains quiet.

**Tech Stack:** Python 3.10+, tqdm 4.66+, pytest 8+, NumPy, h5py

## Global Constraints

- Preserve cache contents, metadata, validation, and atomic publication behavior.
- Write progress to stderr using standard `tqdm` behavior.
- Keep Hugging Face upload progress under `huggingface_hub` control.
- Test real progress output without mocking `tqdm`.

---

### Task 1: Add conversion and hashing progress

**Files:**
- Modify: `world_model/conversion.py`
- Test: `tests/test_conversion.py`

**Interfaces:**
- Consumes: `ConversionConfig`, `_write_cache()`, and `_sha256()`.
- Produces: `_sha256(path, block_size=..., progress=None) -> str` plus two stderr progress bars during `convert_dataset()`.

- [ ] **Step 1: Write the failing integration test**

Add this test to `tests/test_conversion.py`:

```python
def test_convert_dataset_reports_conversion_and_hashing_progress(
    synthetic_h5, tmp_path, capsys
):
    convert_dataset(
        ConversionConfig(
            input_path=synthetic_h5,
            output_dir=tmp_path / "cache",
            height=4,
            width=6,
            history=2,
            break_indices=(6,),
            workers=1,
        )
    )

    stderr = capsys.readouterr().err
    assert "Converting transitions" in stderr
    assert "Hashing cache" in stderr
    assert stderr.count("100%") >= 2
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_conversion.py::test_convert_dataset_reports_conversion_and_hashing_progress -q
```

Expected: FAIL because stderr contains neither progress label.

- [ ] **Step 3: Implement the minimal progress behavior**

Import `tqdm`:

```python
from tqdm.auto import tqdm
```

Allow hashing to advance an optional progress object:

```python
def _sha256(
    path: Path,
    block_size: int = 8 * 1024 * 1024,
    progress: Any = None,
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
            if progress is not None:
                progress.update(len(block))
    return digest.hexdigest()
```

Wrap the HDF5 block loop in a transition-count bar and advance it only after a
block is fully written:

```python
with tqdm(
    total=n_transitions,
    desc="Converting transitions",
    unit="transition",
    unit_scale=True,
    dynamic_ncols=True,
) as conversion_progress:
    for block_start in range(0, n_transitions, source_chunk):
        # Existing conversion block body remains unchanged.
        conversion_progress.update(block_end - block_start)
```

Hash generated arrays with one shared byte-count bar:

```python
hash_total = sum((temporary_dir / name).stat().st_size for name in ARRAY_FILES)
with tqdm(
    total=hash_total,
    desc="Hashing cache",
    unit="B",
    unit_scale=True,
    unit_divisor=1024,
    dynamic_ncols=True,
) as hash_progress:
    metadata["sha256"] = {
        name: _sha256(temporary_dir / name, progress=hash_progress)
        for name in ARRAY_FILES
    }
```

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
.venv/bin/python -m pytest tests/test_conversion.py::test_convert_dataset_reports_conversion_and_hashing_progress -q
.venv/bin/python -m pytest -q
git diff --check
```

Expected: focused test passes, all tests pass, and `git diff --check` is silent.

- [ ] **Step 5: Run a bounded real-data conversion**

Run into a new temporary directory:

```bash
.venv/bin/python -u prepare_world_model_data.py \
  --input_h5 mario_1-1_live.h5 \
  --output_dir /tmp/mario-progress-smoke/cache \
  --break-index 10000 \
  --workers 8 \
  --limit-transitions 4096
```

Expected: both labeled progress bars reach 100% and the cache validates.

- [ ] **Step 6: Commit**

```bash
git add world_model/conversion.py tests/test_conversion.py
git commit -m "feat: show cache preparation progress"
```
