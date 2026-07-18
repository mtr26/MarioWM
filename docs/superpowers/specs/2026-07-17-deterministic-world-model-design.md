# Deterministic Mario World Model Design

## Goal

Build a deterministic, action-conditioned next-frame world model for Super Mario Bros. World 1-1, together with a high-throughput training data cache, H100-oriented training pipeline, and optional Hugging Face Hub publication workflow.

The first success criterion is a recognizable short autoregressive rollout whose future changes when the controller actions change. Remote PPO checkpoints are not required for the baseline.

## Source Dataset

The source file is `mario_1-1_live.h5` and remains immutable. It contains 267,536 transitions with this schema:

- `observations`: `(267536, 240, 256, 3)`, `uint8`
- `next_obs`: `(267536, 240, 256, 3)`, `uint8`
- `actions`: `(267536,)`, `int32`
- `rewards`: `(267536,)`, `float32`
- `dones`: `(267536,)`, boolean

Action `actions[t]` was passed to `env.step()` to produce `next_obs[t]`. Each transition represents four repeated NES emulator frames because collection used frame skip 4.

There are 1,535 complete episodes and one trailing partial trajectory. Index 10,000 starts a new collection environment even though `dones[9999]` is false. The conversion pipeline must treat 10,000 as an explicit trajectory start and must not create a sample spanning 9,999 to 10,000.

## Data Format Decision

Training uses an uncompressed NumPy `.npy` cache rather than random reads from the source HDF5 file. The source HDF5 uses 1,024-frame chunks, approximately 180 MiB uncompressed per frame dataset chunk. Measured sequential reads are fast, while sparse reads are too slow to keep an H100 occupied.

The cache is stored as:

```text
<output-directory>/
├── frames.npy
├── actions.npy
├── rewards.npy
├── episode_offsets.npy
├── episode_splits.npy
├── source_transition_indices.npy
├── metadata.json
└── README.md
```

`frames.npy` is a C-contiguous `uint8` array with shape `(M, 120, 128, 3)`. Each trajectory of `L` transitions contributes exactly `L+1` frames: its `L` observations followed by the final transition's `next_obs`. This removes the near-total duplication between source `observations` and `next_obs`. At the observed dataset size, the frame cache is approximately 11.6 GiB.

`actions.npy`, `rewards.npy`, and `source_transition_indices.npy` contain one entry per transition. `episode_offsets.npy` has length `E+1` and indexes transition offsets; a trajectory `e` occupies transitions `[episode_offsets[e], episode_offsets[e+1])`. Its frame start is derived as `episode_offsets[e] + e`, because every previous trajectory contributes one additional terminal frame.

`episode_splits.npy` contains one `uint8` value per trajectory: 0 for train, 1 for validation, and 2 for test. Splits are assigned deterministically with a configurable seed and default proportions 90%, 5%, and 5%. Complete trajectories or valid partial trajectory segments remain wholly inside one split.

`metadata.json` records source file size, source schema, output shapes, resize method, history length, action names, frame skip, explicit break indices, split seed, split fractions, creation timestamp, package versions, and SHA-256 hashes for all output artifacts. `README.md` is a Hugging Face dataset card describing alignment, format, loading, and limitations.

## Conversion Pipeline

`prepare_world_model_data.py` performs these steps:

1. Validate that all five source datasets exist, have equal first dimensions, and match the expected dtypes and RGB shapes.
2. Read actions, rewards, and done flags into memory.
3. Construct trajectory boundaries from `dones`, the end of the file, and explicit new-trajectory indices supplied through repeatable `--break-index` options. The project command uses `--break-index 10000`.
4. Allocate final `.npy` files through `numpy.lib.format.open_memmap` only after all output shapes are known.
5. Read source frame arrays sequentially in native HDF5 chunks. Resize RGB frames from 240×256 to 120×128 using OpenCV `INTER_AREA`.
6. Write each source observation once and append the final `next_obs` at every trajectory boundary.
7. Flush and reopen every array read-only, then validate shapes, dtypes, offsets, split isolation, action range, random aligned windows, and the explicit 10,000 boundary.
8. Calculate artifact SHA-256 hashes and atomically write `metadata.json` and `README.md`.

Conversion writes to a new output directory and refuses to overwrite a non-empty directory. A temporary sibling directory is renamed to the requested output only after validation succeeds. Interrupted temporary directories are left in place and identified clearly so the operator can inspect or remove them.

The conversion uses a bounded worker pool for resizing. HDF5 access remains in the main process because h5py file handles are not shared across workers. The default worker count is `min(16, os.cpu_count())`, configurable by `--workers`; a 32-vCPU conversion machine can use `--workers 32`.

## Hugging Face Hub Publication

Publication is optional and occurs only when `--hf-repo namespace/name` is passed. Authentication uses the Hugging Face credential store or the `HF_TOKEN` environment variable. Tokens are never accepted as a command-line argument, written to metadata, or printed.

The script uses `huggingface_hub.HfApi` to:

1. Create or reuse `namespace/name` with `repo_type="dataset"`.
2. Apply private visibility when `--hf-private` is passed.
3. Upload the completed output directory with `upload_folder`, using Hugging Face's Xet-backed resumable transfer path.
4. Print the resulting dataset URL only after upload completion.

The cache is uploaded as ordinary repository files rather than converted into Hugging Face Datasets/Arrow. Training first downloads a local snapshot and then memory-maps the `.npy` files. This preserves the intended local-SSD access pattern.

The default is local conversion only. Upload never begins if conversion or post-write validation fails.

## Training Sample Interface

With history length 4, a valid local transition index `t` satisfies `3 <= t < L` within one trajectory. It produces:

```text
frames:  x[t-3], x[t-2], x[t-1], x[t] -> float32/bfloat16 (12, 120, 128)
actions: a[t-3], a[t-2], a[t-1], a[t] -> int64 (4,)
target:  x[t+1]                        -> float32/bfloat16 (3, 120, 128)
```

Frames are converted from channels-last `uint8` to channels-first floating point in `[0, 1]` inside DataLoader workers. No resize or image decoding occurs during training. Valid sample indices are built from trajectory offsets, so samples cannot cross a death, reset, file end, or the explicit collection discontinuity.

## Model

The baseline is a deterministic action-conditioned U-Net:

- Input: four RGB frames concatenated into 12 channels.
- Output: one RGB frame with 3 channels and values in `[0, 1]`.
- Base width: 64.
- Channel multipliers: `[1, 2, 3, 4]`.
- Residual blocks: two per resolution.
- Normalization: GroupNorm.
- Activation: SiLU.
- Conditioning: learned action and action-position embeddings for all four actions, combined by an MLP and injected into every residual block with FiLM scale and shift.
- Attention: none in the deterministic baseline.
- Objective: L1 reconstruction loss.

Deterministic means identical frame and action histories yield the same predicted next frame at inference. It does not mean action-independent: changing the supplied action must be able to change the prediction. A later conditional diffusion stage may represent multiple plausible futures after this baseline proves alignment and controllability.

## H100 Training Pipeline

`train_world_model.py` loads a YAML configuration and supports explicit CLI overrides. The default training configuration uses:

- bfloat16 autocast on CUDA;
- TF32 matrix multiplication and convolution support;
- channels-last model and image tensors;
- optional `torch.compile`;
- AdamW;
- linear learning-rate warmup followed by cosine decay;
- gradient norm clipping;
- exponential moving average weights;
- configurable batch size, default 64;
- pinned memory, persistent DataLoader workers, and configurable prefetching;
- TensorBoard scalar and image logging;
- validation L1, MSE, and L1 grouped by current action;
- best and latest checkpoints containing model, EMA, optimizer, scheduler, configuration, epoch, global step, and RNG state;
- exact resume from either checkpoint;
- a tiny-overfit mode that restricts training to a fixed number of batches.

Periodic prediction grids compare the last context frame, target, prediction, and absolute error. A short autoregressive validation rollout uses recorded actions and confirms the recursive inference path, while the dedicated interactive controller program remains a subsequent deliverable.

## Commands

Local conversion:

```bash
python prepare_world_model_data.py \
  mario_1-1_live.h5 \
  dataset/mario-1-1-120x128 \
  --break-index 10000 \
  --workers 32
```

Private Hugging Face dataset publication:

```bash
export HF_TOKEN=<write-token>
python prepare_world_model_data.py \
  mario_1-1_live.h5 \
  dataset/mario-1-1-120x128 \
  --break-index 10000 \
  --workers 32 \
  --hf-repo <namespace>/mario-1-1-world-model \
  --hf-private
```

Tiny overfit and full training:

```bash
python train_world_model.py \
  --config configs/deterministic_unet.yaml \
  --overfit-batches 4

python train_world_model.py \
  --config configs/deterministic_unet.yaml
```

## Verification

Automated tests use small synthetic HDF5 files and temporary output directories. They verify:

- schema and dtype failures are reported before allocation;
- trajectory offsets include done boundaries and explicit break indices;
- no training window crosses a boundary;
- frame/action/target alignment is exact;
- aspect ratio and output shapes are correct;
- deterministic split assignment is stable;
- output hashes and metadata agree with generated files;
- Hub publication is not called after failed validation and uses dataset repository semantics when mocked;
- model output shape and range are correct;
- changing action inputs reaches the conditioning path;
- one CPU training step produces a finite loss and gradients;
- checkpoint save and resume restore the exact global step.

Before full training, the operator must run the tiny-overfit command and inspect its prediction grid. Full training begins only after the tiny subset loss falls substantially and the predicted frames visibly track their targets.

## Out of Scope

- Recollecting data from remote PPO checkpoints.
- Conditional diffusion training.
- A keyboard or controller user interface.
- Multi-GPU distributed training.
- Long-horizon evaluation beyond the small validation rollout embedded in training.
