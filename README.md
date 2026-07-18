# Mario World Model — Phase 1: PPO Agent & Data Collection

## Project Structure
```
MarioWM/
├── env/
│   └── wrappers.py          # Preprocessing wrappers
├── train_ppo.py             # Train PPO agent
├── evaluate_agent.py        # Visualise the trained policy
├── collect_data.py          # Collect offline HDF5 dataset
├── inspect_dataset.py       # Sanity check the dataset
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
cd MarioWM
pip install -r requirements.txt
```

### 2. Train the PPO agent (~1-3h on a GPU)
```bash
# Default: 1M steps, 8 parallel envs
python train_ppo.py

# Longer run for a more capable policy
python train_ppo.py --timesteps 2_000_000 --n-envs 8
```
Checkpoints saved to `checkpoints/`. Monitor training:
```bash
tensorboard --logdir logs/ppo_mario
```

### 3. Evaluate the trained agent
```bash
python evaluate_agent.py --save-video eval.mp4
```

### 4. Collect the offline dataset
```bash
# 100k steps, 20% random actions (recommended starting point)
python collect_data.py --model checkpoints/ppo_mario_final.zip

# Larger dataset for better world model coverage
python collect_data.py --n-steps 200000 --epsilon 0.25
```

### 5. Sanity check the dataset
```bash
python inspect_dataset.py dataset/mario_1-1_100k_eps20.h5
```
Outputs a frame mosaic (`dataset/preview.png`) and a trajectory GIF (`dataset/sample_trajectory.gif`).

---

## Design Decisions

### Why ε-greedy during collection?
A pure PPO policy will nearly always win the level, producing extremely
narrow coverage of the state space. With ε=0.20, 20% of actions are random,
which naturally creates diverse situations: Mario dying, back-tracking,
standing still — all critical for the world model to generalise.

### Why HDF5?
HDF5 with `lzf` compression gives ~3–5× storage reduction over raw numpy
while supporting fast random-access indexing during training. A 100k-step
dataset at 84×84 RGB takes roughly **300-400 MB** on disk.

### Observation pipeline
| Purpose | Obs format |
|---|---|
| PPO training | Grayscale 84×84, stacked 4 frames |
| World-model source | RGB 240×256 transitions |
| World-model cache | RGB 120×128 trajectory frames |

---

# Phase 2: Deterministic Action-Conditioned World Model

The first world-model baseline predicts the next RGB frame from four recent
frames and their aligned controller actions. It is deterministic: the same
frame/action history produces the same next frame, while changing the action
can change the predicted future. The default FiLM-conditioned U-Net has
**15,774,339 parameters**.

## Inspect the source dataset

The live dataset contains one known collector reset that is not marked by a
`done`: transition index 10,000 starts a new trajectory. Include this explicit
boundary in inspection and conversion commands.

```bash
python inspect_dataset.py mario_1-1_live.h5 \
  --stats-only \
  --break-index 10000
```

## Prepare world-model data

The original HDF5 file uses 1,024-frame chunks, which are efficient for
sequential conversion but inefficient for random training windows. Convert it
once into a 120×128 NumPy memory-mapped cache:

```bash
python prepare_world_model_data.py \
  mario_1-1_live.h5 \
  dataset/mario-1-1-120x128 \
  --break-index 10000 \
  --workers 32
```

The source file remains unchanged. The output is an approximately 11.6 GiB
uncompressed cache designed to be copied to local SSD before training.

## Publish to Hugging Face Hub

Use a write-scoped token through the environment or authenticate with
`hf auth login`. The token is never supplied to the script as an argument.

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

Download the validated cache onto the H100 instance's local SSD:

```bash
hf download <namespace>/mario-1-1-world-model \
  --repo-type dataset \
  --local-dir dataset/mario-1-1-120x128
```

## Validate with a tiny overfit

Run this before committing H100 time to the full dataset:

```bash
python train_world_model.py \
  --config configs/deterministic_unet.yaml \
  --overfit-batches 4
```

Inspect `runs/deterministic-unet/previews/` and TensorBoard. The fixed subset
should be reconstructed closely before starting the full run.

```bash
tensorboard --logdir runs/deterministic-unet/tensorboard
```

## Train on one H100

```bash
python train_world_model.py --config configs/deterministic_unet.yaml
```

The default configuration uses BF16 autocast, TF32, channels-last tensors,
`torch.compile`, fused AdamW when supported, gradient clipping, EMA validation,
warmup followed by cosine decay, and batch size 64.

Resume model, optimizer, scheduler, EMA, epoch, step, and RNG state exactly:

```bash
python train_world_model.py \
  --config configs/deterministic_unet.yaml \
  --resume runs/deterministic-unet/latest.pt
```

Generated caches, runs, checkpoints, and previews are intentionally excluded
from Git.
