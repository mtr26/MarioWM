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
| World model training | RGB 84×84, single frame |
