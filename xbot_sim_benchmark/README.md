# Xbot Benchmark

This repository contains a cleaned Xbot simulation benchmark built on Isaac Lab.

## Installation

### 1. Create a Python environment

This project expects Python `3.10`.

```bash
conda create -n humanoid python=3.10
conda activate humanoid
```

### 2. Install PyTorch

Install a PyTorch build compatible with your CUDA driver. Example for CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Isaac Lab with Isaac Sim

```bash
pip install isaaclab[isaacsim,all]==2.0.2 --extra-index-url https://pypi.nvidia.com
```

### 4. Install this repository

```bash
cd /path/to/active-adaptation-xbot_cleanup
pip install -e .
```

## Usage

### Run the simulation with the VLA loop

```bash
python scripts/vla_test_xbot.py task=Xbot/XbotPAP vla.server=tcp://<server-ip>:<port>
```

This launches the Xbot tabletop benchmark, renders the three cameras, queries the VLA server over ZMQ, decodes the returned 38-dim action, and executes it in simulation.

Example:

```bash
python scripts/vla_test_xbot.py task=Xbot/XbotPAP vla.server=tcp://192.168.1.10:8003
```