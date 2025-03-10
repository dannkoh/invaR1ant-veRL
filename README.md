# invaR1ant fork of veRL

_This repository is a fork of [veRL](https://github.com/volcengine/verl). It retains the powerful features of the original project while introducing custom modifications specific to invaR1ant's goals._

---

## Overview

**invaR1ant** leverages the robust reinforcement learning library originally developed in veRL. It provides a flexible and efficient RL training framework for large language models while adding specialized hooks and integrations tailored for my project needs.

### Highlights
- Forked and customized version of the original veRL library.
- Retains all core verl features including FSDP, Megatron-LM, and support for vLLM.
- Easily extendable RL algorithms and modular APIs for integrating various LLM frameworks.
- Custom modifications aimed at enhancing project-specific functionality.
- Original veRL news, updates, and contributions can be found at the [veRL repository](https://github.com/volcengine/verl).

---

## Key Features
For a full list of verl’s features and future updates, please refer to the [original documentation](https://verl.readthedocs.io).

---

## Getting Started

#### You should install these first before running the `requirements.txt` file because these packages are not included in the `requirements.txt` file

```bash
pip install torch torchvision torchaudio
```
[Torch install guide](https://pytorch.org/get-started/locally/)

idk what torch version you can use, I was on CUDA 12.4

---
```bash
pip install flash-attn --no-build-isolation
```
[Flash-Attn install guide](https://github.com/Dao-AILab/flash-attention)

I couldnt install flash-attn with build isolation, so I had to use the `--no-build-isolation` flag

---
```bash
##########################DEPENDENCIES############################
# install Python dependency
pip install ninja pybind11 torch
##################################################################

# 1. clone from GitHub
git clone --recursive https://github.com/mlc-ai/xgrammar.git && cd xgrammar
# 2. build XGrammar core and Python bindings
mkdir build && cd build
cmake .. -G Ninja
ninja
# 3. install the Python package
cd ../python
pip install .
```
[XGrammar install guide](https://xgrammar.mlc.ai/docs/start/install)

Xgrammar failed to install so i had to build it from source

---
```bash
pip install -U xformers
```
[XFormers install guide](https://github.com/facebookresearch/xformers)

I couldnt install it through the requirements.txt file so I had to install it manually

---
```bash
pip install vllm==0.7.3
```
[VLLM install guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html)

My specific hpc environmnet couldnt let me install vllm.

 > `xformers-0.0.28.post3.tar.gz` failed to install fo me when I was installing vllm.

### Project Structure

- **/examples/**: Example scripts and notebooks demonstrating project usage.
- **/verl/**: Core veRL library files.
- **/data/**: Data files and datasets used in the project (Some datasets are not included due to size constraints).
- **scripts/**: Utility scripts for running and managing the project.




---

## Custom Modifications

This fork contains several key differences from the original veRL project:

- **/verl/utils/reward_score/custom.py**: Custom reward function logic.
- **/verl/utils/dataset/rl_dataset.py**: Removal of chat template.


---

## Referencing the Original verl Project

For any original features, news updates, and community contributions not specific to this fork, please check out the [veRL official repository](https://github.com/volcengine/verl). The original project handles:
- Release news and announcements.
- Official performance tuning guides.
- Citation and acknowledgment information.
