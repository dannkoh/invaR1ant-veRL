# ConStruct Fork of verl

_This repository is a fork of [verl](https://github.com/volcengine/verl). It retains the powerful features of the original project while introducing custom modifications specific to ConStruct's goals._

---

## Overview

**ConStruct** leverages the robust reinforcement learning library originally developed in verl. It provides a flexible and efficient RL training framework for large language models while adding specialized hooks and integrations tailored for my project needs.

### Highlights
- Forked and customized version of the original verl library.
- Retains all core verl features including FSDP, Megatron-LM, and support for vLLM.
- Easily extendable RL algorithms and modular APIs for integrating various LLM frameworks.
- Custom modifications aimed at enhancing project-specific functionality.
- Original verl news, updates, and contributions can be found at the [verl repository](https://github.com/volcengine/verl).

---

## Key Features
For a full list of verl’s features and future updates, please refer to the [original documentation](https://verl.readthedocs.io).

---

## Getting Started

### Project Structure

- **/docs/**: Custom documentation for project-specific modifications.
- **/examples/**: Example scripts and notebooks demonstrating project usage.
- Core training modules and hooks were modified to integrate with my project’s logic while keeping the original verl’s structure intact.

---

## Custom Modifications

This fork contains several key differences from the original verl project:

- **Project-Specific Hooks:** Additional code to integrate with internal data pipelines.
- **Modified Configurations:** Preconfigured settings tuned for our deployment environments.
- **Extended API Endpoints:** Custom APIs added to support unique workflow requirements.

For code differences and detailed descriptions, please see the CHANGELOG.

---

## Referencing the Original verl Project

For any original features, news updates, and community contributions not specific to this fork, please check out the [verl official repository](https://github.com/volcengine/verl). The original project handles:
- Release news and announcements.
- Official performance tuning guides.
- Citation and acknowledgment information.

---

## Contribution Guide

Contributions to this fork are welcome. If you want to improve project-specific code or add new features:
1. Fork this repository.
2. Create a branch for your changes.
3. Submit a pull request describing your modifications.

For code formatting, we use [yapf](https://github.com/google/yapf):
```bash
pip install yapf --upgrade
bash scripts/format.sh