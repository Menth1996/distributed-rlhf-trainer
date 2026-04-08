# Distributed RLHF Trainer

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Ray](https://img.shields.io/badge/Ray-Distributed-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

A highly scalable framework for aligning Large Language Models using Reinforcement Learning from Human Feedback (RLHF), built on top of PyTorch and Ray.

## Features
- Proximal Policy Optimization (PPO) implementation optimized for LLMs
- Distributed reward model training and inference
- Seamless integration with HuggingFace models
- Memory-efficient experience replay buffers

## Architecture
The system separates the Actor, Critic, Reward, and Reference models across different GPU nodes using Ray actors to maximize throughput and minimize memory bottlenecks.

## Getting Started
```bash
pip install -r requirements.txt
python scripts/train_ppo.py --config configs/llama_7b_rlhf.yaml
```
