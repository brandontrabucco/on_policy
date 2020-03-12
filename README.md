# On Policy Hierarchal RL Algorithms

This package implements on policy reinforcement learning that supports training policies arranged in a graph structure. Two special cases are multi-agent rl and hierarchical rl. Have Fun! -Brandon

# Installation

You may install this package by cloning it from github and using pip.

```
git clone https://github.com/brandontrabucco/on_policy
pip install -e on_policy
```

# Performance

Below is an evaluation of the Policy Gradient implementation on `InvertedPendulum-v2` with a maximum episode length of 256 and a batch size of 6144. The maximum achievable reward is 256.

<p align="center">
    <img src="./img/inverted_pendulum_pg.svg" alt="Inverted Pendulum PG" width="600" height="300" />
</p>

Below is an evaluation of the Proximal Policy Optimization implementation on `InvertedPendulum-v2` with a maximum episode length of 256 and a batch size of 6144. The maximum achievable reward is 256.

<p align="center">
    <img src="./img/inverted_pendulum_ppo.svg" alt="Inverted Pendulum PPO" width="600" height="300" />
</p>
