# Traffic Signal Optimization System Using Deep Q-Network (DQN)

## Overview

This project aims to optimize traffic signal control at a four-way intersection using Deep Q-Networks (DQN) and SUMO (Simulation of Urban MObility). By applying reinforcement learning, the system learns optimal traffic signal policies to minimize vehicle and pedestrian waiting times and reduce traffic congestion. The agent interacts with the SUMO environment, learning in real-time to optimize the control of traffic lights based on various states and actions.

## Key Features

- **Reinforcement Learning-Based Optimization**: Implements DQN with replay buffers and target networks for stable and efficient learning.
- **Realistic Environment**: Simulates random vehicle and pedestrian flows every 600 seconds to mimic real-world traffic dynamics.
- **Performance Metrics**: Tracks cumulative waiting time, average waiting time per vehicle, and total entities per episode to evaluate system performance.
- **Graphical Outputs**: Produces visualized training results, including waiting time improvements.

## System Architecture

- **Input**: Traffic state vectors (e.g., vehicle queue lengths, signal states, elapsed signal time).
- **Hidden Layers**: Three fully connected layers with ReLU activation.
- **Output**: Signal actions (maintain current signal or switch to the next signal).
- **Training Environment**: Runs on Ubuntu 20.04 with GPU acceleration (NVIDIA RTX 3090) for fast training.

## Results

- **Initial Average Waiting Time**: 61 seconds.
- **Final Average Waiting Time After Training**: 33 seconds.
- **Average Waiting Time Reduction**: 46%.
- The agent stabilizes after approximately 10 episodes and consistently learns efficient signal control strategies.

## Future Work

- Extend the system to optimize multiple intersections with cooperative agents.
- Integrate additional real-world factors such as weather, pedestrian density, or road conditions.
- Explore advanced reinforcement learning algorithms for further optimization.
