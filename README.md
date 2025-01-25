Traffic Signal Optimization System Using Deep Q-Network (DQN)
Overview
This project aims to optimize traffic signal control at a four-way intersection using Deep Q-Networks (DQN) and SUMO (Simulation of Urban MObility). By applying reinforcement learning, the system learns optimal traffic signal policies to minimize vehicle and pedestrian waiting times and reduce traffic congestion. The agent interacts with the SUMO environment, learning in real-time to optimize the control of traffic lights based on various states and actions.

Key Features
Reinforcement Learning-Based Optimization: Implements DQN with replay buffers and target networks for stable and efficient learning.
Realistic Environment: Simulates random vehicle and pedestrian flows every 600 seconds to mimic real-world traffic dynamics.
Performance Metrics: Tracks cumulative waiting time, average waiting time per vehicle, and total entities per episode to evaluate system performance.
Graphical Outputs: Produces visualized training results, including waiting time improvements.
System Architecture
Input: Traffic state vectors (e.g., vehicle queue lengths, signal states, elapsed signal time).
Hidden Layers: Three fully connected layers with ReLU activation.
Output: Signal actions (maintain current signal or switch to the next signal).
Training Environment: Runs on Ubuntu 20.04 with GPU acceleration (NVIDIA RTX 3090) for fast training.
Installation and Execution
Prerequisites
Install SUMO:

bash
복사
편집
sudo apt-get install sumo sumo-tools sumo-doc
Install Required Python Libraries:

bash
복사
편집
pip install sumolib traci
Setup and Execution
Place the following files in the same directory:

case_b_new.rou.xml
case_b.add.xml
case_b.net.xml
case_b.sumocfg
caseb_ped6.py
Update the paths in caseb_ped6.py:

sumocfg_dir: Path to the case_b.sumocfg file.
route_dir: Path to the case_b_new.rou.xml file.
result_dir: Directory to save training result graphs.
Run the training script:

bash
복사
편집
python caseb_ped6.py
Outputs
The training results, including graphs for cumulative waiting time, average waiting time per vehicle, and total entities per episode, will be saved in the specified result_dir.
Results
Initial average waiting time: 61 seconds.
Final average waiting time after training: 33 seconds.
Average waiting time reduction: 46%.
The agent stabilizes after approximately 10 episodes and consistently learns efficient signal control strategies.
Future Work
Extend the system to optimize multiple intersections with cooperative agents.
Integrate additional real-world factors such as weather, pedestrian density, or road conditions.
Explore advanced reinforcement learning algorithms for further optimization.
