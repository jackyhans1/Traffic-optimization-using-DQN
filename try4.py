import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import traci
import numpy as np
import collections
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

buffer_limit = 100000
batch_size = 64
gamma = 0.99

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, 400)
        self.output = nn.Linear(400, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.output(x))
        return x

    def sample_action(self, obs, traffic_time, phase_min_time, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            out = self.forward(obs).argmax().item()
            if out == 1 and traffic_time < phase_min_time:
                out = 0
            return out

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = zip(*mini_batch)
        return np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst)

    def size(self):
        return len(self.buffer)

def overwrite_route_file(file_path, num_repeats):
    with open(file_path, "w", encoding="UTF-8") as file:
        file.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
""")
        begin = 0
        step = 3600
        for i in range(num_repeats):
            end = (i + 1) * step
            flow_number_1 = random.randint(0, 300)
            flow_number_2 = random.randint(0, 300)
            flow_number_3 = random.randint(0, 300)
            flow_number_4 = random.randint(0, 300)
            flow = f"""    <flow id="F{i*4+1}" begin="{begin}" departLane="random" departSpeed="1" fromTaz="taz_0" toTaz="taz_1" end="{end}" number="{flow_number_1}"/>
    <flow id="F{i*4+2}" begin="{begin}" departLane="random" fromTaz="taz_1" departSpeed="1" toTaz="taz_0" end="{end}" number="{flow_number_2}"/>
    <flow id="F{i*4+3}" begin="{begin}" departLane="random" fromTaz="taz_2" departSpeed="1" toTaz="taz_0" end="{end}" number="{flow_number_3}"/>
    <flow id="F{i*4+4}" begin="{begin}" departLane="random" fromTaz="taz_1" departSpeed="1" toTaz="taz_2" end="{end}" number="{flow_number_4}"/>"""
            file.write(flow + "\n")
            begin = end + 1
        file.write("</routes>\n")

def get_halted_vehicles_vector(lane_ids):
    halted_counts = []
    for lane_id in lane_ids:
        halted_count = 0
        vehicle_ids_in_lane = traci.lane.getLastStepVehicleIDs(lane_id)
        for vehicle_id in vehicle_ids_in_lane:
            speed = traci.vehicle.getSpeed(vehicle_id)
            if speed == 0:
                halted_count += 1
        halted_counts.append(halted_count)
    return halted_counts

def get_traffic_light_vector(tls_id):
    light_state = traci.trafficlight.getRedYellowGreenState(tls_id)
    state_vector = [1 if light == 'G' or light == 'g' else 0 for light in light_state]
    return state_vector

def get_halted_vehicle_count():
    total_halted_vehicles = 0
    lane_ids = traci.lane.getIDList()
    for lane_id in lane_ids:
        stopped_cars = traci.lane.getLastStepHaltingNumber(lane_id)
        total_halted_vehicles -= stopped_cars
    return total_halted_vehicles

def change_traffic_light_phase(tls_id, phase_index):
    traci.trafficlight.setPhase(tls_id, phase_index)

def train(q, q_target, memory, optimizer):
    if memory.size() < 1000:
        return

    s, a, r, s_prime = memory.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float).to(device)
    a = torch.tensor(a, dtype=torch.long).to(device)
    r = torch.tensor(r, dtype=torch.float).to(device)
    s_prime = torch.tensor(s_prime, dtype=torch.float).to(device)

    q_out = q(s)
    q_a = q_out.gather(1, a.unsqueeze(1)).squeeze()

    max_q_prime = q_target(s_prime).max(1)[0]
    target = r + gamma * max_q_prime

    loss = torch.nn.functional.mse_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    sumo_binary = "/usr/bin/sumo"
    sumocfg_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/case_a.sumocfg"
    route_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/route.rou.xml"
    result_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/result2"
    os.makedirs(result_dir, exist_ok=True)

    q = Qnet(18, 2).to(device)
    q_target = Qnet(18, 2).to(device)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer(buffer_limit)
    optimizer = optim.Adam(q.parameters(), lr=0.0001)

    episodes = 30
    episode_cumulative_waiting_times = []
    episode_avg_waiting_times = []  # 추가: 차량 1대당 평균 대기 시간 저장
    
    overwrite_route_file(route_dir, 3)

    for episode in range(episodes):
        
        sumo_cmd = [sumo_binary, "-c", sumocfg_dir, "-r", route_dir, "--no-warnings", "--random"]
        traci.start(sumo_cmd)

        num_steps = 10001
        traffic_time = 0
        phase = 0
        phase_min_time = [30, 3, 10, 3]
        s = np.zeros(18)
        cumulate_waitingTime = 0
        total_vehicles = 0  # 추가: 에피소드 동안의 총 차량 수
        # epsilon = max(0.01, 0.08 - 0.01 * (episode / 200))
        epsilon = max(0.01, 0.1 * np.exp(-episode / 10))  # 더 빠른 감소


        for step in range(num_steps):
            if phase == 0 or phase == 2:
                a = q.sample_action(torch.from_numpy(s).float().to(device), traffic_time, phase_min_time[phase], epsilon)

            if a == 1:
                phase = (phase + 1) % 4
                traffic_time = 0
                change_traffic_light_phase("J6", phase)

            traci.simulationStep()
            traffic_time += 1

            s_prime = np.concatenate([
                get_halted_vehicles_vector(traci.lane.getIDList()),
                get_traffic_light_vector("J6")
            ])
            s_prime = np.append(s_prime, traffic_time)

            r = get_halted_vehicle_count()
            memory.put((s, a, r, s_prime))

            train(q, q_target, memory, optimizer)

            s = s_prime
            cumulate_waitingTime += r * (-1)

            # 총 차량 수 갱신
            total_vehicles += len(traci.simulation.getDepartedIDList())

            if step == 10000:
                episode_cumulative_waiting_times.append(cumulate_waitingTime)
            
            if step % 1000 == 0 and step != 0:
                print(f"\n Episode : {episode} Step: {step}")
                print(f"reward: {r}")
                print(f"Total Vehicles: {total_vehicles}")
                print(f"culmulate_waitingTime: {cumulate_waitingTime}")
            
        # 추가: 차량 1대당 평균 대기 시간 계산 및 저장
        avg_waiting_time = cumulate_waitingTime / total_vehicles if total_vehicles > 0 else 0
        episode_avg_waiting_times.append(avg_waiting_time)

        if episode % 2 == 0:
            q_target.load_state_dict(q.state_dict())
        
    traci.close()

    # 에피소드별 누적 대기 시간 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), episode_cumulative_waiting_times, color='red', marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative waiting time (s)")
    plt.title("Cumulative Waiting Time per Episode")
    plt.grid()
    output_path_cum = os.path.join(result_dir, "cumulative_waiting_time_target_network.png")
    plt.savefig(output_path_cum)
    plt.close()
    print(f"Graph saved at {output_path_cum}")

    # 추가: 차량 1대당 평균 대기 시간 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), episode_avg_waiting_times, color='blue', marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time per vehicle (s)")
    plt.title("Average Waiting Time per Vehicle per Episode")
    plt.grid()
    output_path_avg = os.path.join(result_dir, "average_waiting_time_per_vehicle.png")
    plt.savefig(output_path_avg)
    plt.close()
    print(f"Graph saved at {output_path_avg}")


if __name__ == '__main__':
    main()
