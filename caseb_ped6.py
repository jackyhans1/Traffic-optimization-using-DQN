import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
batch_size = 32
gamma = 0.95
initial_lr = 0.001

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.output = nn.Linear(400, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)

    def sample_action(self, obs, traffic_time, phase_min_time, epsilon):
        if traffic_time >= phase_min_time:
            if random.random() < epsilon:
                return random.randint(0, 1)
            else:
                out = self.forward(obs).argmax().item()
                return out
        else:
            return 0

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = zip(*mini_batch)
        return (np.array(s_lst), np.array(a_lst), np.array(r_lst), 
                np.array(s_prime_lst), np.array(done_lst))

    def size(self):
        return len(self.buffer)

def overwrite_route_file(file_path, num_repeats):
    with open(file_path, "w", encoding="UTF-8") as file:
        file.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
""")
        begin = 0
        step = 600
        for i in range(num_repeats):
            end = (i + 1) * step
            flow_number_1 = random.randint(100, 300)
            flow_number_2 = random.randint(100, 300)
            flow_number_3 = random.randint(100, 300)
            flow_number_4 = random.randint(100, 300)
            flow_number_5 = random.randint(100, 300)
            flow_number_6 = random.randint(100, 300)
            flow_number_7 = random.randint(100, 300)
            flow_number_8 = random.randint(100, 300)
            
            pedestrian_count = random.randint(1, 10)
            for j in range(pedestrian_count):
                pedestrian = f"""    <person id="p_{i*200+j+1}" depart="{begin}">
        <personTrip from="E{random.randint(0, 3)}" to="-E{random.randint(0, 3)}"/>
    </person>"""
                file.write(pedestrian + "\n")
            
            flow = f"""    <flow id="F{i*8+1}" begin="{begin}" departLane="2" departSpeed="1" fromTaz="taz_1" toTaz="taz_0" end="{end}" number="{flow_number_1}"/>
    <flow id="F{i*8+2}" begin="{begin}" departLane="2" fromTaz="taz_3" departSpeed="1" toTaz="taz_2" end="{end}" number="{flow_number_2}"/>
    <flow id="F{i*8+3}" begin="{begin}" departLane="1" fromTaz="taz_1" departSpeed="1" toTaz="taz_3" end="{end}" number="{flow_number_3}"/>
    <flow id="F{i*8+4}" begin="{begin}" departLane="1" fromTaz="taz_3" departSpeed="1" toTaz="taz_1" end="{end}" number="{flow_number_4}"/>
    <flow id="F{i*8+5}" begin="{begin}" departLane="2" departSpeed="1" fromTaz="taz_2" toTaz="taz_1" end="{end}" number="{flow_number_5}"/>
    <flow id="F{i*8+6}" begin="{begin}" departLane="2" fromTaz="taz_0" departSpeed="1" toTaz="taz_3" end="{end}" number="{flow_number_6}"/>
    <flow id="F{i*8+7}" begin="{begin}" departLane="1" fromTaz="taz_0" departSpeed="1" toTaz="taz_2" end="{end}" number="{flow_number_7}"/>
    <flow id="F{i*8+8}" begin="{begin}" departLane="1" fromTaz="taz_2" departSpeed="1" toTaz="taz_0" end="{end}" number="{flow_number_8}"/>"""
            
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

def get_stopped_pedestrians_vector(edge_ids):
    stopped_counts = []
    for edge_id in edge_ids:
        stopped_count = 0
        pedestrian_ids_in_edge = traci.edge.getLastStepPersonIDs(edge_id)
        for pedestrian_id in pedestrian_ids_in_edge:
            if traci.person.getSpeed(pedestrian_id) == 0:
                stopped_count += 1
        stopped_counts.append(stopped_count)
    return stopped_counts

def get_traffic_light_vector(tls_id):
    light_state = traci.trafficlight.getRedYellowGreenState(tls_id)
    state_vector = [1 if light in ['G','g'] else 0 for light in light_state]
    return state_vector

def get_pedestrian_waiting_time(ped_id):
    return 1 if traci.person.getSpeed(ped_id) == 0 else 0

def calculate_vehicle_avg_waiting_time():
    total_waiting_time = sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList())
    num_vehicles = len(traci.vehicle.getIDList())
    return total_waiting_time / num_vehicles if num_vehicles > 0 else 0

def calculate_pedestrian_avg_waiting_time():
    total_waiting_time = sum(get_pedestrian_waiting_time(ped_id) for ped_id in traci.person.getIDList())
    num_persons = len(traci.person.getIDList())
    return total_waiting_time / num_persons if num_persons > 0 else 0

def change_traffic_light_phase(tls_id, phase_index):
    traci.trafficlight.setPhase(tls_id, phase_index)

def train(q, q_target, memory, optimizer):
    if memory.size() < 1000:
        return

    s, a, r, s_prime, done = memory.sample(batch_size)

    s = torch.tensor(s, dtype=torch.float).to(device)
    a = torch.tensor(a, dtype=torch.long).to(device)
    r = torch.tensor(r, dtype=torch.float).to(device)
    s_prime = torch.tensor(s_prime, dtype=torch.float).to(device)
    done_mask = torch.tensor(done, dtype=torch.float).to(device)

    q_out = q(s)
    q_a = q_out.gather(1, a.unsqueeze(1)).squeeze()

    max_q_prime = q_target(s_prime).max(1)[0]
    target = r + gamma * (1 - done_mask) * max_q_prime

    loss = torch.nn.functional.mse_loss(q_a, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    sumo_binary = "/usr/bin/sumo"
    sumocfg_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/case_b.sumocfg"
    route_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/case_b_new.rou.xml"
    result_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/4ways_ped_result6"
    os.makedirs(result_dir, exist_ok=True)

    q = Qnet(97, 2).to(device)
    q_target = Qnet(97, 2).to(device)
    q_target.load_state_dict(q.state_dict())

    memory = ReplayBuffer(buffer_limit)
    optimizer = optim.Adam(q.parameters(), lr=initial_lr)

    episodes = 30
    episode_cumulative_waiting_times = []
    episode_avg_waiting_times = []
    episode_total_entities_list = []  # 에피소드별 총 엔티티(차량+보행자) 수 저장

    overwrite_route_file(route_dir, 17)

    for episode in range(episodes):
        sumo_cmd = [sumo_binary, "-c", sumocfg_dir, "-r", route_dir, "--no-warnings", "--random"]
        traci.start(sumo_cmd)

        num_steps = 10001
        traffic_time = 0
        phase = 0
        phase_time_for_first_episode = [50, 5, 50, 5, 50, 5, 50, 5]
        phase_min_time = [20, 5, 30, 5, 20, 5, 30, 5]
        s = np.zeros(97)

        sum_total_waiting_times = 0.0
        sum_entities = 0
        cumulative_loss = 0.0

        epsilon = max(0.01, 0.04 - 0.01 * (episode / 10))

        for step in range(num_steps):
            done = (step == num_steps - 1)

            if episode == 0:
                # 첫 에피소드: 고정 신호
                if traffic_time >= phase_time_for_first_episode[phase]:
                    a = 1
                else:
                    a = 0
            else:
                # 이후 에피소드: RL 액션
                a = q.sample_action(torch.from_numpy(s).float().to(device), traffic_time, phase_min_time[phase], epsilon)

            if a == 1:
                phase = (phase + 1) % 8
                traffic_time = 0
                change_traffic_light_phase("J1", phase)

            traci.simulationStep()
            traffic_time += 1

            s_prime = np.concatenate([
                get_halted_vehicles_vector(traci.lane.getIDList()),
                get_traffic_light_vector("J1"),
                get_stopped_pedestrians_vector(traci.edge.getIDList())
            ])
            s_prime = np.append(s_prime, traffic_time)

            avg_v_waiting_time = calculate_vehicle_avg_waiting_time()
            avg_p_waiting_time = calculate_pedestrian_avg_waiting_time()

            # 보상 계산
            r = -(avg_v_waiting_time + (avg_p_waiting_time)*5) / 100.0

            num_vehicles = len(traci.vehicle.getIDList())
            num_persons = len(traci.person.getIDList())

            total_v_waiting = avg_v_waiting_time * num_vehicles
            total_p_waiting = avg_p_waiting_time * num_persons

            sum_total_waiting_times += (total_v_waiting + total_p_waiting)
            sum_entities += (num_vehicles + num_persons)

            if episode > 0:
                memory.put((s, a, r, s_prime, done))
                step_loss = train(q, q_target, memory, optimizer)
                if step_loss is not None:
                    cumulative_loss += step_loss

            s = s_prime

            if step % 1000 == 0 and step != 0:
                current_cumulative_waiting_time = sum_total_waiting_times
                current_average_waiting_time = (sum_total_waiting_times / sum_entities) if sum_entities > 0 else 0.0
                print(f"\nEpisode: {episode}, Step: {step}")
                print(f"Total Entities: {sum_entities}")
                print(f"Cumulative Waiting Time: {current_cumulative_waiting_time}")
                print(f"Average Waiting Time per Entity: {current_average_waiting_time:.4f}")
                print(f"Loss: {cumulative_loss:.4f}")
                cumulative_loss = 0.0

            if done:
                break

        episode_cumulative_waiting_time = sum_total_waiting_times
        episode_avg_waiting_time = (sum_total_waiting_times / sum_entities) if sum_entities > 0 else 0.0
        episode_cumulative_waiting_times.append(episode_cumulative_waiting_time)
        episode_avg_waiting_times.append(episode_avg_waiting_time)

        # 에피소드 종료 후 현재 에피소드의 총 엔티티 수를 저장
        episode_total_entities_list.append(sum_entities)

        if episode % 2 == 0:
            q_target.load_state_dict(q.state_dict())
        
        traci.close()

    # 그래프 저장
    # 누적 대기시간 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), episode_cumulative_waiting_times, color='red', marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Cumulative waiting time (s)")
    plt.title("Cumulative Waiting Time per Episode (Vehicles+Pedestrians)")
    plt.grid()
    output_path_cum = os.path.join(result_dir, "cumulative_waiting_time_target_network.png")
    plt.savefig(output_path_cum)
    plt.close()
    print(f"Graph saved at {output_path_cum}")

    # 평균 대기시간 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), episode_avg_waiting_times, color='blue', marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Average waiting time per entity (s)")
    plt.title("Average Waiting Time per Entity per Episode (Vehicles+Pedestrians)")
    plt.grid()
    output_path_avg = os.path.join(result_dir, "average_waiting_time_per_vehicle.png")
    plt.savefig(output_path_avg)
    plt.close()
    print(f"Graph saved at {output_path_avg}")

    # 에피소드별 총 엔티티(차량+보행자) 수 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), episode_total_entities_list, color='green', marker='o', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Total Entities (Vehicles+Pedestrians)")
    plt.title("Total Entities (Vehicles+Pedestrians) per Episode")
    plt.grid()
    output_path_entities = os.path.join(result_dir, "total_entities_per_episode.png")
    plt.savefig(output_path_entities)
    plt.close()
    print(f"Graph saved at {output_path_entities}")

if __name__ == '__main__':
    main()
