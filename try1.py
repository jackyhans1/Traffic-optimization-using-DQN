import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import traci
import numpy as np
import collections
import xml.etree.ElementTree as ET
import random
from xml.dom import minidom
import torch.nn.functional as F
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

buffer_limit = 100000
batch_size = 128
gamma = 0.99

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        # 입력층부터 은닉층 5층 구성
        self.fc1 = nn.Linear(state_dim, 400)  # 입력 -> 첫 번째 은닉층
        self.fc2 = nn.Linear(400, 400)       # 두 번째 은닉층
        self.fc3 = nn.Linear(400, 400)       # 세 번째 은닉층
        self.fc4 = nn.Linear(400, 400)       # 네 번째 은닉층
        self.fc5 = nn.Linear(400, 400)       # 다섯 번째 은닉층
        self.output = nn.Linear(400, action_dim)  # 출력층 (행동 차원)

    def forward(self, x):
        # 은닉층 활성화 함수: ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # 출력층 활성화 함수: Sigmoid
        x = torch.sigmoid(self.output(x))
        return x

    def sample_action(self, obs, traffic_time, phase_min_time, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 1)  # 랜덤 행동 선택
        else:
            out = self.forward(obs).argmax().item()  # 최대 Q-값을 갖는 행동 선택
            if out == 1 and traffic_time < phase_min_time:
                out = 0
            return out



def overwrite_route_file(file_path, num_repeats):
    """
    기존 route XML 파일에 새로운 flow 항목을 덮어쓰는 함수.

    :param file_path: route 파일의 경로
    :param num_repeats: flow 항목을 반복할 횟수
    :param flow_number: 각 flow 항목의 차량 수 (기본값 10)
    :param time_interval: 각 flow 항목의 시간 간격 (기본값 3600초)
    """
    with open(file_path, "w", encoding="UTF-8") as file:
        file.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Vehicles, persons and containers (sorted by depart) -->
""")
        begin = 0
        step = 3600
        # flow 항목 반복하여 생성
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

        # XML 종료 태그
        file.write("</routes>\n")

def get_halted_vehicles_vector(lane_ids):
    """
    주어진 차선 ID 리스트에 대해 현재 스텝에서 정지한 차량 수를 벡터 형태로 반환합니다.

    :param lane_ids: 차선 ID 리스트
    :return: 각 차선에 정지한 차량 수를 나타내는 리스트
    """
    halted_counts = []
    for lane_id in lane_ids:
        halted_count = 0
        vehicle_ids_in_lane = traci.lane.getLastStepVehicleIDs(lane_id)  # 차선에 있는 차량 ID 리스트 가져오기

        for vehicle_id in vehicle_ids_in_lane:
            speed = traci.vehicle.getSpeed(vehicle_id)  # 차량의 속도를 확인
            if speed == 0:  # 차량이 정지한 경우
                halted_count += 1

        halted_counts.append(halted_count)

    return halted_counts

def get_traffic_light_vector(tls_id):
    """
    특정 신호등의 현재 상태를 벡터로 반환합니다.

    :param tls_id: 신호등 ID
    :param controlled_links: 해당 신호등이 제어하는 링크 리스트
    :return: 각 신호 상태를 나타내는 리스트 (초록: 1, 빨강: 0)
    """
    # 현재 신호 상태 문자열 가져오기 (예: "rGrG" 등)
    light_state = traci.trafficlight.getRedYellowGreenState(tls_id)

    # 신호 상태를 벡터로 변환 (초록: 1, 빨강/노랑: 0)
    state_vector = [1 if light == 'G' or light == 'g' else 0 for light in light_state]

    return state_vector

def get_halted_vehicle_count():
    """
    모든 차선에서 현재 스텝에서 정지한 차량의 총 수를 계산하는 함수.
    Returns:
        int: 한 스텝에서 모든 차선에서 정지한 차량의 수.
    """
    total_halted_vehicles = 0
    lane_ids = traci.lane.getIDList()  # 모든 차선 ID 가져오기

    for lane_id in lane_ids:
        # 특정 차선에서 정지한 차량 수 얻기 (기본 정지 기준: 속도 <= 0.1 m/s)
        stopped_cars = traci.lane.getLastStepHaltingNumber(lane_id)
        total_halted_vehicles -= stopped_cars  # 보상에 추가 (음수 값)

    return total_halted_vehicles
    # vehicle_ids = traci.vehicle.getIDList()  # 모든 차량 ID 가져오기

    # for vehicle_id in vehicle_ids:
    #     speed = traci.vehicle.getSpeed(vehicle_id)  # 차량 속도 확인
    #     if speed == 0:  # 차량이 정지한 경우
    #         total_halted_vehicles += 1

    # return total_halted_vehicles

def change_traffic_light_phase(tls_id, phase_index):
    """
    특정 신호등의 phase를 변경합니다.

    :param tls_id: 신호등 ID
    :param phase_index: 설정할 phase의 인덱스
    """
    traci.trafficlight.setPhase(tls_id, phase_index)


# def train(q, q_target, memory, optimizer):
#     s, a, r, s_prime = memory
#     q_out = q(s)  # (batch_size, action_dim)

#     # 다음 상태에서의 최대 Q-값을 계산
#     q_prime_out = q_target(s_prime)  # q_target(s_prime)의 출력 (batch_size, action_dim)
#     if q_prime_out.dim() == 1:  # 1D 텐서라면 차원 추가
#         q_prime_out = q_prime_out.unsqueeze(0)  # 차원을 맞추기 위해 unsqueeze(0)

#     max_q_prime = q_prime_out.max(1)[0].unsqueeze(1)  # 최대 Q-값 추출

#     # 타겟 Q-값 계산 (벨만 방정식)
#     target = r + 0.99 * max_q_prime

#     # 행동 인덱스를 정수형 텐서로 변환
#     target = torch.tensor(a, dtype=torch.long)  # a는 행동 인덱스이며, 정수형 텐서로 변환해야 함

#     # CrossEntropyLoss는 예측값(softmax 확률)을 기대합니다.
#     # q_out은 이미 Q-값으로, CrossEntropyLoss는 softmax를 내부적으로 사용하므로
#     # 추가적인 softmax 처리 없이 그대로 입력할 수 있습니다.

#     # 손실 계산 (CrossEntropy)
#     loss_fn = nn.CrossEntropyLoss()
#     loss = loss_fn(q_out, target)  # q_out: (batch_size, action_dim), target: (batch_size,)

#     # 역전파 및 최적화
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime
        loss = F.cross_entropy(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for trainsition in mini_batch:
            s, a, r, s_prime = trainsition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
        
        # print("s_lst:", s_lst)  # s_lst의 형태 확인
        # print("a_lst:", a_lst)  # a_lst의 형태 확인
        # print("r_lst:", r_lst)  # r_lst의 형태 확인
        # print("s_prime_lst:", s_prime_lst)  # s_prime_lst의 형태 확인

        # tensor_s = torch.tensor(s_lst, dtype=torch.float).to(device)
        # tensor_a = torch.tensor(a_lst).to(device)
        # tensor_r = torch.tensor(r_lst).to(device)
        # tensor_s_prime = torch.tensor(s_prime_lst, dtype=torch.float).to(device)

        # 리스트를 numpy 배열로 변환한 후 텐서로 변환 -> 속도 개선
        tensor_s = torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
        tensor_a = torch.tensor(np.array(a_lst), dtype=torch.long).to(device)
        tensor_r = torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
        tensor_s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
        
        return tensor_s, tensor_a, tensor_r, tensor_s_prime

    def size(self):
        return len(self.buffer)

def main():
  sumo_binary = "/usr/bin/sumo"
  sumocfg_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/case_a.sumocfg"
  route_dir = "/workspace/UndergraduateResearchAssistant/GraduateProject/sumo/route.rou.xml"

  q = Qnet(18,2).to(device) # 8 + 7 + 1
  q_target = Qnet(18,2).to(device)
  q_target.load_state_dict(q.state_dict())

  optimizer = optim.Adam(q.parameters(), lr=0.00025)
  memory = ReplayBuffer()

  


  for episode in range(100):
     #차량 수요 랜덤으로 설정
    # overwrite_route_file(route_dir, 3)
    sumo_cmd = [sumo_binary, "-c", sumocfg_dir, "-r", route_dir, "--no-warnings", "--random"]

    traci.start(sumo_cmd)


    num_steps = 10000


    #dt 측정하기 위함
    current_phase = 0
    traffic_time = 0

    #신호 phase
    phase = 0

    #신호 phase 당 최소 시간
    phase_min_time = [30, 3, 10, 3]

    #관찰값 초기화
    s = [0] * 18
    s = np.array(s)
    # s = torch.from_numpy(s).float()

    #누적 대기시간
    cumulate_waitingTime = 0

    epsilon = max(0.01, 0.08 - 0.01 * (episode/200))

    for step in range(num_steps):
        s_tensor = torch.from_numpy(s).float().to(device)
        if phase == 0 or phase == 2: # 노란불은 고정으로 3초로 해야하니깐
            a = q.sample_action(s_tensor, traffic_time, phase_min_time[phase], epsilon)
            # print(f"a : {a}")

        #신호 바꾸기
        if a == 1 :
          phase = (phase + 1) % 4
          traffic_time = 0
          change_traffic_light_phase("J6", phase)

        traci.simulationStep()
        traffic_time += 1

        s_prime = np.array([])
        s_prime = np.concatenate([
          get_halted_vehicles_vector(traci.lane.getIDList()),
          get_traffic_light_vector("J6")
        ])
        s_prime = np.append(s_prime, traffic_time)
        # s_prime = torch.from_numpy(s_prime).float()
        # print(f"s_prime: {s_prime}")
        # s_prime = np.array(s_prime)
        # if current_phase == traci.trafficlight.getPhase("J6"):
        #   start_time += 1
        # else:
        #   start_time = 0
        # phase_info = traci.trafficlight.getPhase("J6")
        # print(f"traffic_phase : {phase_info}")

        

        r = get_halted_vehicle_count()
        memory.put((s,a,r,s_prime))
        s = s_prime

        cumulate_waitingTime += r * (-1)
        # print(f"cumulate_waitingTime{cumulate_waitingTime}")

        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)
        # # q_target.load_state_dict(q.state_dict())

        # if memory.size() > 1000:
        #     train(q, q_target, memory, optimizer)

        if step % 1000 == 0 and step != 0:
            q_target.load_state_dict(q.state_dict())
            print(f"\nEpisode : {episode+1} Step: {step}")
            print(f"reward: {r}")
            print(f"culmulate_waitingTime: {cumulate_waitingTime}")

    
            
    traci.close()

if __name__ == '__main__':
    main()

# traci.close()



