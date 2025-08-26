import ray
import copy
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float32
import time
import pandas as pd

ax_iteration_without_improvment = 100
max_itteration_for_mval = 100
min_itteration_for_mval = 50
cadence = 5
num_cities = 0
tabu = {}

USE_CUDA = True

if USE_CUDA:
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure that CUDA is installed and a compatible GPU is present.")
    cuda.select_device(0)

ray.init(address="ray://192.168.1.81:10001")

def read_csv_to_cost_matrix(file_path):
    global num_cities
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader)
        num_cities = int(first_row[0])
        next(csvreader)
        cost_matrix = np.zeros((num_cities + 1, num_cities + 1))
        for row in csvreader:
            point1_id, point2_id = map(int, row[0].split('-'))
            distance = float(row[5])
            cost_matrix[point1_id, point2_id] = distance
            cost_matrix[point2_id, point1_id] = distance
    return cost_matrix

def genarate_random_solution(base_id: int):
    global num_cities
    city_ids = list(range(1, num_cities + 1))
    if base_id in city_ids:
        city_ids.remove(base_id)
    client_random_ids = random.sample(city_ids, len(city_ids))
    path = [base_id] + client_random_ids + [base_id]
    return path

@cuda.jit
def calculate_cost_kernel(solution, cost_matrix, total_cost):
    shared_cost = cuda.shared.array(shape=16, dtype=float32) #16 32 64 96 128 164 256 
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    local_cost = 0.0

    for i in range(tid, solution.shape[0] - 1, stride):
        start = solution[i]
        end = solution[i + 1]
        local_cost += cost_matrix[start, end]

    shared_cost[tid] = local_cost
    cuda.syncthreads()

    if tid == 0:
        total_cost[0] = 0.0
        for i in range(cuda.blockDim.x):
            total_cost[0] += shared_cost[i]

def calculate_cost_cuda(solution, cost_matrix):
    global global_gpu_transfer_time

    start_transfer_time = time.time()
    solution = np.array(solution, dtype=np.int32)

    if not hasattr(calculate_cost_cuda, 'cost_matrix_device'):
        calculate_cost_cuda.cost_matrix_device = cuda.to_device(cost_matrix)

    if not hasattr(calculate_cost_cuda, 'solution_device'):
        calculate_cost_cuda.solution_device = cuda.device_array_like(solution)

    if not hasattr(calculate_cost_cuda, 'cost_device'):
        calculate_cost_cuda.cost_device = cuda.device_array(1, dtype=np.float32)

    calculate_cost_cuda.solution_device.copy_to_device(solution)

    threadsperblock = 16 #16 32 64 96 128 164 256 
    blockspergrid = (solution.size + threadsperblock - 1) // threadsperblock

    end_transfer_time = time.time()
    global_gpu_transfer_time += (end_transfer_time - start_transfer_time)

    calculate_cost_kernel[blockspergrid, threadsperblock](
        calculate_cost_cuda.solution_device,
        calculate_cost_cuda.cost_matrix_device,
        calculate_cost_cuda.cost_device
    )

    cuda.synchronize()
    cost = calculate_cost_cuda.cost_device.copy_to_host()

    return np.sum(cost)

def calculate_cost(solution, cost_matrix, USE_CUDA=False):
    if USE_CUDA:
        return calculate_cost_cuda(solution, cost_matrix)
    else:
        start_ids = np.array(solution[:-1])
        end_ids = np.array(solution[1:])
        return np.sum(cost_matrix[start_ids, end_ids])

def change_solution(current_soultion: dict):
    moves = []
    candidate_to_solution = copy.deepcopy(current_soultion)

    for _ in range(10000):
        random_node_number_1 = random.randint(1, len(candidate_to_solution)-2)
        random_node_number_2 = random.randint(1, len(candidate_to_solution)-2)
        move = (random_node_number_1,random_node_number_2)
        value_1 = candidate_to_solution[random_node_number_1]
        value_2 = candidate_to_solution[random_node_number_2]

        if move not in tabu:
            value_1 = candidate_to_solution[random_node_number_1]
            value_2 = candidate_to_solution[random_node_number_2]
            candidate_to_solution[random_node_number_1] = value_2
            candidate_to_solution[random_node_number_2] = value_1
            moves.append(move)
            break
    
    return candidate_to_solution, moves

def caculate_new_solution(current_solution, cost_matrix, USE_CUDA=False):
    MVal = {}
    current_cost = calculate_cost(current_solution,cost_matrix)
    itteration_for_mval = random.randint(min_itteration_for_mval,max_itteration_for_mval)
    for _ in range(itteration_for_mval):
        new_solution, moves =  change_solution(current_solution)
        
        new_cost = calculate_cost(new_solution,cost_matrix)
        m_value = current_cost - new_cost
        MVal[m_value] = (new_solution, moves)
        
    max_m_value = max(MVal.keys())
    return MVal[max_m_value]

@ray.remote(num_gpus=1)
def tabu_search_remote(cost_matrix, current_solution, iterations):
    global cadence
    best_solution = copy.deepcopy(current_solution)
    best_cost = calculate_cost(best_solution, cost_matrix)
    cost_history = [best_cost]
    for _ in range(iterations):
        new_solution, moves = caculate_new_solution(current_solution, cost_matrix)
        new_cost = calculate_cost(new_solution, cost_matrix)
        if new_cost < best_cost:
            current_solution = copy.deepcopy(new_solution)
            best_solution = copy.deepcopy(new_solution)
            best_cost = new_cost
        cost_history.append(best_cost)

        for move in moves:
            tabu[move] = cadence

        for move_key, cadence in list(tabu.items()):
            new_cadence = cadence - 1
            if new_cadence == 0:
                del tabu[move_key]
            else:
                tabu[move_key] = new_cadence
    return best_solution, best_cost, cost_history

def tabu_search_local(cost_matrix, current_solution, iterations):
    global cadence
    cost_history = []
    cost_matrix = read_csv_to_cost_matrix(file_path)
    current_solution = genarate_random_solution(start_id)
    best_solution = copy.deepcopy(current_solution)
    best_cost = calculate_cost(best_solution, cost_matrix, USE_CUDA=True)
    cost_history = [best_cost]

    for _ in range(iterations):
        new_solution, moves = caculate_new_solution(current_solution, cost_matrix, USE_CUDA=True)
        new_cost = calculate_cost(new_solution, cost_matrix, USE_CUDA=True)
        if new_cost < best_cost:
            current_solution = copy.deepcopy(new_solution)
            best_solution = copy.deepcopy(new_solution)
            best_cost = new_cost
        cost_history.append(best_cost)

        for move in moves:
            tabu[move] = cadence

        for move_key, cadence in list(tabu.items()):
            new_cadence = cadence - 1
            if new_cadence == 0:
                del tabu[move_key]
            else:
                tabu[move_key] = new_cadence
    return best_solution, best_cost, cost_history


def is_valid_solution(solution, num_cities):
    if solution[0] != solution[-1]:
        return False
    visited_cities = solution[1:-1]
    if len(visited_cities) != len(set(visited_cities)) or len(visited_cities) != num_cities - 1:
        return False
    return True

def merge_solutions(solution1, solution2, cost_matrix):
    merged_solution = list(set(solution1[1:-1] + solution2[1:-1]))
    random.shuffle(merged_solution)
    merged_solution = [solution1[0]] + merged_solution + [solution1[0]]
    assert is_valid_solution(merged_solution, len(cost_matrix) - 1), "Invalid merged solution!"
    return merged_solution

def distributed_tabu_search_combined(file_path, start_id, total_iterations=100):
    cost_matrix = read_csv_to_cost_matrix(file_path)
    initial_solution_local = genarate_random_solution(start_id)
    initial_solution_remote = genarate_random_solution(start_id)

    remote_iterations = total_iterations // 2
    local_iterations = total_iterations - remote_iterations

    remote_result = tabu_search_remote.remote(cost_matrix, initial_solution_remote, remote_iterations)
    best_local_solution, best_local_cost, local_cost_history = tabu_search_local(cost_matrix, initial_solution_local, local_iterations)
    best_remote_solution, best_remote_cost, remote_cost_history = ray.get(remote_result)

    merged_solution = merge_solutions(best_local_solution, best_remote_solution, cost_matrix)
    merged_cost = calculate_cost(merged_solution, cost_matrix)

    combined_cost_history = local_cost_history + remote_cost_history

    return merged_solution, merged_cost, combined_cost_history

def plot_points(file_path, start_id, path=None):
    coordinates = {}
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        next(csvreader)
        
        for row in csvreader:
            point1_id, point2_id = map(int, row[0].split('-'))
            point1x, point1y = float(row[1]), float(row[2])
            point2x, point2y = float(row[3]), float(row[4])
            
            if point1_id not in coordinates:
                coordinates[point1_id] = (point1x, point1y)
            if point2_id not in coordinates:
                coordinates[point2_id] = (point2x, point2y)
    
    plt.figure(figsize=(10, 10))
    
    for point_id, (x, y) in coordinates.items():
        if point_id == start_id:
            plt.scatter(x, y, color='red', label='Start/End')
            plt.text(x, y, f'{point_id}', color='red', fontsize=12, ha='right')
        else:
            plt.scatter(x, y, color='blue')
            plt.text(x, y, f'{point_id}', color='blue', fontsize=12, ha='right')

    if path:
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            if start in coordinates and end in coordinates:
                x1, y1 = coordinates[start]
                x2, y2 = coordinates[end]
                plt.plot([x1, x2], [y1, y2], 'k-', lw=1)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Visualization of Points and Paths')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    results = []

    for _ in range(30):
        file_path = r'C:\Users\jakub\Visual Studio Code sem2\Supercomputer\transformed_data_medium.csv'
        start_id = 1
        total_iterations = 1000
        global_gpu_transfer_time = 0.0

        start_time = time.time()
        best_solution, best_cost, cost_history = distributed_tabu_search_combined(file_path, start_id, total_iterations)
        end_time = time.time()

        #results.append([end_time - start_time, global_gpu_transfer_time, cost_history[-1]])

        #df = pd.DataFrame(results, columns=["Execution Time", "GPU Transfer Time", "Best Cost"])
        #df.to_csv('results16s.csv', sep=';', index=False)
        
        print(f"Execution Time: {end_time - start_time} seconds")
        print(f"GPU Data Transfer and Memory Allocation Time: {global_gpu_transfer_time} seconds")
        print("Best Solution:", best_solution)
        print("Best Cost:", best_cost)
        cost_dictionary = read_csv_to_cost_matrix(file_path)
        if best_solution:
            plot_points(file_path, start_id, best_solution)
            
            plt.figure()
            plt.title('Cost history')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.grid(True)
            plt.plot(cost_history)
            plt.show()
