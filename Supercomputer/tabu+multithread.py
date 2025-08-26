import copy
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

max_itteration_for_mval = 100
min_itteration_for_mval = 50
cadence = 5
num_cities = 0

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

def calculate_cost(solution, cost_matrix):
    start_ids = np.array(solution[:-1])
    end_ids = np.array(solution[1:])
    return np.sum(cost_matrix[start_ids, end_ids])

def change_solution(current_solution, tabu):
    moves = []
    candidate_solution = current_solution[:]
    for _ in range(10000):
        i, j = sorted(random.sample(range(1, len(candidate_solution) - 1), 2))
        move = (i,j)
        if move not in tabu:
            candidate_solution[i:j+1] = reversed(candidate_solution[i:j+1])
            moves.append(move)
            break
    return candidate_solution, moves

def __find_best_solution(tabu_results, current_best_cost):
    current_best_solution = None
    for  solution in tabu_results:
        if solution[1] < current_best_cost:
            current_best_cost = solution[1]
            current_best_solution = solution

    return current_best_solution

import threading
def __search_other_solution(current_solution,cost_matrix,tabu,num_candidates):
        local_best_solution = None
        local_best_cost = float('inf')
        local_moves = None
        s_t = time.perf_counter()

        for _ in range(num_candidates):
            new_solution, moves = change_solution(current_solution, tabu)
            new_cost = calculate_cost(new_solution, cost_matrix)
            if new_cost < local_best_cost:
                local_best_solution = new_solution
                local_best_cost = new_cost
                local_moves = moves
        
        return (local_best_solution, local_best_cost, local_moves)

def tabu_search(file_path, start_id, iteration_number, num_threads: int):
    global cadence
    cost_matrix = read_csv_to_cost_matrix(file_path)
    current_solution = genarate_random_solution(start_id)
    best_solution = copy.deepcopy(current_solution)
    best_cost = calculate_cost(best_solution, cost_matrix)
    cost_history = [best_cost]
    tabu = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for _ in range(iteration_number):
            num_candidates = random.randint(min_itteration_for_mval, max_itteration_for_mval)
            
            futures = [(executor.submit(__search_other_solution,current_solution, cost_matrix, tabu, num_candidates//num_threads)) for _ in range(num_threads)] 
            
            results = [f.result() for f in futures]

            solution_candidate = __find_best_solution(results,best_cost)
            if solution_candidate:
                new_solution, new_cost, moves = solution_candidate
                current_solution = copy.deepcopy(new_solution)
                best_solution = copy.deepcopy(new_solution)
                best_cost = new_cost
            for move in moves:
                tabu[move] = cadence
            for move_key, cadence in list(tabu.items()):
                new_cadence = cadence - 1
                if new_cadence == 0:
                    del tabu[move_key]
                else:
                    tabu[move_key] = new_cadence
            cost_history.append(best_cost)
    return best_solution, best_cost, cost_history

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
        file_path = r'C:\Users\jakub\Visual Studio Code sem2\Supercomputer\transformed_data_small.csv'
        start_id = 1
        total_iterations = 1000
        num_threads = 1 # 1 2 4 6 8 10 12 14 16 18 20

        start_time = time.time()
        best_solution, best_cost, cost_history = tabu_search(file_path, start_id, total_iterations, num_threads)
        end_time = time.time()

        results.append([end_time - start_time, best_cost])

        df = pd.DataFrame(results, columns=["Execution Time", "Best Cost"])
        df.to_csv('resultsmulti1s.csv', sep=';', index=False)

        #print(f"Execution Time: {end_time - start_time} seconds")
        #print("Best Solution:", best_solution)
        #print("Best Cost:", best_cost)
        #cost_dictionary = read_csv_to_cost_matrix(file_path)
        #if best_solution:
        #    plot_points(file_path, start_id, best_solution)
        #    
        #    plt.figure()
        #    plt.title('Cost history')
        #    plt.xlabel('Iteration')
        #    plt.ylabel('Cost')
        #    plt.grid(True)
        #    plt.plot(cost_history)
        #    plt.show()
