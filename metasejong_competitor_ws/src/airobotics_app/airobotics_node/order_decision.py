from typing import List, Dict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os, random, numpy as np


population_size = 1000
generations = 100
mutation_rate = 0.1
alpha = 5.0  # 거리 중요도
beta = 2.0   # 각도 변화 중요도
START=0


def create_initial_population():
    population = []
    for _ in range(population_size):
        chromosome = middle.copy()
        random.shuffle(chromosome)
        population.append([START] + chromosome)
    return population

def angle_between(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.arccos(dot_product)  # radians

def fitness(chromosome):
    dist = sum(distance_matrix[chromosome[i], chromosome[i+1]] for i in range(num_objects-1))

    angle_change = 0.0
    for i in range(num_objects-2):
        p_prev, p_cur, p_next = object_coords[chromosome[i:i+3]]
        v1, v2 = p_cur - p_prev, p_next - p_cur
        angle_change += np.arccos(
            np.clip(np.dot(v1, v2) /
                    (np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0))

    return alpha * dist + beta * angle_change

def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(1, num_objects), 2))
    child = [None]*num_objects
    child[0] = START
    child[start:end+1] = parent1[start:end+1]

    fill_positions = [item for item in parent2 if item not in child]
    fill_idx = 0
    for i in range(1, num_objects):
        if child[i] is None:
            child[i] = fill_positions[fill_idx]
            fill_idx += 1
    return child

def mutate(chromosome):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(1, num_objects), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def genetic_algorithm(worker_num, min_delta = 1e-6, patience = 20):
    population = create_initial_population()
    best_chromosome = None
    best_fitness = float('inf')

    wait_count = 0

    for generation in range(generations):
        population.sort(key=lambda chromo: fitness(chromo))

        current_best = population[0]
        current_fitness = fitness(current_best)

        if current_fitness + min_delta < best_fitness:
            best_chromosome = current_best
            best_fitness = current_fitness
            wait_count = 0
        else:
            wait_count += 1
        
        if wait_count >= patience:
            print(f"Worker #{worker_num}: [Early-Stop] {patience}세대 연속 개선 없음, 종료합니다.")
            break

        new_population = population[:2]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:10], 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

        # if generation % 10 == 0:
            # print(f"Generation {generation}: Best Fitness = {best_fitness:.3f}")

    print(f"\nWorker #{worker_num}: 최적 오브젝트 방문 순서: {best_chromosome}")
    print(f"Worker #{worker_num}: 최종 Fitness 값 (거리+각도): {best_fitness:.3f}")

    if best_chromosome == None:
        raise Exception("염색체 선정 없음")
    return best_chromosome, best_fitness

# order, fit = genetic_algorithm()


def init_worker(coords, dmat):
    global object_coords, distance_matrix, num_objects, middle
    object_coords  = coords
    distance_matrix = dmat
    num_objects    = coords.shape[0]
    middle          = [c for c in range(num_objects) if c != START]



def visit_order(data: list[dict[str, list[float]]], robot_pos) -> list[dict[str, list[float]]]:
    items = [value
             for d in data
             for value in d.values()]

    object_coords = np.concatenate([robot_pos, items])
    print(object_coords)
    num_objects = np.size(object_coords, axis=0)

    objects = list(range(num_objects))
    middle = [c for c in objects if c != START]

    diff = object_coords[:, None, :] - object_coords[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=2))

    RESTARTS = os.cpu_count()
    if RESTARTS == None:
        RESTARTS = 8
    else:
        RESTARTS //= 2
    
    print(f"방문 노드 개수: {num_objects}")
    print(f"병렬 스레드 개수: {RESTARTS}")
    with ProcessPoolExecutor(max_workers=RESTARTS,
                             initializer=init_worker,
                             initargs=(object_coords, distance_matrix)) as pool:
        bests = list(pool.map(genetic_algorithm, range(RESTARTS)))

    order, fit = min(bests, key=lambda x: x[1])
    print(f"\n최적 오브젝트 방문 순서: {order}")
    print(f"최종 Fitness 값 (거리+각도): {fit:.3f}")


    result = [data[i-1] for i in order]
    return result