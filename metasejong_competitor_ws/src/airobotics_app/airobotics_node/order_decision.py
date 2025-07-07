#!/usr/bin/env python3
# order_decision.py

from typing import List, Dict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os, random, numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from tf2_msgs.msg import TFMessage

from astar import astar, world_to_grid

# A* 전역 변수
_dist_grid_coords = None
_dist_resolution  = None

def _init_dist_worker(args):
    """
    ProcessPoolExecutor initializer.
    grid_coords: List[ (row, col) ] 인덱스 쌍 리스트,
    resolution: map.info.resolution
    """
    global _dist_grid_coords, _dist_resolution
    _dist_grid_coords, _dist_resolution = args

def _compute_dist(pair):
    """
    worker_fn for one (i,j) 쌍.
    """
    i, j = pair
    si, sj = _dist_grid_coords[i]
    gi, gj = _dist_grid_coords[j]
    path = astar(_grid, (si, sj), (gi, gj))
    if path:
        return (i, j, len(path) * _dist_resolution)
    else:
        return (i, j, float('inf'))


# 유전 알고리즘 파라미터
population_size = 300
generations = 100
mutation_rate = 0.1
alpha = 5.0  # 거리 중요도
beta = 2.0   # 각도 변화 중요도
START = 0
END = None

object_coords = None
distance_matrix = None
num_objects = None
middle = None

def create_initial_population():
    population = []
    for _ in range(population_size):
        chromosome = middle.copy()
        random.shuffle(chromosome)
        if END is not None:
            population.append([START] + chromosome + [END])
        else:
            population.append([START] + chromosome)
    return population

def angle_between(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.arccos(dot_product)  # radians

def fitness(chromosome):
    dist = sum(distance_matrix[chromosome[i], chromosome[i+1]] for i in range(len(chromosome)-1))

    angle_change = 0.0
    for i in range(len(chromosome)-2):
        p_prev, p_cur, p_next = object_coords[chromosome[i:i+3]]
        v1, v2 = p_cur - p_prev, p_next - p_cur
        angle_change += np.arccos(
            np.clip(np.dot(v1, v2) /
                    (np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0))

    return alpha * dist + beta * angle_change

def crossover(parent1, parent2):
    max_pick = num_objects if END is None else num_objects - 1
    start, end = sorted(random.sample(range(1, max_pick), 2))
    child = [None]*num_objects
    child[0] = START
    if END is not None: child[END] = END
    child[start:end+1] = parent1[start:end+1]

    fill_positions = [item for item in parent2 if item not in child]
    fill_idx = 0
    for i in range(1, num_objects):
        if END is not None and i == END:
            continue

        if child[i] is None:
            child[i] = fill_positions[fill_idx]
            fill_idx += 1
    return child

def mutate(chromosome):
    if random.random() < mutation_rate:
        picks = [i for i in range(1, num_objects) if i != END]
        idx1, idx2 = random.sample(picks, 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def genetic_algorithm(worker_num, min_delta = 1e-6, patience = 20):
    population = create_initial_population()
    best_chromosome = None
    best_fitness = float('inf')

    wait_count = 0

    for generation in range(generations):
        population.sort(key=fitness)

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

def init_worker(coords, dmat, end_idx):
    global object_coords, distance_matrix, num_objects, middle, END
    object_coords  = coords
    distance_matrix = dmat
    num_objects    = coords.shape[0]
    END = end_idx
    if END is not None:
        middle = [c for c in range(num_objects) if c not in (START, END)]
    else:
        middle = [c for c in range(num_objects) if c != START]



def visit_order(data: List[Dict[str, List[float]]], entry_pos: List[float], exit_pos: None|List[float] = None) -> List[Dict[str, List[float]]]:
    print("visit_order 호출")
    print(f"data: {data}")
    print(f"entry: {entry_pos}")
    print(f"exit: {exit_pos}")

    items = [value
             for d in data
             for value in d.values()]
    
    if exit_pos is not None:
        coords = np.vstack([entry_pos, np.array(items), exit_pos])
        END = coords.shape[0] - 1
    else:
        coords = np.vstack([entry_pos, np.array(items)])
        END = None
    
    n_obj  = coords.shape[0]

    # A*로 채운 거리 행렬
    print("A* 거리 행렬 계산 시작")
    grid_coords = [
        world_to_grid(coords[k], _map_msg, _tf)
        for k in range(n_obj)
    ]
    resolution = _map_msg.info.resolution

    tasks = [
        (i, j)
        for i in range(n_obj)
        for j in range(n_obj)
        if i != j
    ]

    dmat = np.zeros((n_obj, n_obj), dtype=float)
    RESTARTS = (os.cpu_count() or 8) // 2
    with ProcessPoolExecutor(
        max_workers=RESTARTS,
        initializer=_init_dist_worker,
        initargs=((grid_coords, resolution),)
    ) as pool:
        for i, j, dist in pool.map(_compute_dist, tasks):
            dmat[i, j] = dist

    # 병렬 GA
    print("병렬 GA 시작")
    RESTARTS = (os.cpu_count() or 8) // 2
    print(f"방문 노드 개수: {n_obj}")
    print(f"병렬 스레드 개수: {RESTARTS}")
    with ProcessPoolExecutor(max_workers=RESTARTS,
                             initializer=init_worker,
                             initargs=(coords, dmat, END)) as pool:
        bests = list(pool.map(genetic_algorithm, range(RESTARTS)))

    
    order, fit = min(bests, key=lambda x: x[1])
    print(f"\n최적 오브젝트 방문 순서: {order}")
    print(f"최종 Fitness 값 (거리+각도): {fit:.3f}")

    result = [
        data[i-1]
        for i in order
        if i != START and not (exit_pos is not None and i == END)
    ]
    return result

class _MapTfListener(Node):
    """
    /metasejong2025/map 과 /tf_static 로부터
    한 번만 map, tf 를 받아옴
    """
    def __init__(self):
        super().__init__('map_tf_listener')
        qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
        )
        self.map_msg = None
        self.tf_msg  = None

        self.create_subscription(
            OccupancyGrid,
            '/metasejong2025/map',
            self._map_cb,
            qos
        )
        self.create_subscription(
            TFMessage,
            '/tf_static',
            self._tf_cb,
            qos
        )

    def _map_cb(self, msg: OccupancyGrid):
        if self.map_msg is None:
            # 1D data → 2D grid (0=free, 1=occupied)
            w, h = msg.info.width, msg.info.height
            arr  = np.array(msg.data, dtype=int).reshape((h, w))
            self.grid   = np.where(arr == 0, 0, 1)
            self.map_msg = msg

    def _tf_cb(self, msg: TFMessage):
        if self.tf_msg is None:
            for t in msg.transforms:
                if (t.header.frame_id == 'metasejong2025/map'
                        and t.child_frame_id == 'odom'):
                    self.tf_msg = t.transform
                    break


def _get_map_and_tf(timeout: float = 10.0):
    """
    rclpy.init() 부터 한 번만 map+TF를 수신하고 반환
    """
    rclpy.init()
    listener = _MapTfListener()
    import time
    start = time.time()
    while ((listener.map_msg is None or listener.tf_msg is None)
           and time.time() - start < timeout):
        rclpy.spin_once(listener, timeout_sec=0.1)

    if listener.map_msg is None:
        raise RuntimeError("타임아웃: 맵 데이터를 못 받았습니다.")
    if listener.tf_msg is None:
        raise RuntimeError("타임아웃: static TF를 못 받았습니다.")

    grid   = listener.grid
    map_msg = listener.map_msg
    tf      = listener.tf_msg

    rclpy.shutdown()
    return grid, map_msg, tf


# 전역으로 한 번만 맵+TF 로드
_grid, _map_msg, _tf = _get_map_and_tf()