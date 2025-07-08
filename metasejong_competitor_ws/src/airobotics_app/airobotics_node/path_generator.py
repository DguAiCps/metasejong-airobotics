# === path_generator.py ===

import numpy as np
import random

# === 유틸 ===
def get_heading(from_point, to_point):
    vec = to_point - from_point
    return np.arctan2(vec[1], vec[0])

def generate_arc_path(start, goal, steps=20, force_final_heading=None):
    path = []
    for i in range(1, steps + 1):
        t = i / steps
        intermediate = (1 - t) * start + t * goal
        heading = get_heading(start, goal) if force_final_heading is None or i < steps else force_final_heading
        path.append((intermediate[0], intermediate[1], heading))
    return path

def is_path_colliding(p1, p2, obstacle, buffer=0.6):
    p1, p2 = np.array(p1), np.array(p2)
    center = obstacle['center']
    radius = obstacle['radius'] + buffer
    d = p2 - p1
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return False
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def create_waypoint(p1, p2, obstacle, buffer=0.6):
    center = obstacle['center']
    radius = obstacle['radius'] + buffer
    path_vec = p2 - p1
    path_dir = path_vec / (np.linalg.norm(path_vec) + 1e-6)
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    wp1 = center + perp_dir * radius
    wp2 = center - perp_dir * radius
    dist1 = np.linalg.norm(p1 - wp1) + np.linalg.norm(wp1 - p2)
    dist2 = np.linalg.norm(p1 - wp2) + np.linalg.norm(wp2 - p2)
    return wp1 if dist1 <= dist2 else wp2

# === 경로 생성 ===
def generate_final_path_with_frontal_pickup(trash_positions, obstacles, start_position):
    final_path = []
    pickup_infos = []
    cur_pos = np.array(start_position)
    used_obstacles = obstacles.copy()

    for t_i in trash_positions:
        direction = t_i - cur_pos
        direction /= (np.linalg.norm(direction) + 1e-6)
        pickup_point = t_i - direction * 0.9
        goal_heading = get_heading(pickup_point, t_i)
        arc = generate_arc_path(cur_pos, pickup_point, steps=20, force_final_heading=goal_heading)

        collision = False
        for j in range(len(arc) - 1):
            for obs in used_obstacles:
                if is_path_colliding(arc[j][:2], arc[j+1][:2], obs):
                    collision = True
                    collision_obs = obs
                    break
            if collision:
                break

        if not collision:
            final_path.extend(arc)
        else:
            wp = create_waypoint(cur_pos, pickup_point, collision_obs)
            arc1 = generate_arc_path(cur_pos, wp, steps=15)
            arc2 = generate_arc_path(wp, pickup_point, steps=15, force_final_heading=goal_heading)
            final_path.extend(arc1)
            final_path.extend(arc2)

        pickup_infos.append((pickup_point, goal_heading))
        cur_pos = pickup_point
        used_obstacles = [obs for obs in used_obstacles if not np.allclose(obs['center'], t_i)]

    return final_path, pickup_infos

# === 유전 알고리즘 ===
def fitness(order, positions):
    path_len = 0
    cur = positions[0]
    for idx in order:
        next_pos = positions[idx]
        path_len += np.linalg.norm(next_pos - cur)
        cur = next_pos
    return -path_len

def mutate(order):
    a, b = random.sample(range(len(order)), 2)
    order[a], order[b] = order[b], order[a]

def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = p1[a:b+1]
    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill[idx]
            idx += 1
    return child

def genetic_algorithm(trash_positions, start_position=(-65, 130), pop_size=100, generations=100, mutation_rate=0.2):
    positions = [np.array(start_position)] + [np.array(p) for p in trash_positions]
    n = len(positions)
    idxs = list(range(1, n))
    population = [random.sample(idxs, len(idxs)) for _ in range(pop_size)]
    best_order = []
    best_fit = float('-inf')

    for _ in range(generations):
        scored = []
        for indiv in population:
            order = [0] + indiv
            fit = fitness(order, positions)
            scored.append((indiv, fit))

        scored.sort(key=lambda x: x[1], reverse=True)
        population = [ind for ind, _ in scored[: pop_size // 2]]

        top_indiv, top_fit = scored[0]
        if top_fit > best_fit:
            best_fit = top_fit
            best_order = [0] + top_indiv[:]

        while len(population) < pop_size:
            p1, p2 = random.sample(population[:10], 2)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                mutate(child)
            population.append(child)

    return [i-1 for i in best_order[1:]], best_fit
