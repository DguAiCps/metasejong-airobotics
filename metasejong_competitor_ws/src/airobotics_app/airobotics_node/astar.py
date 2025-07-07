#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from tf2_msgs.msg import TFMessage
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import numpy as np
import heapq, math


# TODO: 쓰레기 근처로 가면 cost 증가하도록 만들기
def soft_cost(r: int, c: int) -> float:
    return 0.0

def astar(grid: np.ndarray,
          start: tuple[int, int],
          goal: tuple[int, int],
          soft_cost_func=soft_cost) -> list[tuple[int, int]] | None:
    print("a star 알고리즘 시작")
    rows, cols = grid.shape
    directions = [(-1,  0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                  (-1,-1, math.sqrt(2)), (-1,1, math.sqrt(2)),
                  (1,-1, math.sqrt(2)), (1,1, math.sqrt(2))]

    g_score = np.full((rows, cols), np.inf, dtype=float)
    parent  = np.full((rows, cols, 2), -1, dtype=int)
    closed  = np.zeros((rows, cols), dtype=bool)

    def h(r, c):
        return math.hypot(goal[0] - r, goal[1] - c)

    open_heap = []
    sr, sc = start; gr, gc = goal
    if grid[sr, sc] != 0 or grid[gr, gc] != 0:
        return None
    g_score[sr, sc] = 0.0
    heapq.heappush(open_heap, (h(sr, sc), sr, sc))

    while open_heap:
        f, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        if (r, c) == goal:
            path = [(r, c)]
            while parent[r, c][0] != -1:
                r, c = parent[r, c]
                path.append((r, c))
            return path[::-1]

        closed[r, c] = True
        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] != 0:
                continue
            tentative = g_score[r, c] + cost + soft_cost_func(nr, nc)
            if tentative < g_score[nr, nc]:
                g_score[nr, nc] = tentative
                parent[nr, nc] = (r, c)
                heapq.heappush(open_heap, (tentative + h(nr, nc), nr, nc))
    return None

def world_to_grid(coords, map, tf):
    x = coords[0]
    y = coords[1]
    res   = map.info.resolution
    ox, oy = map.info.origin.position.x, map.info.origin.position.y

    print(f"map coords: {(x + 65 + tf.translation.x)}, {(y - 130 + tf.translation.y)}")
    col = int((x + 65 - ox + tf.translation.x)/res)
    row = int((y - 130 - oy + tf.translation.y)/res)
    print(row, col)
    return row, col