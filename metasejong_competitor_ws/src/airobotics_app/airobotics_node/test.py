from collections import defaultdict
print("i")
from obj_detect_module import detect_objects
print("i")
import order_decision
print("i")
import matplotlib
print("i")
#matplotlib.use("TkAgg")
print("i")
import matplotlib.pyplot as plt
print("i")
import numpy as np
print("i")

foo = detect_objects()

robot_position = [-65.0, 130.0, 17.0] 

bar = order_decision.visit_order(foo, robot_position)[1:]
object_coords = np.array([value for d in bar for value in d.values()])
print(object_coords)
"""
plt.scatter(object_coords[:,0], object_coords[:,1])
plt.scatter(robot_position[0], robot_position[1], c='green')
plt.plot([robot_position[0], object_coords[0,0]], [robot_position[1], object_coords[0,1]], 'r')
for i in range(np.size(object_coords, axis=0)-1):
    p, q = object_coords[i], object_coords[i+1]
    plt.plot([p[0], q[0]], [p[1], q[1]], 'r')
plt.show()
"""