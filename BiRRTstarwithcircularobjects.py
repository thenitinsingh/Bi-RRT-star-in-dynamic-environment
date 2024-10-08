import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Node class definition
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

# BiRRTStar class definition
class BiRRTStar:
    def __init__(self, start, goal, obstacles, map_size, max_iter=1000, step_size=1.0, goal_sample_rate=5):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.map_size = map_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.start_tree = [self.start]
        self.goal_tree = [self.goal]

# Initialize the parameters
start = (0, 0)
goal = (10, 10)
obstacles = [
    (5, 5, 1),
    (3, 6, 2),
    (3, 8, 2),
    (3, 10, 2),
    (7, 5, 2),
    (9, 5, 2)
]
map_size = (60, 60)
rrt_star = BiRRTStar(start, goal, obstacles, map_size)

# Planning the path
path = None
for i in range(rrt_star.max_iter):
    if random.randint(0, 100) > rrt_star.goal_sample_rate:
        rnd_node = Node(random.uniform(0, rrt_star.map_size[0]), random.uniform(0, rrt_star.map_size[1]))
    else:
        rnd_node = Node(rrt_star.goal.x, rrt_star.goal.y)

    # Find the nearest node in the start tree
    nearest_node_start = rrt_star.start_tree[0]
    min_dist_start = math.hypot(nearest_node_start.x - rnd_node.x, nearest_node_start.y - rnd_node.y)
    for node in rrt_star.start_tree:
        dist = math.hypot(node.x - rnd_node.x, node.y - rnd_node.y)
        if dist < min_dist_start:
            nearest_node_start = node
            min_dist_start = dist

    # Steer towards the random node
    new_node_start = Node(nearest_node_start.x, nearest_node_start.y)
    dist = math.hypot(nearest_node_start.x - rnd_node.x, nearest_node_start.y - rnd_node.y)
    angle = math.atan2(rnd_node.y - nearest_node_start.y, rnd_node.x - nearest_node_start.x)
    new_node_start.x += rrt_star.step_size * math.cos(angle)
    new_node_start.y += rrt_star.step_size * math.sin(angle)
    new_node_start.parent = nearest_node_start

    # Collision checking for the new node in the start tree
    is_collision_free_start = True
    for ox, oy, size in rrt_star.obstacles:
        dx, dy = new_node_start.x - nearest_node_start.x, new_node_start.y - nearest_node_start.y
        fx, fy = nearest_node_start.x - ox, nearest_node_start.y - oy
        a = dx ** 2 + dy ** 2
        b = 2 * (fx * dx + fy * dy)
        c = (fx ** 2 + fy ** 2) - size ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            discriminant = math.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)
            if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                is_collision_free_start = False
                break

    if is_collision_free_start:
        rrt_star.start_tree.append(new_node_start)

        # Find the nearest node in the goal tree
        nearest_node_goal = rrt_star.goal_tree[0]
        min_dist_goal = math.hypot(nearest_node_goal.x - new_node_start.x, nearest_node_goal.y - new_node_start.y)
        for node in rrt_star.goal_tree:
            dist = math.hypot(node.x - new_node_start.x, node.y - new_node_start.y)
            if dist < min_dist_goal:
                nearest_node_goal = node
                min_dist_goal = dist

        # Steer towards the new node from the start tree
        new_node_goal = Node(nearest_node_goal.x, nearest_node_goal.y)
        dist = math.hypot(nearest_node_goal.x - new_node_start.x, nearest_node_goal.y - new_node_start.y)
        angle = math.atan2(new_node_start.y - nearest_node_goal.y, new_node_start.x - nearest_node_goal.x)
        new_node_goal.x += rrt_star.step_size * math.cos(angle)
        new_node_goal.y += rrt_star.step_size * math.sin(angle)
        new_node_goal.parent = nearest_node_goal

        # Collision checking for the new node in the goal tree
        is_collision_free_goal = True
        for ox, oy, size in rrt_star.obstacles:
            dx, dy = new_node_goal.x - nearest_node_goal.x, new_node_goal.y - nearest_node_goal.y
            fx, fy = nearest_node_goal.x - ox, nearest_node_goal.y - oy
            a = dx ** 2 + dy ** 2
            b = 2 * (fx * dx + fy * dy)
            c = (fx ** 2 + fy ** 2) - size ** 2
            discriminant = b ** 2 - 4 * a * c
            if discriminant >= 0:
                discriminant = math.sqrt(discriminant)
                t1 = (-b - discriminant) / (2 * a)
                t2 = (-b + discriminant) / (2 * a)
                if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                    is_collision_free_goal = False
                    break

        if is_collision_free_goal:
            rrt_star.goal_tree.append(new_node_goal)

            # Check if the trees have connected
            is_trees_connected = True
            for ox, oy, size in rrt_star.obstacles:
                dx, dy = new_node_goal.x - new_node_start.x, new_node_goal.y - new_node_start.y
                fx, fy = new_node_start.x - ox, new_node_start.y - oy
                a = dx ** 2 + dy ** 2
                b = 2 * (fx * dx + fy * dy)
                c = (fx ** 2 + fy ** 2) - size ** 2
                discriminant = b ** 2 - 4 * a * c
                if discriminant >= 0:
                    discriminant = math.sqrt(discriminant)
                    t1 = (-b - discriminant) / (2 * a)
                    t2 = (-b + discriminant) / (2 * a)
                    if 0 <= t1 <= 1 or 0 <= t2 <= 1:
                        is_trees_connected = False
                        break

            if is_trees_connected:
                path = []
                node = new_node_start
                while node is not None:
                    path.append((node.x, node.y))
                    node = node.parent
                path.reverse()

                node = new_node_goal.parent
                while node is not None:
                    path.append((node.x, node.y))
                    node = node.parent
                break

# Display the results
if path is None:
    print("No path found!")
else:
    print("Path found!")

    plt.figure()
    for ox, oy, size in obstacles:
        circle = plt.Circle((ox, oy), size, color='r')
        plt.gca().add_patch(circle)

    plt.plot(start[0], start[1], "bs")
    plt.plot(goal[0], goal[1], "gs")
    plt.plot([x for (x, y) in path], [y for (x, y) in path], 'b')
    plt.grid(True)
    plt.axis("equal")
    plt.show()
