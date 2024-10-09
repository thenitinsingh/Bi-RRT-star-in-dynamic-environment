import numpy as np
import matplotlib.pyplot as plt
import random
import math
from shapely.geometry import Point, Polygon, LineString
start_x, start_y = 0, 0
goal_x, goal_y = 25, 25
bonus_boundary = 0.5
#Obstacles
original_obstacles = [
    (8, 20, 2),   # Circle obstacle with center (8, 20) and radius 2
    (22, 7, 4),   # Circle obstacle with center (22, 7) and radius 4
    (15, 15, 3),  # Circle obstacle with center (15, 15) and radius 3
    [(18, 22), (25, 22), (25, 20), (18, 20)],  # Rectangle obstacle
    [(12, 5), (14, 5), (14, 7), (12, 7)]       # Rectangle obstacle
]
#Adding the bonus boundary
effective_obstacles = []
for obstacle in original_obstacles:
    if isinstance(obstacle, tuple) and len(obstacle) == 3:
        ox, oy, radius = obstacle
        effective_obstacles.append((ox, oy, radius + bonus_boundary))
    elif isinstance(obstacle, list):
        obstacle_polygon = Polygon(obstacle)
        enlarged_polygon = obstacle_polygon.buffer(bonus_boundary)
        effective_obstacles.append(list(enlarged_polygon.exterior.coords))
map_size_x, map_size_y = 30, 30
max_iter = 1000
step_size = 1.0
goal_sample_rate = 5

start_tree = [(start_x, start_y, None)]
goal_tree = [(goal_x, goal_y, None)]

path = None
for i in range(max_iter):
    #Randomly sample a point or choose the goal
    if random.randint(0, 100) > goal_sample_rate:
        rnd_x, rnd_y = random.uniform(0, map_size_x), random.uniform(0, map_size_y)
    else:
        rnd_x, rnd_y = goal_x, goal_y
    #Finding the nearest node in the start tree
    nearest_node_start = start_tree[0]
    min_dist_start = math.hypot(nearest_node_start[0] - rnd_x, nearest_node_start[1] - rnd_y)
    for node in start_tree:
        dist = math.hypot(node[0] - rnd_x, node[1] - rnd_y)
        if dist < min_dist_start:
            nearest_node_start = node
            min_dist_start = dist
    new_x_start, new_y_start = nearest_node_start[0], nearest_node_start[1]
    angle = math.atan2(rnd_y - nearest_node_start[1], rnd_x - nearest_node_start[0])
    new_x_start += step_size * math.cos(angle)
    new_y_start += step_size * math.sin(angle)
    new_node_start = (new_x_start, new_y_start, nearest_node_start)
    is_collision_free_start = True
    for obstacle in effective_obstacles:
        if isinstance(obstacle, tuple) and len(obstacle) == 3:
            ox, oy, radius = obstacle
            line = LineString([(nearest_node_start[0], nearest_node_start[1]), (new_x_start, new_y_start)])
            if line.distance(Point(ox, oy)) <= radius:
                is_collision_free_start = False
                break
        elif isinstance(obstacle, list):
            obstacle_polygon = Polygon(obstacle)
            line = LineString([(nearest_node_start[0], nearest_node_start[1]), (new_x_start, new_y_start)])
            if line.intersects(obstacle_polygon):
                is_collision_free_start = False
                break

    if is_collision_free_start:
        start_tree.append(new_node_start)
        nearest_node_goal = goal_tree[0]
        min_dist_goal = math.hypot(nearest_node_goal[0] - new_x_start, nearest_node_goal[1] - new_y_start)
        for node in goal_tree:
            dist = math.hypot(node[0] - new_x_start, node[1] - new_y_start)
            if dist < min_dist_goal:
                nearest_node_goal = node
                min_dist_goal = dist
        new_x_goal, new_y_goal = nearest_node_goal[0], nearest_node_goal[1]
        angle = math.atan2(new_y_start - nearest_node_goal[1], new_x_start - nearest_node_goal[0])
        new_x_goal += step_size * math.cos(angle)
        new_y_goal += step_size * math.sin(angle)
        new_node_goal = (new_x_goal, new_y_goal, nearest_node_goal)
        is_collision_free_goal = True
        for obstacle in effective_obstacles:
            if isinstance(obstacle, tuple) and len(obstacle) == 3:
                ox, oy, radius = obstacle
                line = LineString([(nearest_node_goal[0], nearest_node_goal[1]), (new_x_goal, new_y_goal)])
                if line.distance(Point(ox, oy)) <= radius:
                    is_collision_free_goal = False
                    break
            elif isinstance(obstacle, list):
                obstacle_polygon = Polygon(obstacle)
                line = LineString([(nearest_node_goal[0], nearest_node_goal[1]), (new_x_goal, new_y_goal)])
                if line.intersects(obstacle_polygon):
                    is_collision_free_goal = False
                    break

        if is_collision_free_goal:
            goal_tree.append(new_node_goal)
            is_trees_connected = True
            for obstacle in effective_obstacles:
                if isinstance(obstacle, tuple) and len(obstacle) == 3:
                    ox, oy, radius = obstacle
                    line = LineString([(new_x_goal, new_y_goal), (new_x_start, new_y_start)])
                    if line.distance(Point(ox, oy)) <= radius:
                        is_trees_connected = False
                        break
                elif isinstance(obstacle, list):
                    obstacle_polygon = Polygon(obstacle)
                    line = LineString([(new_x_goal, new_y_goal), (new_x_start, new_y_start)])
                    if line.intersects(obstacle_polygon):
                        is_trees_connected = False
                        break
            if is_trees_connected:
                path = []
                node = new_node_start
                while node is not None:
                    path.append((node[0], node[1]))
                    node = node[2]
                path.reverse()
                node = new_node_goal[2]
                while node is not None:
                    path.append((node[0], node[1]))
                    node = node[2]
                break
# Display the results
if path is None:
    print("No path found!")
else:
    print("Path found!")
    plt.figure()

    # Draw original obstacles
    for obstacle in original_obstacles:
        if isinstance(obstacle, tuple) and len(obstacle) == 3:  # Circle obstacle
            ox, oy, radius = obstacle
            circle = plt.Circle((ox, oy), radius, color='r', alpha=0.5)
            plt.gca().add_patch(circle)
        elif isinstance(obstacle, list):  # Rectangle (Polygon) obstacle
            obstacle_polygon = Polygon(obstacle)
            x, y = obstacle_polygon.exterior.xy
            plt.fill(x, y, color='r', alpha=0.5)

    # Draw effective boundaries as dotted lines
    for obstacle in effective_obstacles:
        if isinstance(obstacle, tuple) and len(obstacle) == 3:  # Circle obstacle
            ox, oy, radius = obstacle
            circle = plt.Circle((ox, oy), radius, color='g', linestyle='--', fill=False)
            plt.gca().add_patch(circle)
        elif isinstance(obstacle, list):  # Rectangle (Polygon) obstacle
            obstacle_polygon = Polygon(obstacle)
            x, y = obstacle_polygon.exterior.xy
            plt.plot(x, y, 'g--')

    # Plot the path
    plt.plot(start_x, start_y, "bs")
    plt.plot(goal_x, goal_y, "gs")
    plt.plot([x for (x, y) in path], [y for (x, y) in path], 'b')
    plt.grid(True)
    plt.axis("equal")
    plt.show()
