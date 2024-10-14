import numpy as np
import matplotlib.pyplot as plt
import random
import math
from shapely.geometry import Point, Polygon, LineString
from scipy.interpolate import splev, splprep


# Collision detection function
def is_collision_free(start, end, obstacles):
    line = LineString([start, end])
    for obstacle in obstacles:
        if isinstance(obstacle, tuple) and len(obstacle) == 3:
            ox, oy, radius = obstacle
            if line.distance(Point(ox, oy)) <= radius:
                return False
        elif isinstance(obstacle, list):
            obstacle_polygon = Polygon(obstacle)
            if line.intersects(obstacle_polygon):
                return False
    return True


# Calculating path distance
def calculate_path_distance(path):
    total_distance = 0.0
    if path is None:
        return float('inf')
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        total_distance += math.hypot(x2 - x1, y2 - y1)
    return total_distance


# Bi-RRT* algorithm implementation
def bi_rrt_star(start_x, start_y, goal_x, goal_y, effective_obstacles):
    map_size_x, map_size_y = 30, 30
    max_iter = 1000
    step_size = 1.0
    goal_sample_rate = 5
    start_tree = [(start_x, start_y, None)]
    goal_tree = [(goal_x, goal_y, None)]
    path = None
    all_paths = []
    for i in range(max_iter):
        if random.randint(0, 100) > goal_sample_rate:
            rnd_x, rnd_y = random.uniform(0, map_size_x), random.uniform(0, map_size_y)
        else:
            rnd_x, rnd_y = goal_x, goal_y
        nearest_node_start = min(start_tree, key=lambda node: math.hypot(node[0] - rnd_x, node[1] - rnd_y))
        new_x_start, new_y_start = nearest_node_start[0], nearest_node_start[1]
        angle = math.atan2(rnd_y - nearest_node_start[1], rnd_x - nearest_node_start[0])
        new_x_start += step_size * math.cos(angle)
        new_y_start += step_size * math.sin(angle)
        new_node_start = (new_x_start, new_y_start, nearest_node_start)
        if is_collision_free((nearest_node_start[0], nearest_node_start[1]), (new_x_start, new_y_start),
                             effective_obstacles):
            start_tree.append(new_node_start)
            nearest_node_goal = min(goal_tree,
                                    key=lambda node: math.hypot(node[0] - new_x_start, node[1] - new_y_start))
            angle = math.atan2(new_y_start - nearest_node_goal[1], new_x_start - nearest_node_goal[0])
            new_x_goal = nearest_node_goal[0] + step_size * math.cos(angle)
            new_y_goal = nearest_node_goal[1] + step_size * math.sin(angle)
            new_node_goal = (new_x_goal, new_y_goal, nearest_node_goal)
            if is_collision_free((nearest_node_goal[0], nearest_node_goal[1]), (new_x_goal, new_y_goal),
                                 effective_obstacles):
                goal_tree.append(new_node_goal)

                # Check if trees can be connected
                if is_collision_free((new_x_goal, new_y_goal), (new_x_start, new_y_start), effective_obstacles):
                    # Build path from start to goal
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

        # Store the intermediate path for later plotting
        if new_node_start[2] is not None:
            all_paths.append((new_node_start, new_node_start[2]))

    return path, all_paths


# Shift obstacles function
def shift_obstacles(obstacles, shift_x, shift_y):
    shifted_obstacles = []
    for obstacle in obstacles:
        if isinstance(obstacle, tuple) and len(obstacle) == 3:  # Circular obstacle
            ox, oy, radius = obstacle
            shifted_obstacles.append((ox + shift_x, oy + shift_y, radius))
        elif isinstance(obstacle, list):  # Polygon obstacle
            shifted_obstacles.append([(x + shift_x, y + shift_y) for x, y in obstacle])
    return shifted_obstacles


# Adding the bonus boundary
def add_bonus_boundary(obstacles, bonus_boundary):
    effective_obstacles = []
    for obstacle in obstacles:
        if isinstance(obstacle, tuple) and len(obstacle) == 3:  # Circular obstacle
            ox, oy, radius = obstacle
            effective_obstacles.append((ox, oy, radius + bonus_boundary))
        elif isinstance(obstacle, list):  # Polygon obstacle
            obstacle_polygon = Polygon(obstacle)
            enlarged_polygon = obstacle_polygon.buffer(bonus_boundary)
            effective_obstacles.append(list(enlarged_polygon.exterior.coords))
    return effective_obstacles


# Smooth path using B-spline
def smooth_path_b_spline(path, smoothing_factor=0):
    x, y = zip(*path)
    tck, u = splprep([x, y], s=smoothing_factor, k=3)
    u_fine = np.linspace(0, 1, num=100)
    x_fine, y_fine = splev(u_fine, tck)
    return list(zip(x_fine, y_fine))


# Display results function
def display_results(obstacles, effective_obstacles, path, smooth_path, all_paths, distance, title):
    plt.figure()
    plt.title(f"{title}")
    print(f"{distance:.2f} units")

    # Draw original obstacles
    for obstacle in obstacles:
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

    # Plot all intermediate paths
    for node, parent in all_paths:
        plt.plot([parent[0], node[0]], [parent[1], node[1]], 'y--')

    # Plot the final path
    if path:
        plt.plot([x for (x, y) in path], [y for (x, y) in path], 'b--', label='Final Path')

    # Plot the smoothed path
    if smooth_path:
        plt.plot([x for (x, y) in smooth_path], [y for (x, y) in smooth_path], 'b', label='Smoothed Path')

    plt.plot(start_x, start_y, "bs", label="Start")
    plt.plot(goal_x, goal_y, "gs", label="Goal")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()


# Initial parameters
start_x, start_y = 0, 0
goal_x, goal_y = 25, 30
bonus_boundary = 0.5
shift_x = 0  # Shift value for x
shift_y = 2  # Shift value for y
# Obstacles definition
original_obstacles = [
    (8, 20, 2),  # Circle obstacle with center (8, 20) and radius 2
    (22, 7, 4),  # Circle obstacle with center (22, 7) and radius 4
    (15, 15, 3),  # Circle obstacle with center (15, 15) and radius 3
    [(18, 22), (25, 22), (25, 20), (18, 20)],  # Rectangle obstacle
    [(12, 5), (14, 5), (14, 7), (12, 7)]  # Rectangle obstacle
]
# Apply shift and bonus boundary
shifted_obstacles = shift_obstacles(original_obstacles, shift_x, shift_y)
effective_obstacles_original = add_bonus_boundary(original_obstacles, bonus_boundary)
effective_obstacles_shifted = add_bonus_boundary(shifted_obstacles, bonus_boundary)
# Find paths for both scenarios
path_original, all_paths_original = bi_rrt_star(start_x, start_y, goal_x, goal_y, effective_obstacles_original)
path_shifted, all_paths_shifted = bi_rrt_star(start_x, start_y, goal_x, goal_y, effective_obstacles_shifted)
# Smooth paths
smooth_path_original = smooth_path_b_spline(path_original)
smooth_path_shifted = smooth_path_b_spline(path_shifted)
# Calculate path distances
distance_original = calculate_path_distance(path_original) if path_original else float('inf')
distance_shifted = calculate_path_distance(path_shifted) if path_shifted else float('inf')
# Display results for original and shifted obstacles
display_results(original_obstacles, effective_obstacles_original, path_original, smooth_path_original,
                all_paths_original, distance_original, "With Original Obstacles")
display_results(shifted_obstacles, effective_obstacles_shifted, path_shifted, smooth_path_shifted, all_paths_shifted,
                distance_shifted, "With Shifted Obstacles")
plt.show()