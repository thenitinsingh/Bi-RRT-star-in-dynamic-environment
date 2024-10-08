import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # Initialize the cost attribute


class BiRRTStar:
    def __init__(self, start, goal, obstacles, max_iter=500, step_size=5, goal_sample_rate=0.1, search_radius=10):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.start_tree = [self.start]
        self.goal_tree = [self.goal]

    def plan(self):
        for _ in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node_start = self.get_nearest_node(self.start_tree, rnd_node)
            new_node_start = self.steer(nearest_node_start, rnd_node)

            if self.is_collision_free(nearest_node_start, new_node_start):
                new_node_start.cost = nearest_node_start.cost + self.distance(nearest_node_start, new_node_start)
                self.start_tree.append(new_node_start)
                self.rewire(self.start_tree, new_node_start)

                nearest_node_goal = self.get_nearest_node(self.goal_tree, new_node_start)
                new_node_goal = self.steer(nearest_node_goal, new_node_start)

                if self.is_collision_free(nearest_node_goal, new_node_goal):
                    new_node_goal.cost = nearest_node_goal.cost + self.distance(nearest_node_goal, new_node_goal)
                    self.goal_tree.append(new_node_goal)
                    self.rewire(self.goal_tree, new_node_goal)

                    if self.distance(new_node_start, new_node_goal) < self.step_size:
                        return self.generate_final_course(new_node_start, new_node_goal)

            # Swap start and goal trees
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree

        return None  # Path not found

    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            return Node(np.random.rand() * 100, np.random.rand() * 100)
        else:
            return self.goal

    def get_nearest_node(self, tree, node):
        return min(tree, key=lambda n: self.distance(n, node))

    def steer(self, from_node, to_node):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.distance_and_angle(from_node, to_node)
        new_node.x += self.step_size * np.cos(theta)
        new_node.y += self.step_size * np.sin(theta)
        new_node.parent = from_node
        return new_node

    def is_collision_free(self, from_node, to_node):
        # Simple collision detection
        for obs in self.obstacles:
            if self.check_collision(from_node, to_node, obs):
                return False
        return True

    def check_collision(self, from_node, to_node, obs):
        # Implement actual collision detection here
        return False

    def rewire(self, tree, new_node):
        for node in tree:
            if self.distance(node, new_node) < self.search_radius:
                potential_cost = node.cost + self.distance(node, new_node)
                if self.is_collision_free(node, new_node) and potential_cost < new_node.cost:
                    new_node.parent = node
                    new_node.cost = potential_cost

    def distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    def distance(self, node1, node2):
        return np.hypot(node1.x - node2.x, node1.y - node2.y)

    def generate_final_course(self, node_start, node_goal):
        path = [(node_goal.x, node_goal.y)]
        node = node_goal
        while node.parent is not None:
            node = node.parent
            path.append((node.x, node.y))

        path.reverse()

        node = node_start
        while node.parent is not None:
            node = node.parent
            path.append((node.x, node.y))

        return path


# Example usage:
start = [0, 0]
goal = [100, 100]
obstacles = []  # Add obstacles if needed
birrt_star = BiRRTStar(start, goal, obstacles)
path = birrt_star.plan()

# Plotting the result
if path:
    plt.figure()
    for obs in obstacles:
        plt.plot(obs[0], obs[1], 'k')
    plt.plot(start[0], start[1], 'ro')
    plt.plot(goal[0], goal[1], 'go')
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-b')
    plt.grid(True)
    plt.show()
else:
    print("Path not found!")
