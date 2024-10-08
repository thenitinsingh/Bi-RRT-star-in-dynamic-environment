import numpy as np
import matplotlib.pyplot as plt
import random
import math


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


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

    def plan(self):
        for i in range(self.max_iter):
            if random.randint(0, 100) > self.goal_sample_rate:
                rnd_node = self.get_random_node()
            else:
                rnd_node = Node(self.goal.x, self.goal.y)

            nearest_node_start = self.get_nearest_node(self.start_tree, rnd_node)
            new_node_start = self.steer(nearest_node_start, rnd_node)

            if self.is_collision_free(new_node_start, nearest_node_start):
                self.start_tree.append(new_node_start)
                nearest_node_goal = self.get_nearest_node(self.goal_tree, new_node_start)
                new_node_goal = self.steer(nearest_node_goal, new_node_start)

                if self.is_collision_free(new_node_goal, nearest_node_goal):
                    self.goal_tree.append(new_node_goal)
                    if self.is_collision_free(new_node_goal, new_node_start):
                        return self.generate_final_path(new_node_start, new_node_goal)

        return None

    def steer(self, from_node, to_node):
        new_node = Node(from_node.x, from_node.y)
        dist = self.get_distance(from_node, to_node)
        angle = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)

        new_node.x += self.step_size * math.cos(angle)
        new_node.y += self.step_size * math.sin(angle)
        new_node.parent = from_node

        return new_node

    def get_random_node(self):
        return Node(random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1]))

    def get_nearest_node(self, tree, node):
        return min(tree, key=lambda n: self.get_distance(n, node))

    def get_distance(self, node1, node2):
        return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def is_collision_free(self, node1, node2):
        for ox, oy, size in self.obstacles:
            if self.line_intersects_circle(node1.x, node1.y, node2.x, node2.y, ox, oy, size):
                return False
        return True

    def line_intersects_circle(self, x1, y1, x2, y2, cx, cy, r):
        dx, dy = x2 - x1, y2 - y1
        fx, fy = x1 - cx, y1 - cy
        a = dx ** 2 + dy ** 2
        b = 2 * (fx * dx + fy * dy)
        c = (fx ** 2 + fy ** 2) - r ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0:
            return False

        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        return 0 <= t1 <= 1 or 0 <= t2 <= 1

    def generate_final_path(self, start_node, goal_node):
        path = []
        node = start_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()

        node = goal_node.parent
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent

        return path


def main():
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

    path = rrt_star.plan()

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


if __name__ == '__main__':
    main()
