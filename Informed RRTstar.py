import numpy as np
import matplotlib.pyplot as plt
import random
import math


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


class InformedRRTStar:
    def __init__(self, start, goal, obstacle_list, rand_area, max_iter=500):
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.obstacle_list = obstacle_list
        self.max_iter = max_iter
        self.node_list = [self.start]

    def get_random_node(self, c_best):
        c_min = np.linalg.norm([self.start.x - self.end.x, self.start.y - self.end.y])
        if c_best < float('inf') and c_best > c_min:
            x_center = np.array([(self.start.x + self.end.x) / 2.0, (self.start.y + self.end.y) / 2.0])
            a1 = np.array([[(self.end.x - self.start.x) / c_min, (self.end.y - self.start.y) / c_min]]).T
            etheta = math.atan2(self.end.y - self.start.y, self.end.x - self.start.x)
            rot = np.array([[math.cos(etheta), -math.sin(etheta)], [math.sin(etheta), math.cos(etheta)]])
            L = np.diag([c_best / 2.0, math.sqrt(c_best ** 2 - c_min ** 2) / 2.0])
            sample = self.sample_unit_ball()
            rnd = np.dot(rot, np.dot(L, sample)) + x_center
            return Node(rnd[0], rnd[1])
        else:
            rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand)]
            return Node(rnd[0], rnd[1])

    def sample_unit_ball(self):
        a = random.random()
        b = random.random()
        if b < a:
            a, b = b, a
        sample = [b * math.cos(2 * math.pi * a / b), b * math.sin(2 * math.pi * a / b)]
        return np.array(sample)

    def planning(self):
        for i in range(self.max_iter):
            c_best = self.get_best_cost()
            rnd_node = self.get_random_node(c_best)
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node)

            if self.collision_check(new_node, self.obstacle_list):
                near_indices = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indices)
                self.node_list.append(new_node)
                self.rewire(new_node, near_indices)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= 0.5:
                final_node = self.steer(self.node_list[-1], self.end)
                if self.collision_check(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def choose_parent(self, new_node, near_indices):
        if not near_indices:
            return new_node

        costs = []
        for i in near_indices:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.collision_check(t_node, self.obstacle_list):
                costs.append(near_node.cost + self.calc_dist(near_node, new_node))
            else:
                costs.append(float('inf'))

        min_cost = min(costs)
        min_ind = near_indices[costs.index(min_cost)]

        if min_cost == float('inf'):
            return new_node

        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]

        return new_node

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length)

        for _ in range(n_expand):
            new_node.x += math.cos(theta)
            new_node.y += math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= 1.0:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def get_best_cost(self):
        return min([node.cost + self.calc_dist_to_goal(node.x, node.y) for node in self.node_list])

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [np.linalg.norm([node.x - rnd_node.x, node.y - rnd_node.y]) for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        dlist = [np.linalg.norm([node.x - new_node.x, node.y - new_node.y]) for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r]
        return near_inds

    def rewire(self, new_node, near_indices):
        for i in near_indices:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue

            edge_node.cost = new_node.cost + self.calc_dist(new_node, near_node)

            if self.collision_check(edge_node, self.obstacle_list) and near_node.cost > edge_node.cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.parent = new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.sqrt(dx ** 2 + dy ** 2)
        theta = math.atan2(dy, dx)
        return d, theta

    def calc_dist(self, node1, node2):
        return np.linalg.norm([node1.x - node2.x, node1.y - node2.y])

    def collision_check(self, node, obstacle_list):
        if node is None:
            return False

        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size ** 2:
                return False

        return True

    def draw_graph(self, rnd=None):
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        plt.grid(True)
        plt.pause(0.01)

    def plot_circle(self, x, y, size):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, "-k")


def main():
    print("Start Informed RRT* planning")

    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]
    rand_area = [0, 15]

    rrt_star = InformedRRTStar(start=[0, 0], goal=[10, 10], obstacle_list=obstacle_list, rand_area=rand_area)
    path = rrt_star.planning()

    if path is None:
        print("No path found")
    else:
        print("Found path!!")

        if True:  # Displaying the final path
            rrt_star.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.show()


if __name__ == '__main__':
    main()
