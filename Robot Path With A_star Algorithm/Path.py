import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import numpy as np
from shapely.geometry.polygon import LinearRing, Polygon
from math import *
import networkx as nx
import random
from shapely.geometry import LineString


def calculate_distance(point1, point2):
    return sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2))


# toate distantele posibile de la varfurile poligoanelor(intre figuri)

possible_distances = []
possible_distances_without_cost = []

# dreptunghi albastru stanga jos
possible_distances.append(((1, 1), (0.5, 8), calculate_distance((1, 1), (0.5, 8))))
possible_distances.append(((1, 1), (0.8, 4.5), calculate_distance((1, 1), (0.8, 4.5))))
possible_distances.append(((1, 1), (1, 3), calculate_distance((1, 1), (1, 3))))
possible_distances.append(((1, 1), (5, 1), calculate_distance((1, 1), (5, 1))))

# dreptunghi albastru stanga sus
possible_distances.append(((1, 3), (0.8, 4.5), calculate_distance((1, 3), (0.8, 4.5))))
possible_distances.append(((1, 3), (2, 4), calculate_distance((1, 3), (2, 4))))
possible_distances.append(((1, 3), (3, 4), calculate_distance((1, 3), (3, 4))))
possible_distances.append(((1, 3), (4, 4), calculate_distance((1, 3), (4, 4))))
possible_distances.append(((1, 3), (5, 3), calculate_distance((1, 3), (5, 3))))

# dreptunghi albastru dreapta sus
possible_distances.append(((5, 3), (3, 4), calculate_distance((5, 3), (3, 4))))
possible_distances.append(((5, 3), (4, 4), calculate_distance((5, 3), (4, 4))))
possible_distances.append(((5, 3), (3.5, 8), calculate_distance((5, 3), (3.5, 8))))
possible_distances.append(((5, 3), (2, 4), calculate_distance((5, 3), (2, 4))))
possible_distances.append(((5, 3), (5, 1), calculate_distance((5, 3), (5, 1))))
possible_distances.append(((5, 3), (5, 8), calculate_distance((5, 3), (5, 8))))
possible_distances.append(((5, 3), (6, 2), calculate_distance((5, 3), (6, 2))))
possible_distances.append(((5, 3), (5.5, 7), calculate_distance((5, 3), (5.5, 7))))

# dreptunghi albastru dreapta jos
possible_distances.append(((5, 1), (6, 2), calculate_distance((5, 1), (6, 2))))
possible_distances.append(((5, 1), (5.5, 7), calculate_distance((5, 1), (5.5, 7))))
possible_distances.append(((5, 1), (7, 2), calculate_distance((5, 1), (7, 2))))
possible_distances.append(((5, 1), (8, 1), calculate_distance((5, 1), (8, 1))))

# verde stanga jos
possible_distances.append(((0.8, 4.5), (2, 4), calculate_distance((0.8, 4.5), (2, 4))))
possible_distances.append(((0.8, 4.5), (0.5, 8), calculate_distance((0.8, 4.5), (0.5, 8))))

# verde stanga sus
possible_distances.append(((0.5, 8), (2, 10), calculate_distance((0.5, 8), (2, 10))))

# verde varf
possible_distances.append(((2, 10), (2.5, 8), calculate_distance((2, 10), (2.5, 8))))
possible_distances.append(((2, 10), (5, 10), calculate_distance((2, 10), (5, 10))))
possible_distances.append(((2, 10), (5, 8), calculate_distance((2, 10), (5, 8))))
possible_distances.append(((2, 10), (3.5, 8), calculate_distance((2, 10), (3.5, 8))))
possible_distances.append(((2, 10), (5.5, 7), calculate_distance((2, 10), (5.5, 7))))
possible_distances.append(((2, 10), (6.5, 6.5), calculate_distance((2, 10), (6.5, 6.5))))

# verde dreapta sus
possible_distances.append(((2.5, 8), (5, 10), calculate_distance((2.5, 8), (5, 10))))
possible_distances.append(((2.5, 8), (3.5, 8), calculate_distance((2.5, 8), (3.5, 8))))
possible_distances.append(((2.5, 8), (3, 4), calculate_distance((2.5, 8), (3, 4))))
possible_distances.append(((2.5, 8), (2, 4), calculate_distance((2.5, 8), (2, 4))))

# verde dreapta jos
possible_distances.append(((2, 4), (3.5, 8), calculate_distance((2, 4), (3.5, 8))))
possible_distances.append(((2, 4), (3, 4), calculate_distance((2, 4), (3, 4))))

# portocaliu stanga jos
possible_distances.append(((3, 4), (4, 4), calculate_distance((3, 4), (4, 4))))
possible_distances.append(((3, 4), (3.5, 8), calculate_distance((3, 4), (3.5, 8))))

# portocaliu varf
possible_distances.append(((3.5, 8), (4, 4), calculate_distance((3.5, 8), (4, 4))))
possible_distances.append(((3.5, 8), (5, 10), calculate_distance((3.5, 8), (5, 10))))
possible_distances.append(((3.5, 8), (5, 8), calculate_distance((3.5, 8), (5, 8))))
possible_distances.append(((3.5, 8), (5.5, 7), calculate_distance((3.5, 8), (5.5, 7))))
possible_distances.append(((3.5, 8), (6, 2), calculate_distance((3.5, 8), (6, 2))))

# portocaliu dreapta jos
possible_distances.append(((4, 4), (5, 10), calculate_distance((4, 4), (5, 10))))
possible_distances.append(((4, 4), (5, 8), calculate_distance((4, 4), (5, 8))))
possible_distances.append(((4, 4), (5.5, 7), calculate_distance((4, 4), (5.5, 7))))

# rosu stanga jos
possible_distances.append(((5, 8), (5, 10), calculate_distance((5, 8), (5, 10))))
possible_distances.append(((5, 8), (6, 9), calculate_distance((5, 8), (6, 9))))
possible_distances.append(((5, 8), (5.5, 7), calculate_distance((5, 8), (5.5, 7))))
possible_distances.append(((5, 8), (6, 2), calculate_distance((5, 8), (6, 2))))
possible_distances.append(((5, 8), (6.5, 6.5), calculate_distance((5, 8), (6.5, 6.5))))

# rosu stanga sus
possible_distances.append(((5, 10), (5.5, 10), calculate_distance((5, 10), (5.5, 10))))

# rosu dreapta sus
possible_distances.append(((5.5, 10), (6, 9), calculate_distance((5.5, 10), (6, 9))))
possible_distances.append(((5.5, 10), (6.5, 9.5), calculate_distance((5.5, 10), (6.5, 9.5))))
possible_distances.append(((5.5, 10), (7, 9.5), calculate_distance((5.5, 10), (7, 9.5))))
possible_distances.append(((5.5, 10), (10, 10), calculate_distance((5.5, 10), (10, 10))))
possible_distances.append(((5.5, 10), (8, 8), calculate_distance((5.5, 10), (8, 8))))

# rosu dreapta jos
possible_distances.append(((6, 9), (6.5, 9.5), calculate_distance((6, 9), (6.5, 9.5))))
possible_distances.append(((6, 9), (6.5, 6.5), calculate_distance((6, 9), (6.5, 6.5))))
possible_distances.append(((6, 9), (5.5, 7), calculate_distance((6, 9), (5.5, 7))))

# mov varf
possible_distances.append(((5.5, 7), (6, 2), calculate_distance((5.5, 7), (6, 2))))
possible_distances.append(((5.5, 7), (7, 3.5), calculate_distance((5.5, 7), (7, 3.5))))
possible_distances.append(((5.5, 7), (6.5, 6.5), calculate_distance((5.5, 7), (6.5, 6.5))))
possible_distances.append(((5.5, 7), (6.5, 9.5), calculate_distance((5.5, 7), (6.5, 9.5))))
possible_distances.append(((5.5, 7), (8, 4), calculate_distance((5.5, 7), (8, 4))))

# mov stanga jos
possible_distances.append(((6, 2), (7, 3.5), calculate_distance((6, 2), (7, 3.5))))
possible_distances.append(((6, 2), (7, 3), calculate_distance((6, 2), (7, 3))))
possible_distances.append(((6, 2), (7, 2), calculate_distance((6, 2), (7, 2))))
possible_distances.append(((6, 2), (8, 1), calculate_distance((6, 2), (8, 1))))

# mov dreapta sus
possible_distances.append(((7, 3.5), (7, 3), calculate_distance((7, 3.5), (7, 3))))
possible_distances.append(((7, 3.5), (8, 4), calculate_distance((7, 3.5), (8, 4))))
possible_distances.append(((7, 3.5), (8, 8), calculate_distance((7, 3.5), (8, 8))))
possible_distances.append(((7, 3.5), (6.5, 6.5), calculate_distance((7, 3.5), (6.5, 6.5))))
possible_distances.append(((7, 3.5), (7, 6.5), calculate_distance((7, 3.5), (7, 6.5))))

# maro stanga jos
possible_distances.append(((6.5, 6.5), (7, 6.5), calculate_distance((6.5, 6.5), (7, 6.5))))
possible_distances.append(((6.5, 6.5), (6.5, 9.5), calculate_distance((6.5, 6.5), (6.5, 9.5))))
possible_distances.append(((6.5, 6.5), (8, 4), calculate_distance((6.5, 6.5), (8, 4))))
possible_distances.append(((6.5, 6.5), (9.5, 3), calculate_distance((6.5, 6.5), (9.5, 3))))
possible_distances.append(((6.5, 6.5), (10.5, 4), calculate_distance((6.5, 6.5), (10.5, 4))))

# maro stanga sus
possible_distances.append(((6.5, 9.5), (7, 9.5), calculate_distance((6.5, 9.5), (7, 9.5))))
possible_distances.append(((6.5, 9.5), (10, 10), calculate_distance((6.5, 9.5), (10, 10))))

# maro dreapta sus
possible_distances.append(((7, 9.5), (7, 6.5), calculate_distance((7, 9.5), (7, 6.5))))
possible_distances.append(((7, 9.5), (10, 10), calculate_distance((7, 9.5), (10, 10))))
possible_distances.append(((7, 9.5), (8, 8), calculate_distance((7, 9.5), (8, 8))))
possible_distances.append(((7, 9.5), (10.5, 4), calculate_distance((7, 9.5), (10.5, 4))))
possible_distances.append(((7, 9.5), (8, 4), calculate_distance((7, 9.5), (8, 4))))
possible_distances.append(((7, 9.5), (9.5, 3), calculate_distance((7, 9.5), (9.5, 3))))

# maro dreapta jos
possible_distances.append(((7, 6.5), (8, 4), calculate_distance((7, 6.5), (8, 4))))
possible_distances.append(((7, 6.5), (9.5, 3), calculate_distance((7, 6.5), (9.5, 3))))
possible_distances.append(((7, 6.5), (10.5, 4), calculate_distance((7, 6.5), (10.5, 4))))
possible_distances.append(((7, 6.5), (8, 8), calculate_distance((7, 6.5), (8, 8))))

# gri varf jos
possible_distances.append(((8, 1), (7, 2), calculate_distance((8, 1), (7, 2))))
possible_distances.append(((8, 1), (9.5, 1.5), calculate_distance((8, 1), (9.5, 1.5))))

# gri stanga jos
possible_distances.append(((7, 2), (7, 3), calculate_distance((7, 2), (7, 3))))

# gri stanga sus
possible_distances.append(((7, 3), (8, 4), calculate_distance((7, 3), (8, 4))))
possible_distances.append(((7, 3), (8, 8), calculate_distance((7, 3), (8, 8))))

# gri varf sus
possible_distances.append(((8, 4), (9.5, 3), calculate_distance((8, 4), (9.5, 3))))
possible_distances.append(((8, 4), (8, 8), calculate_distance((8, 4), (8, 8))))
possible_distances.append(((8, 4), (10.5, 4), calculate_distance((8, 4), (10.5, 4))))

# gri dreapta sus
possible_distances.append(((9.5, 3), (9.5, 1.5), calculate_distance((9.5, 3), (9.5, 1.5))))
possible_distances.append(((9.5, 3), (8, 8), calculate_distance((9.5, 3), (8, 8))))
possible_distances.append(((9.5, 3), (10.5, 4), calculate_distance((9.5, 3), (10.5, 4))))

# gri dreapta jos
possible_distances.append(((9.5, 1.5), (10.5, 4), calculate_distance((9.5, 1.5), (10.5, 4))))

# roz varf jos
possible_distances.append(((10.5, 4), (8, 8), calculate_distance((10.5, 4), (8, 8))))
possible_distances.append(((10.5, 4), (11, 9), calculate_distance((10.5, 4), (11, 9))))

# roz stanga
possible_distances.append(((8, 8), (10, 10), calculate_distance((8, 8), (10, 10))))

# roz varf sus
possible_distances.append(((10, 10), (11, 9), calculate_distance((10, 10), (11, 9))))

# possible distances to G(final destination)
possible_distances.append(((10, 10), (12, 10), calculate_distance((10, 10), (12, 10))))
possible_distances.append(((11, 9), (12, 10), calculate_distance((11, 9), (12, 10))))
possible_distances.append(((10.5, 4), (12, 10), calculate_distance((10.5, 4), (12, 10))))

# possible distances from S(robot-first try(0,2))
# possible_distances.append(((0,2),(1,3),calculate_distance((0,2),(1,3))))

for possib_dist in possible_distances:
    possible_distances_without_cost.append((possib_dist[0], possib_dist[1]))

# Plot the figures
fig1_coord = [[1, 1], [5, 1], [5, 3], [1, 3]]
fig1_coord.append(fig1_coord[0])
xs, ys = zip(*fig1_coord)
fig1_segments = [[[1, 1], [5, 1]], [[5, 1], [5, 3]], [[5, 3], [1, 3]], [[1, 3], [1, 1]]]

fig2_coord = [[3, 4], [4, 4], [3.5, 8]]
fig2_coord.append(fig2_coord[0])
a, b = zip(*fig2_coord)
fig2_segments = [[[3, 4], [4, 4]], [[4, 4], [3.5, 8]], [[3.5, 8], [3, 4]]]

fig3_coord = [[2, 4], [2.5, 8], [2, 10], [0.5, 8], [0.8, 4.5]]
fig3_coord.append(fig3_coord[0])
fig3_segments = [[[2, 4], [2.5, 8]], [[2.5, 8], [2, 10]], [[2, 10], [0.5, 8]], [[0.5, 8], [0.8, 4.5]],
                 [[0.8, 4.5], [2, 4]]]
c, d = zip(*fig3_coord)

fig4_coord = [[5, 8], [5, 10], [5.5, 10], [6, 9]]
fig4_coord.append(fig4_coord[0])
fig4_segments = [[[5, 8], [5, 10]], [[5, 10], [5.5, 10]], [[5.5, 10], [6, 9]], [[6, 9], [5, 8]]]
e, f = zip(*fig4_coord)

fig5_coord = [[6, 2], [5.5, 7], [7, 3.5]]
fig5_coord.append(fig5_coord[0])
fig5_segments = [[[6, 2], [5.5, 7]], [[5.5, 7], [7, 3.5]], [[7, 3.5], [6, 2]]]
g, h = zip(*fig5_coord)

fig6_coord = [[6.5, 6.5], [6.5, 9.5], [7, 9.5], [7, 6.5]]
fig6_coord.append(fig6_coord[0])
fig6_segments = [[[6.5, 6.5], [6.5, 9.5]], [[6.5, 9.5], [7, 9.5]], [[7, 9.5], [7, 6.5]], [[7, 6.5], [6.5, 6.5]]]
i, j = zip(*fig6_coord)

fig7_coord = [[8, 8], [10, 10], [11, 9], [10.5, 4]]
fig7_coord.append(fig7_coord[0])
fig7_segments = [[[8, 8], [10, 10]], [[10, 10], [11, 9]], [[11, 9], [10.5, 4]], [[10.5, 4], [8, 8]]]
k, l = zip(*fig7_coord)

fig8_coord = [[8, 1], [9.5, 1.5], [9.5, 3], [8, 4], [7, 3], [7, 2]]
fig8_coord.append(fig8_coord[0])
fig8_segments = [[[8, 1], [9.5, 1.5]], [[9.5, 1.5], [9.5, 3]], [[9.5, 3], [8, 4]], [[8, 4], [7, 3]], [[7, 3], [7, 2]],
                 [[7, 2], [8, 1]]]
m, n = zip(*fig8_coord)

fig_segments = [fig1_segments, fig2_segments, fig3_segments, fig4_segments, fig5_segments, fig6_segments, fig7_segments,
                fig8_segments]
all_segments = sum(fig_segments, [])

# G coordinates
g_x_coordinates = [12]
g_y_coordinates = [10]

nodes = []
figures = [fig1_coord[:-1], fig2_coord[:-1], fig3_coord[:-1], fig4_coord[:-1], fig5_coord[:-1], fig6_coord[:-1],
           fig7_coord[:-1], fig8_coord[:-1]]
for figure in figures:
    for point in figure:
        nodes.append(tuple(point))

nodes_without_duplicates = []
for node in nodes:
    if node not in nodes_without_duplicates:
        nodes_without_duplicates.append(node)


# S coordinates
# s_x_coordinates = [0]
# s_y_coordinates = [2]


# A* Search
# This class represent a graph
class Graph:
    # Initialize the class
    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    # Create an undirected graph by adding symmetric edges
    def make_undirected(self):
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.graph_dict.setdefault(b, {})[a] = dist

    # Add a link from A and B of given distance, and also add the inverse link if the graph is undirected
    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        if not self.directed:
            self.graph_dict.setdefault(B, {})[A] = distance

    # Get neighbors or a neighbor
    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    # Return a list of nodes in the graph
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


# This class represent a node
class Node:
    # Initialize the class
    def __init__(self, name: str, parent: str):
        self.name = name
        self.parent = parent
        self.g = 0  # Distance to start node
        self.h = 0  # Distance to goal node
        self.f = 0  # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name

    # Sort nodes
    def __lt__(self, other):
        return self.f < other.f

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.f))


robot_path = []


# A* search
def astar_search(graph, heuristics, start, end):
    # Create lists for open nodes and closed nodes
    open = []
    closed = []
    # Create a start node and an goal node
    start_node = Node(start, None)
    goal_node = Node(end, None)
    # Add the start node
    open.append(start_node)

    # Loop until the open list is empty
    while len(open) > 0:
        # Sort the open list to get the node with the lowest cost first
        open.sort()
        # Get the node with the lowest cost
        current_node = open.pop(0)
        # Add the current node to the closed list
        closed.append(current_node)

        # Check if we have reached the goal, return the path
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(str(current_node.name) + ': ' + str(current_node.g))
                robot_path.append(current_node.name)
                current_node = current_node.parent
            path.append(str(start_node.name) + ': ' + str(start_node.g))
            # Return reversed path
            return path[::-1]
        # Get neighbours
        neighbors = graph.get(current_node.name)
        # Loop neighbors
        for key, value in neighbors.items():
            # Create a neighbor node
            neighbor = Node(key, current_node)
            # Check if the neighbor is in the closed list
            if (neighbor in closed):
                continue
            # Calculate full path cost
            neighbor.g = current_node.g + graph.get(current_node.name, neighbor.name)
            neighbor.h = heuristics.get(neighbor.name)
            neighbor.f = neighbor.h + neighbor.g
            # Check if neighbor is in open list and if it has a lower f value
            if (add_to_open(open, neighbor) == True):
                # Everything is green, add neighbor to open list
                open.append(neighbor)
    # Return None, no path is found
    return None


# A* search
def astar_search_with_distance_limit(graph, heuristics, start, end, d):
    # Create lists for open nodes and closed nodes
    open = []
    closed = []
    # Create a start node and an goal node
    start_node = Node(start, None)
    goal_node = Node(end, None)
    # Add the start node
    open.append(start_node)

    # Loop until the open list is empty
    while len(open) > 0:
        # Sort the open list to get the node with the lowest cost first
        open.sort()
        # Get the node with the lowest cost
        current_node = open.pop(0)
        if current_node == start_node or current_node.g < d:
            # Add the current node to the closed list
            closed.append(current_node)

            # Check if we have reached the goal, return the path
            if current_node == goal_node:
                path = []
                while current_node != start_node:
                    path.append(str(current_node.name) + ': ' + str(current_node.g))
                    robot_path.append(current_node.name)
                    current_node = current_node.parent
                path.append(str(start_node.name) + ': ' + str(start_node.g))
                # Return reversed path
                return path[::-1]
            # Get neighbours
            neighbors = graph.get(current_node.name)
            # Loop neighbors
            for key, value in neighbors.items():
                # Create a neighbor node
                neighbor = Node(key, current_node)
                # Check if the neighbor is in the closed list
                if (neighbor in closed):
                    continue
                # Calculate full path cost
                neighbor.g = current_node.g + graph.get(current_node.name, neighbor.name)
                neighbor.h = heuristics.get(neighbor.name)
                neighbor.f = neighbor.h + neighbor.g
                # Check if neighbor is in open list and if it has a lower f value
                if (add_to_open(open, neighbor) == True):
                    # Everything is green, add neighbor to open list
                    open.append(neighbor)
    # Return None, no path is found
    return None


# Check if a neighbor should be added to open list
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f > node.f):
            return False
    return True


def on_segment(p, q, r):
    # check if r lies on (p,q)
    if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
        return True
    return False


# check to see if two lines intersect
def orientation(p, q, r):
    # return 0/1/-1 for colinear/clockwise/counterclockwise
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0: return 0
    return 1 if val > 0 else -1


def intersects(seg1, seg2):
    # check if seg1 and seg2 intersect
    p1, q1 = seg1
    p2, q2 = seg2
    o1 = orientation(p1, q1, p2)
    # find all orientations
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        # check general case
        return True

    if o1 == 0 and on_segment(p1, q1, p2): return True
    # check special cases
    if o2 == 0 and on_segment(p1, q1, q2): return True
    if o3 == 0 and on_segment(p2, q2, p1): return True
    if o4 == 0 and on_segment(p2, q2, q1): return True
    return False


def check_if_point_in_shape(point_to_be_checked):
    figure_number = 0
    ok = True
    for figure in figures:
        # print("Figure we are checking in:")
        # print(figure)

        is_in_figure = True
        for node_of_figure in figure:
            # print("Every node of the figure:")
            # print(node_of_figure)
            segments = fig_segments[figure_number]
            # print("Segments of the figure:")
            # print(segments)

            wrong_intersection = False
            for segment in segments:
                segment_one = ((point_to_be_checked[0], point_to_be_checked[1]), (node_of_figure[0], node_of_figure[1]))
                segment_two = ((segment[0][0], segment[0][1]), (segment[1][0], segment[1][1]))
                # print("The two segments to be checked")
                # print(segment_one)
                # print(segment_two)

                line1 = LineString([segment_one[0], segment_one[1]])
                line2 = LineString([segment_two[0], segment_two[1]])
                point_of_intersection = line1.intersection(line2)
                # print("Point of intersection")
                # print(point_of_intersection)

                # print(intersects(segment_one, segment_two))
                if intersects(segment_one, segment_two):
                    # print("Intersects!!!!!")
                    found = False
                    for point_of_segment_two in segment_two:
                        # print("Point of segment we are checking with")
                        # print(point_of_segment_two)
                        for point_of_segment_one in segment_one:
                            if point_of_segment_two == point_of_segment_one:
                                # print("Not wrongly intersected")
                                found = True
                    if found == False:
                        # print("wrongly intersected")
                        wrong_intersection = True
                        break
                else:
                    # print("They don't intersect")
                    continue

            if wrong_intersection == True:
                is_in_figure = False
                break
        if is_in_figure:
            return True
        figure_number += 1
    return False
    #   if ok == True:
    #     return ok
    # return False


# print("Is in shape? : " + str(check_if_point_in_shape((4, 9))))

# Run the search algorithm
# print("START POINTS")

def generate_search_algorithm(start_point, with_error, subgoal_point):
    global robot_path

    if not with_error:
        connections = []
        # Sort connections from all nodes to generated start node by distance
        for node in nodes_without_duplicates:
            distance = calculate_distance(node, start_point)
            connections.append([distance, node])

        ok = True
        connections.sort()
        # print(connections)
        for connection in connections:
            # print(connection)
            for segment in all_segments:
                segment_one = (start_point, connection[1])
                segment_two = ((segment[0][0], segment[0][1]), (segment[1][0], segment[1][1]))
                line1 = LineString([start_point, connection[1]])
                line2 = LineString([(segment[0][0], segment[0][1]), (segment[1][0], segment[1][1])])
                point_of_intersection = line1.intersection(line2)

                if intersects(segment_one, segment_two):
                    if point_of_intersection.x != connection[1][0] or point_of_intersection.y != connection[1][1]:
                        ok = False

                if ok == False:
                    break

            if ok == True:
                # print("Start point:")
                # print(start_point)
                # print("Distanta")
                # print(connection[0])
                # print("Nod cu care facem legatura:")
                # print(connection[1])
                possible_distances.append((start_point, connection[1], connection[0]))
                possible_distances_without_cost.append((start_point, connection[1]))
                # print(possible_distances)
                # print("E buna conexiunea")
                break

    # Create a graph
    graph = Graph()
    for pair in possible_distances:
        graph.connect(pair[0], pair[1], pair[2])

    # Make graph undirected, create symmetric connections
    graph.make_undirected()

    heuristics_list = []
    for node in nodes_without_duplicates:
        distance = calculate_distance(start_point, node)
        heuristics_list.append((node, distance))

    # print("Heuristics list: ")
    # print(heuristics_list)
    # print()

    heuristics = {}
    for pair in heuristics_list:
        heuristics[pair[0]] = pair[1]
    heuristics[start_point] = 0
    heuristics[(12, 10)] = calculate_distance([start_point[0], start_point[1]], [12, 10])

    # print("Heuristics: ")
    # print(heuristics)
    # print()

    # print(start_point)
    path = astar_search(graph, heuristics, start_point, (12, 10))
    # path2 = astar_search_with_distance_limit(graph, heuristics, start_point, (12,10), 7)
    # path3 = astar_search_with_distance_limit(graph, heuristics, start_point, (12,10), 10)
    # print("A* Search Result Distance: ")
    print(path)
    # print()

    if with_error:
        possible_distances.pop()
        possible_distances_without_cost.pop()

    possible_distances.pop()
    possible_distances_without_cost.pop()

    # Plot the graph
    if robot_path:
        points = 1000
        print(robot_path)
        print("Points left after A* search:" + str(points - len(robot_path)))
        # print("Graph no." + str(w + 1))
        plt.figure()
        for dist in possible_distances_without_cost:
            p1 = [dist[0][0], dist[0][1]]
            p2 = [dist[1][0], dist[1][1]]
            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]
            plt.plot(x, y, color="black")

        plt.plot(xs, ys, a, b, c, d, e, f, g, h, i, j, k, l, m, n, linewidth=4.0)

        plt.plot([start_point[0], robot_path[len(robot_path) - 1][0]],
                 [start_point[1], robot_path[len(robot_path) - 1][1]], color="yellow");

        for varf in range(len(robot_path) - 1):
            p1 = [robot_path[varf][0], robot_path[varf][1]]
            p2 = [robot_path[varf + 1][0], robot_path[varf + 1][1]]
            x = [p1[0], p2[0]]
            y = [p1[1], p2[1]]
            line = plt.plot(x, y, color="yellow")

        plt.scatter([start_point[0]], [start_point[1]])
        plt.scatter([subgoal_point[0]], [subgoal_point[1]])
        plt.scatter(g_x_coordinates, g_y_coordinates)

        plt.show()
        robot_path = []

    # else:
    #   print("A* failed")


start_point = (0, 0)

# punctul B
# for w in range(10):
#   start_point = (random.random() * 12, random.random() * 10)
#   # Check if start point is inside shape
#   while check_if_point_in_shape(start_point):
#     start_point = (random.random() * 12, random.random() * 10)
#   # print(check_if_point_in_shape(start_point))
#   # print(start_point)
#   generate_search_algorithm(start_point, False, (0,0))


# punctul C
for counter in range(10):
    random_number = random.random() * 10
    start_point = (random.random() * 12, random.random() * 10)
    # Check if start point is inside shape
    while check_if_point_in_shape(start_point):
        start_point = (random.random() * 12, random.random() * 10)

    if 3 < random_number < 11:
        generate_search_algorithm(start_point, False, (0, 0))
    else:
        # Error case 30%
        subgoal = (random.random() * 12, random.random() * 10)
        segment_one = (start_point, subgoal)

        lines_intersected = []
        intersected = False
        for segment in all_segments:
            segment_two = ((segment[0][0], segment[0][1]), (segment[1][0], segment[1][1]))

            line1 = LineString([start_point, subgoal])
            line2 = LineString([(segment[0][0], segment[0][1]), (segment[1][0], segment[1][1])])
            point_of_intersection = line1.intersection(line2)

            if intersects(segment_one, segment_two):
                lines_intersected.append(segment_two)
                intersected = True

        if intersected:
            if len(lines_intersected) == 1:
                line = lines_intersected[0]
                distance_to_one_point = calculate_distance(start_point, line[0])
                distance_to_other_point = calculate_distance(start_point, line[1])

                if distance_to_one_point < distance_to_other_point:
                    possible_distances.append((start_point, line[0], distance_to_one_point))
                else:
                    possible_distances.append((start_point, line[1], distance_to_other_point))
            else:
                list_of_coordinates_distances = []
                for line in lines_intersected:
                    distance_one_coordinate = calculate_distance(start_point, line[0])
                    distance_second_coordinate = calculate_distance(start_point, line[1])
                    list_of_coordinates_distances.append(distance_one_coordinate)
                    list_of_coordinates_distances.append(distance_second_coordinate)

                list_of_coordinates_distances.sort()

                for line in lines_intersected:
                    distance1 = calculate_distance(start_point, line[0])
                    distance2 = calculate_distance(start_point, line[1])
                    if distance1 == list_of_coordinates_distances[0]:
                        possible_distances.append((start_point, line[0], distance1))
                    elif distance2 == list_of_coordinates_distances[0]:
                        possible_distances.append((start_point, line[1], distance2))
            generate_search_algorithm(start_point, True, subgoal)
        else:
            print("Generated subgoal - start node line is actually not intersecting anything.")
            # generate_search_algorithm(start_point, False, (0,0))
