import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import heapq

from PIL import Image
from bresenham import bresenham

def load_map(filename):
    img = Image.open(filename).convert("RGB")
    
    pixels = np.array(img)
    cost_map = np.ones((pixels.shape[0:2]))

    for i in range(cost_map.shape[0]):
        for j in range(cost_map.shape[1]):
            if np.all(pixels[i, j, :] == [0, 0, 0]):
                cost_map[i, j] = 100
            elif np.all(pixels[i, j, :] == [0, 0, 255]):
                cost_map[i, j] = 10
            else:
                pass

    return img, pixels, cost_map

def load_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_json_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    return None

def plot_graph(G: nx.Graph, figsize=(6,6), with_labels=False, node_size=10, 
               xlims=None, ylims=None, img=None):
    plt.figure(figsize=figsize)
    if img is not None:
        plt.imshow(img)
        pos = {(x, y): (y,-x+img.size[1]-1.0) for x,y in G.nodes()}
    else:
        pos = {(x, y): (y,-x) for x,y in G.nodes()}
    nx.draw(G, pos=pos, node_color="lightgreen", 
            with_labels=with_labels, node_size=node_size)
    if xlims is not None:
        plt.xlim(*xlims)
    if ylims is not None:
        plt.ylim(*ylims)
    
    plt.show()

def plot_path_on_map(img, G, path):
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    pos = {(x, y): (y,x) for x,y in G.nodes()}
    nx.draw(G, pos=pos, node_color="lightgreen", 
            with_labels=False, node_size=10)
    plt.plot(*([p[1] for p in path], [p[0] for p in path]), linewidth=3, marker="o")

    plt.show()

def expand_path_with_bresenham(coarse_path):
    full_path = []
    for i in range(len(coarse_path)-1):
        u = coarse_path[i]
        v = coarse_path[i+1]
        line = list(bresenham(u[0], u[1], v[0], v[1]))
        if i == 0:
            full_path.extend(line)
        else:
            full_path.extend(line[1:])
    return full_path

def get_neighbors(position: tuple[int,int], cost_map: np.ndarray, allow_diagonal_movements: bool = False, shelves_are_impassable: bool = False):
    x, y = position
    if shelves_are_impassable and not cost_map[x, y] < 100:
        return
    height, width = cost_map.shape
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    if allow_diagonal_movements:
        dirs += [(-1,-1), (-1,1), (1,-1), (1,1)]
    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            if shelves_are_impassable:
                if cost_map[nx, ny] < 100:
                    yield (nx, ny), cost_map[nx, ny]
            else:
                yield (nx, ny), cost_map[nx, ny]

def build_graph(cost_map: np.ndarray, allow_diagonal_movement: bool = False, shelves_are_impassable: bool = False):
    width, height = cost_map.shape
    import networkx as nx

    G = nx.grid_2d_graph(width, height)

    for e in G.edges(): G.remove_edge(*e)

    for u in G.nodes():
        nbrs = [n for n,c in get_neighbors(u, cost_map, allow_diagonal_movements=allow_diagonal_movement, shelves_are_impassable=shelves_are_impassable)]
        for v in nbrs:
            G.add_edge(u, v)

    G.remove_nodes_from(list(nx.isolates(G)))

    return G

def reconstruct_path(came_from, start, goal):
    if goal not in came_from:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path

def dijkstra(G: nx.Graph, cost_map, start, goal):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current == goal:
            break

        for neighbor in G.neighbors(current):
            x, y = neighbor
            move_cost = cost_map[x, y]
            new_cost = current_cost + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                came_from[neighbor] = current

    path = reconstruct_path(came_from, start, goal)
    return path, cost_so_far.get(goal, np.inf), len(cost_so_far)