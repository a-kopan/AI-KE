import csv
import heapq
from datetime import datetime
from time import time
import math
FILE_PATH = r"List1/data.csv"
TIME_FORMAT = "%I:%M:%S %p"


class Graph:
    def __init__(self, 
                 max_depth: int = 4,
                 board_size: tuple[int,int] = (5,6)
                ): 
        self.max_depth = max_depth
        self.board_size = board_size
        self.nodes: set[Node] = set()
        self.edges: dict[(Node,Node):Edge] = dict()
    
    def get_edges_paths(self,node):
        return [(node, neighbor) for (start, neighbor) in self.edges.keys() if start == node]
    
class Edge:
    def __init__(self, move):
        self.move = move
        
class Node:
    def __init__(self, game_state):
        self.game_state = game_state
        
    def __eq__(self, other):
        # Check if 'other' is a Node object before comparing attributes
        if isinstance(other, Node):
            return (self.game_state == other.game_state)
        return False
    
class GameState:
    def __init__(self):
        pass

def load_graph() -> Graph:
    graph: Graph = Graph()


def get_or_create_node(graph, game_state) -> Node:
    for node in graph.nodes:
        if node.game_state == game_state:
            return node
    new_node = Node(game_state)
    graph.nodes.add(new_node)
    return new_node

def main():

    graph = load_graph()
    print("Graph loaded")
    
      
if __name__ == '__main__':
    main()