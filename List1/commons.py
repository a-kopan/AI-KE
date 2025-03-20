import csv
from datetime import datetime

FILE_PATH = "connection_graph.csv"
TIME_FORMAT = "%H:%M:%S"

class Edge:
    def __init__(self, line, cost):
        self.line = line
        self.cost = cost

class Node:
    def __init__(self): pass
        
        
class Graph:
    def __init__(self, n, edges):
        self.adj = [set() for _ in range(n)]
        for u, v in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
            
    def dijkstra_t(): pass
    def dijkstra_p(): pass
    
    def dijkstra(): pass
    
    
def load_graph(starting_time):
    stime = datetime.strptime(starting_time, TIME_FORMAT)
    with open(FILE_PATH) as f:
        reader = csv.reader(f)
        
        for row in reader:
            i0, i1, \
            company, line, \
            departure_time, arrival_time, \
            start_station, end_station, \
            start_station_lat, start_station_lon, \
            end_station_lat, end_station_lon = row \
            = row
            #Initial filtering for most rows before chosen hour
            if departure_time < stime: continue
                