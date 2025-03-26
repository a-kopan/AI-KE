import csv
import heapq
from datetime import datetime
from time import time

FILE_PATH = r"List1\data.csv"
TIME_FORMAT = "%I:%M:%S %p"


class Node:
    def __init__(self, name, company, lat, lon):
        self.name = name
        self.company = company
        self.lat = lat
        self.lon = lon
        
    def __eq__(self, other):
        # Check if 'other' is a Node object before comparing attributes
        if isinstance(other, Node):
            return (self.name == other.name and self.company == other.company)
        return False  # If 'other' is not a Node, they are not equal

    def __hash__(self):
        # Return a hash value based on node's unique attributes
        return hash((self.name, self.company))

    def __repr__(self):
        return f"Node({self.name}, {self.company})"
        
class Edge:
    def __init__(self, line, time_cost):
        self.time_cost = time_cost
        self.line = line
        self.all_times = []
        
    def add_time(self, time:datetime):
        self.all_times.append(time)

    def get_closest_time_after_given(self, time:datetime):
        for t in sorted(self.all_times):
            if t > time: return t
        return None
        
class Graph:
    def __init__(self): 
        self.nodes = set()
        self.edges = dict()
        
    def get_neighbors(self, node):
        return [neighbor for neighbor in self.nodes if self.edges.get((node, neighbor), False)]
            
    def get_node_with_name_and_company(self, name, company):
        for node in self.nodes:
            if node.name == name and node.company == company:
                return node
        return None
    
    def dijkstra_t(self, starting_stop_name, ending_stop_name, start_time):
        time_start = time()
        
        starting_stop = None
        ending_stop = None
        for node in self.nodes:
            if node.name == starting_stop_name:
                starting_stop = node
            if node.name == ending_stop_name:
                ending_stop = node
        
        # Validate that both nodes exist
        if starting_stop is None:
            raise ValueError(f"Starting stop '{starting_stop_name}' not found in the graph")
        if ending_stop is None:
            raise ValueError(f"Ending stop '{ending_stop_name}' not found in the graph")

        visited = set()
        # Use a dictionary to store the best (minimum) arrival time to each node
        arrival_times = {node: None for node in self.nodes}
        arrival_times[starting_stop] = start_time
        
        # Track previous nodes to reconstruct the path
        previous = {node: None for node in self.nodes}
        
        # Priority queue to store (arrival_time, unique_id, node)
        prio_queue = []
        # Use a counter to create a unique identifier for each queue entry
        counter = 0
        heapq.heappush(prio_queue, (start_time, counter, starting_stop))
        
        while prio_queue:
            current_arrival_time, _, current_node = heapq.heappop(prio_queue)
            
            # Skip nodes we've already visited with an earlier or equal arrival time
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            # If we've reached the ending stop, we can return
            if current_node == ending_stop:
                break
            
            # Iterate over all neighbors of current_node
            for neighbor in self.get_neighbors(current_node):
                # Find the edge between current_node and neighbor
                edge = self.edges.get((current_node, neighbor))
                if not edge:
                    continue
                
                # Find the next available departure time after current arrival time
                departure_time = edge.get_closest_time_after_given(current_arrival_time)
                if departure_time is None:
                    continue
                
                # Calculate arrival time at the neighbor
                # Use current_arrival_time's date with the new departure and add time cost
                neighbor_arrival_time = datetime.combine(current_arrival_time.date(), departure_time.time()) + edge.time_cost
                
                # Only update if this is a better (earlier) path
                if (arrival_times[neighbor] is None or 
                    neighbor_arrival_time < arrival_times[neighbor]):
                    arrival_times[neighbor] = neighbor_arrival_time
                    previous[neighbor] = current_node
                    # Increment counter to ensure unique sorting
                    counter += 1
                    heapq.heappush(prio_queue, (neighbor_arrival_time, counter, neighbor))
        
        # Reconstruct the path
        path = []
        current_node = ending_stop
        while current_node is not None:
            path.insert(0, current_node)
            current_node = previous.get(current_node)
        
        # If no path was found
        if arrival_times[ending_stop] is None:
            return None, []
        
        # Calculate total travel time
        total_time = arrival_times[ending_stop] - start_time
        
        time_finish = time()
        
        #return total_time, path
        return f"Starting stop: {starting_stop}\nEnding stop: {ending_stop}\nStarting time: {start_time}\nEnding time: {arrival_times[ending_stop]}\nTotal time: {total_time}\nCalculation time: {time_finish - time_start}\nPath taken: {path}\n"
        
    def dijkstra_p(starting_stop, ending_stop, start_time): pass
    
    def dijkstra(self, starting_stop, ending_stop, optimization_criterion, start_time):
        if optimization_criterion == 't':
            return self.dijkstra_t(starting_stop, ending_stop, start_time)
        elif optimization_criterion == 'p':
            return self.dijkstra_p(starting_stop, ending_stop, start_time)
        else:
            raise ValueError("Invalid optimization criterion")
    
def load_graph() -> Graph:
    
    graph: Graph = Graph()
    
    with open(FILE_PATH) as f:
        reader = csv.reader(f)
        #Ignore header row
        next(reader)
        
        for row in reader:
            i0, i1, \
            company, line, \
            departure_time, arrival_time, \
            start_station, end_station, \
            start_station_lat, start_station_lon, \
            end_station_lat, end_station_lon = row \
            = row
        
            departure_time = datetime.strptime(departure_time, TIME_FORMAT)
            arrival_time = datetime.strptime(arrival_time, TIME_FORMAT)

            starting_node = Node(start_station, company, start_station_lat, start_station_lon)
            finish_node = Node(end_station, company, end_station_lat, end_station_lon)
            
            if starting_node not in graph.nodes: graph.nodes.add(starting_node)
            if finish_node not in graph.nodes: graph.nodes.add(finish_node)
            
            edge = Edge(line, arrival_time - departure_time)
            
            if graph.edges.get((starting_node, finish_node), False):
                graph.edges[(starting_node, finish_node)].add_time(departure_time)
            else:
                graph.edges[(starting_node, finish_node)] = edge
                
    return graph