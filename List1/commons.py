import csv
import heapq
from datetime import datetime
from time import time
from logging import Logger

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
        return False

    def __hash__(self):
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
        
        log = Logger('Logger name')
        Logger.setLevel(log, "DEBUG")
        log.log(2 ,sorted(self.all_times))
        log.log(2 ,self.all_times)
        for t in sorted(self.all_times):
            if t > time: return t
        return None
    
class Graph:
    def __init__(self): 
        self.nodes = set()
        self.edges = dict()
    
    def get_edges_paths(self,node):
        return [(node, neighbor) for neighbor in self.nodes if self.edges.get((node, neighbor), False)]
    
    def get_node_with_name_and_company(self, name, company):
        for node in self.nodes:
            if node.name == name and node.company == company:
                return node
        return None
    
    def dijkstra_t(self, starting_stop_name, ending_stop_name, start_time):
        time_start = time()
        
        starting_stop = None
        ending_stop = None
        
        #Get the starting and ending nodes based on their names
        for node in self.nodes:
            if node.name == starting_stop_name:
                starting_stop = node
            if node.name == ending_stop_name:
                ending_stop = node
        
        if starting_stop is None:
            raise ValueError(f"Starting stop '{starting_stop_name}' not found in the graph")
        if ending_stop is None:
            raise ValueError(f"Ending stop '{ending_stop_name}' not found in the graph")

        visited = set()
        
        # Use a dictionary to store the best (minimum) arrival time to each node
        arrival_times = {node: None for node in self.nodes}
        arrival_times[starting_stop] = start_time
        
        previous = {node: None for node in self.nodes}
        
        prio_queue = []
        
        counter = 0
        heapq.heappush(prio_queue, (start_time, counter, starting_stop))
        
        while prio_queue:
            current_arrival_time, _, current_node = heapq.heappop(prio_queue)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # If we've reached the ending stop, we can return
            if current_node == ending_stop:
                arrival_times[ending_stop] = current_arrival_time
                break
            
            # Iterate over all neighbors of current_node
            for edgePath in self.get_edges_paths(current_node):
                # Find the edge between current_node and neighbor
                edge: Edge = self.edges.get(edgePath, None)
                neighbor = edgePath[1]
                if not edge:
                    continue
                
                # Find the next available departure time after current arrival time
                departure_time = edge.get_closest_time_after_given(current_arrival_time)
                if departure_time is None:
                    continue
                
                neighbor_arrival_time = departure_time + edge.time_cost
                
                # Only update if this is a better (earlier) path
                if (arrival_times[neighbor] is None or 
                    neighbor_arrival_time < arrival_times[neighbor]):
                    arrival_times[neighbor] = neighbor_arrival_time
                    previous[neighbor] = current_node
                    # Increment counter to ensure unique sorting
                    counter += 1
                    heapq.heappush(prio_queue, (neighbor_arrival_time, counter, neighbor))
        
        # If no path was found
        if arrival_times[ending_stop] is None:
            return None, []
        
        # Reconstruct the path
        path = []
        current_node = ending_stop
        while current_node is not None:
            path.insert(0, current_node)
            current_node = previous.get(current_node)
        
        # Calculate total travel time
        total_time = arrival_times[ending_stop] - start_time
        
        time_finish = time()
        
        final_string = \
        f" \
        Starting stop: {starting_stop}\n \
        Ending stop: {ending_stop}\n \
        Starting time: {start_time}\n \
        Ending time: {arrival_times[ending_stop]}\n \
        Total time: {total_time}\n \
        Calculation time: {time_finish - time_start}\n \
        Path taken: {print_path(path)}\n"
        
        #return total_time, path
        return final_string
        
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
        next(reader)  # Skip header

        for row in reader:
            _, _, company, line, departure_time, arrival_time, start_station, end_station, start_station_lat, start_station_lon, end_station_lat, end_station_lon = row

            departure_time = datetime.strptime(departure_time, TIME_FORMAT)
            arrival_time = datetime.strptime(arrival_time, TIME_FORMAT)

            starting_node = get_or_create_node(graph, start_station, company, start_station_lat, start_station_lon)
            finish_node = get_or_create_node(graph, end_station, company, end_station_lat, end_station_lon)

            edge = graph.edges.get((starting_node, finish_node), None)
            if edge is None:
                edge = Edge(line, arrival_time - departure_time)
                graph.edges[(starting_node, finish_node)] = edge
            
            edge.add_time(departure_time)

    return graph

def get_or_create_node(graph, name, company, lat, lon):
    for node in graph.nodes:
        if node.name == name and node.company == company:
            return node
    new_node = Node(name, company, lat, lon)
    graph.nodes.add(new_node)
    return new_node

def print_path(path: list[Node]):
    acc = str()
    for node in path:
        acc+= node.name
        acc+= " -> "
    return acc

def main():
    
    start_station = "KRZYKI"
    end_station = "Chłodna"
    start_time = datetime.strptime("8:00:00 PM", TIME_FORMAT)

    graph = load_graph()
    print("Graph loaded")
    
    
    print("Dijkstra with t parameter: ")
    ans = graph.dijkstra(start_station, end_station, 't', start_time)
    print(ans)
    
    
if __name__ == '__main__':
    main()