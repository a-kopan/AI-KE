import csv
import heapq
from datetime import datetime, timedelta
from time import time
import logging 
import math

FILE_PATH = r"List1\data.csv"
TIME_FORMAT = "%I:%M:%S %p"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
#logging.disable(logging.CRITICAL + 1)
logger.info('Started')
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
        return f"({self.name})"
        
class Edge:
    def __init__(self, line, time_cost, self_parent_node_departure):
        self.time_cost = time_cost
        self.line = line
        self.all_times = []
        self.parent_node_departure = self_parent_node_departure
        
    def add_time(self, time:datetime):
        self.all_times.append(time)

    def get_closest_time_after_given(self, time:datetime):
        sorted_times = sorted(self.all_times)
        for t in sorted_times:
            if t > time: 
                return t
        #logger.info('NEXT TIME NOT FOUND FOR NODE: %s | AT TIME: %s', self.parent_node_departure, time)
        return None
    
class Graph:
    def __init__(self): 
        self.nodes: set[Node] = set()
        self.edges: dict[(Node,Node):Edge] = dict()
    
    def get_edges_paths(self,node):
        #return [(node, neighbor) for neighbor in self.nodes if self.edges.get((node, neighbor), False)]
        return [(node, neighbor) for (start, neighbor) in self.edges.keys() if start == node]
    
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

        if not starting_stop or not ending_stop:
            logger.log("STARTING AND/OR ENDING STOP NOT FOUND")
            return None
        
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
            logger.info("-------------")
            logger.info("CURRENT CONSIDERED NODE: %s", current_node.name)
            if current_node in visited:
                logger.info("NODE %s WAS ALREADY VISITED.", current_node.name)
                continue
            logger.info("ADDED NODE %s TO VISITED", current_node.name)
            visited.add(current_node)
            
            # If we've reached the ending stop, we can return
            if current_node == ending_stop:
                arrival_times[ending_stop] = current_arrival_time
                break
            
            # Iterate over all neighbors of current_node
            for edgePath in self.get_edges_paths(current_node):
                # Find the edge between current_node and neighbor
                edge: Edge = self.edges.get(edgePath, None)
                if not edge:
                    continue
                neighbor = edgePath[1]
                logger.info("CONSIDER NEIGHBOR %s", neighbor)
                
                # Find the next available departure time after current arrival time
                logger.info("CURRENT TIME %s", current_arrival_time)
                departure_time = edge.get_closest_time_after_given(current_arrival_time)
                logger.info("CLOSEST DEPARTURE TIME: %s", departure_time)
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
        Starting time: {start_time.time()}\n \
        Ending time: {arrival_times[ending_stop].time()}\n \
        Total time: {total_time}\n \
        Calculation time: {time_finish - time_start}\n \
        Path taken: {print_path(path)}\n"
        
        return final_string
    
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
                edge = Edge(line, arrival_time - departure_time, starting_node)
                graph.edges[(starting_node, finish_node)] = edge
            
            edge.add_time(departure_time)

    return graph

def get_or_create_node(graph, name, company, lat, lon) -> Node:
    for node in graph.nodes:
        if node.name == name and node.company == company:
            return node
    new_node = Node(name, company, lat, lon)
    graph.nodes.add(new_node)
    return new_node

def print_path(path: list[Node]):
    acc = " -> ".join(str(node) for node in path)
    return acc

def main():
    start_station = "KRZYKI"
    end_station = "Solskiego"
    start_time = datetime.strptime("7:58:00 PM", TIME_FORMAT)

    graph = load_graph()
    print("Graph loaded")
    
    
    print("Dijkstra with t parameter: ")
    #ans = graph.dijkstra(start_station, end_station, 't', start_time)
    ans = graph.a_star_t(start_station, end_station, start_time)
    print(ans)
    
    
if __name__ == '__main__':
    main()