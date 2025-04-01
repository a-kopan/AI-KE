import csv
import heapq
from datetime import datetime, timedelta
from time import time, mktime
import logging 
import math

FILE_PATH = r"List1\data.csv"
TIME_FORMAT = "%I:%M:%S %p"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL + 1)
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
        return f"({self.name}, {self.company})"
        
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
            if t >= time: 
                return t
        #logger.info('NEXT TIME NOT FOUND FOR NODE: %s | AT TIME: %s', self.parent_node_departure, time)
        return None
    
class Graph:
    def __init__(self): 
        self.nodes: set[Node] = set()
        self.edges: dict[(Node,Node):Edge] = dict()
    
    def get_edges_paths(self,node):
        return [(node, neighbor) for (start, neighbor) in self.edges.keys() if start == node]
    
    def get_node_with_name_and_company(self, name, company):
        for node in self.nodes:
            if node.name == name and node.company == company:
                return node
        return None

    def dijkstra_t(self, starting_stop_name_and_company, ending_stop_name_and_company, start_time):
        time_start = time()
        
        starting_stop = None
        ending_stop = None
        
        #Get the starting and ending nodes based on their names
        for node in self.nodes:
            if node.name == starting_stop_name_and_company[0] and node.company == starting_stop_name_and_company[1]:
                starting_stop = node
            if node.name == ending_stop_name_and_company[0] and node.company == ending_stop_name_and_company[1]:
                ending_stop = node
        print(starting_stop.company)
        if not starting_stop or not ending_stop:
            logger.info("STARTING AND/OR ENDING STOP NOT FOUND")
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
                # if neighbor == ending_stop:
                #     arrival_times[ending_stop] = neighbor_arrival_time
                #     break
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
    
    def dijkstra_p(self, starting_stop_name_and_company, ending_stop_name_and_company, start_time):
        time_start = time()
        
        starting_stop = None
        ending_stop = None
        
        # Get the starting and ending nodes based on their names
        for node in self.nodes:
            if node.name == starting_stop_name_and_company[0] and node.company == starting_stop_name_and_company[1]:
                starting_stop = node
            if node.name == ending_stop_name_and_company[0] and node.company == ending_stop_name_and_company[1]:
                ending_stop = node
        
        if not starting_stop or not ending_stop:
            logger.info("STARTING AND/OR ENDING STOP NOT FOUND")
            return None
        
        visited = set()
        
        arrival_times = {node: None for node in self.nodes}
        arrival_times[starting_stop] = start_time
        
        line_changes = {node: float('inf') for node in self.nodes}
        line_changes[starting_stop] = 0
        
        previous = {node: None for node in self.nodes}
        previous_line = {node: None for node in self.nodes}
        
        prio_queue = []
        counter = 0
        heapq.heappush(prio_queue, (0, start_time, counter, starting_stop, None))
        
        while prio_queue:
            current_changes, current_arrival_time, _, current_node, current_line = heapq.heappop(prio_queue)
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_node == ending_stop:
                arrival_times[ending_stop] = current_arrival_time
                break
            
            for edgePath in self.get_edges_paths(current_node):
                edge = self.edges.get(edgePath, None)
                if not edge:
                    continue
                
                neighbor = edgePath[1]
                departure_time = edge.get_closest_time_after_given(current_arrival_time)
                if departure_time is None:
                    continue
                
                neighbor_arrival_time = departure_time + edge.time_cost
                new_line_changes = current_changes + (1 if edge.line != current_line else 0)
                
                if (new_line_changes < line_changes[neighbor] or
                    (new_line_changes == line_changes[neighbor] and neighbor_arrival_time < arrival_times[neighbor])):
                    
                    line_changes[neighbor] = new_line_changes
                    arrival_times[neighbor] = neighbor_arrival_time
                    previous[neighbor] = current_node
                    previous_line[neighbor] = edge.line
                    counter += 1
                    heapq.heappush(prio_queue, (new_line_changes, neighbor_arrival_time, counter, neighbor, edge.line))
        
        if arrival_times[ending_stop] is None:
            return None, []
        
        path = []
        current_node = ending_stop
        while current_node is not None:
            path.insert(0, current_node)
            current_node = previous.get(current_node)
        
        total_time = arrival_times[ending_stop] - start_time
        
        final_string = f"""
        Starting stop: {starting_stop}
        Ending stop: {ending_stop}
        Starting time: {start_time.time()}
        Ending time: {arrival_times[ending_stop].time()}
        Total time: {total_time}
        Line changes: {line_changes[ending_stop]}
        Path taken: {print_path(path)}
        """
        
        return final_string

    def dijkstra(self, starting_stop, ending_stop, optimization_criterion, start_time):
        if optimization_criterion == 't':
            return self.dijkstra_t(starting_stop, ending_stop, start_time)
        elif optimization_criterion == 'p':
            return self.dijkstra_p(starting_stop, ending_stop, start_time)
        else:
            raise ValueError("Invalid optimization criterion")
        
    def heuristic(self, node: Node, goal: Node, average_speed: float = 30) -> float:
        lat1, lon1 = math.radians(float(node.lat)), math.radians(float(node.lon))
        lat2, lon2 = math.radians(float(goal.lat)), math.radians(float(goal.lon))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        R = 6371
        distance = R * c  # distance in km
        estimated_time_sec = (distance / average_speed) * 3600
        return estimated_time_sec

    def a_star_t(self, starting_stop_name_and_company: str, ending_stop_name_and_company: str, start_time: datetime):
        for node in self.nodes:
            if node.name == starting_stop_name_and_company[0] and node.company == starting_stop_name_and_company[1]:
                starting_stop = node
            if node.name == ending_stop_name_and_company[0] and node.company == ending_stop_name_and_company[1]:
                goal_stop = node
                
        if not starting_stop or not goal_stop:
            logger.info("STARTING AND/OR ENDING STOP NOT FOUND")
            return None
        
        visited = set()
        arrival_times = {node: None for node in self.nodes}
        arrival_times[starting_stop] = start_time
        
        previous = {node: None for node in self.nodes}
        prio_queue = []
        counter = 0
        
        start_ts = start_time.timestamp()
        h_start = self.heuristic(starting_stop, goal_stop)
        f_start = start_ts + h_start
        heapq.heappush(prio_queue, (f_start, counter, starting_stop))
        
        while prio_queue:
            f, _, current_node = heapq.heappop(prio_queue)
            if current_node == goal_stop:
                break
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            for edgePath in self.get_edges_paths(current_node):
                edge: Edge = self.edges.get(edgePath)
                if not edge:
                    continue
                neighbor = edgePath[1]
                departure_time = edge.get_closest_time_after_given(arrival_times[current_node])
                if departure_time is None:
                    continue
                neighbor_arrival_time = departure_time + edge.time_cost
                
                if (arrival_times[neighbor] is None or 
                    neighbor_arrival_time < arrival_times[neighbor]):
                    arrival_times[neighbor] = neighbor_arrival_time
                    previous[neighbor] = current_node
                    neighbor_arrival_ts = neighbor_arrival_time.timestamp()
                    h_neighbor = self.heuristic(neighbor, goal_stop)
                    f_neighbor = neighbor_arrival_ts + h_neighbor
                    counter += 1
                    heapq.heappush(prio_queue, (f_neighbor, counter, neighbor))
        
        if arrival_times[goal_stop] is None:
            return None, []
        
        path = []
        current_node = goal_stop
        while current_node is not None:
            path.insert(0, current_node)
            current_node = previous[current_node]
        
        total_time = arrival_times[goal_stop] - start_time
        final_string = (
            f"Starting stop: {starting_stop}\n"
            f"Ending stop: {goal_stop}\n"
            f"Starting time: {start_time.time()}\n"
            f"Ending time: {arrival_times[goal_stop].time()}\n"
            f"Total time: {total_time}\n"
            f"Path taken: {print_path(path)}\n"
        )
        
        return final_string

def load_graph() -> Graph:
    graph: Graph = Graph()

    with open(FILE_PATH) as f:
        reader = csv.reader(f)
        next(reader) 

        for row in reader:
            _, _, company, line, departure_time, arrival_time, start_station, end_station, start_station_lat, start_station_lon, end_station_lat, end_station_lon = row

            departure_time = datetime.strptime(departure_time, TIME_FORMAT).replace(year = 2000)
            arrival_time = datetime.strptime(arrival_time, TIME_FORMAT).replace(year = 2000)

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

    graph = load_graph()
    start_station = ("KRZYKI", "MPK Autobusy")
    end_station = ("Bukowskiego", "MPK Autobusy")
    start_time = datetime.strptime("7:59:00 PM", TIME_FORMAT).replace(year = 2000)
    print("Graph loaded")
    
    
    print("Dijkstra with t parameter: ")
    #ans = graph.dijkstra(start_station, end_station, 't', start_time)
    #ans = graph.a_star_t(start_station, end_station, start_time)
    ans = graph.dijkstra(start_station, end_station, 'p', start_time)
    print(ans)
    
    
if __name__ == '__main__':
    main()