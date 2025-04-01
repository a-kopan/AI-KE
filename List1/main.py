import pandas as pd
import networkx as nx
import heapq
from datetime import datetime, timedelta
import math

TIME_FORMAT = "%I:%M:%S %p"

def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def load_graph(csv_path):
    df = pd.read_csv(csv_path, low_memory=False, encoding='latin-1')
    G = nx.Graph()
    for index, row in df.iterrows():
        start_stop = row['start_stop']
        end_stop = row['end_stop']
        travel_time = (datetime.strptime(row['arrival_time'], TIME_FORMAT) - 
                       datetime.strptime(row['departure_time'], TIME_FORMAT)).total_seconds()
        distance_between_stops = haversine(row['start_stop_lat'], row['start_stop_lon'],
                                           row['end_stop_lat'], row['end_stop_lon'])
        G.add_edge(start_stop, end_stop, travel_time=travel_time, line=row['line'], 
                   company=row['company'], distance_between_stops=distance_between_stops, 
                   arrival_time=row['arrival_time'], departure_time=row['departure_time'])
    return G

def dijkstra(graph, start, end):
    pq = [(0, start, [])]
    visited = set()
    while pq:
        (travel_time, node, path) = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return travel_time, path
            for neighbor in graph[node]:
                travel_time_to_neighbor = graph[node][neighbor]['travel_time']
                if neighbor not in visited:
                    heapq.heappush(pq, (travel_time + travel_time_to_neighbor, neighbor, path))
    return None

def dijkstra_transfers(graph, start, end):
    pq = [(0, start, [], set())]
    visited = set()
    while pq:
        (transfers, node, path, visited_lines) = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return transfers, path
            for neighbor in graph[node]:
                line = graph[node][neighbor]['line']
                if line not in visited_lines:
                    visited_lines.add(line)
                    if neighbor not in visited:
                        heapq.heappush(pq, (transfers + 1, neighbor, path, visited_lines))
                    visited_lines.remove(line)
    return None

def astar(graph, start, end):
    pq = [(0, start, [])]
    visited = set()
    while pq:
        (f, node, path) = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return path
            for neighbor in graph[node]:
                travel_time_to_neighbor = graph[node][neighbor]['travel_time']
                distance_to_neighbor = graph[node][neighbor]['distance_between_stops']
                if neighbor not in visited:
                    g = f + travel_time_to_neighbor
                    h = distance_to_neighbor
                    heapq.heappush(pq, (g + h, neighbor, path))
    return None

def astar_transfer(graph, start, end):
    pq = [(0, start, [], set())]
    visited = set()
    while pq:
        (f, node, path, visited_lines) = heapq.heappop(pq)
        if node not in visited:
            visited.add(node)
            path = path + [node]
            if node == end:
                return len(visited_lines) - 1, path
            for neighbor in graph.neighbors(node):
                line = graph[node][neighbor]['line']
                if line not in visited_lines:
                    visited_lines.add(line)
                    travel_time_to_neighbor = graph[node][neighbor]['travel_time']
                    distance_to_neighbor = graph[node][neighbor]['distance_between_stops']
                    if neighbor not in visited:
                        g = f + travel_time_to_neighbor
                        h = distance_to_neighbor
                        heapq.heappush(pq, (g + h, neighbor, path, visited_lines))
                    visited_lines.remove(line)
    return None

def main():
    # Load graph from CSV file
    G = load_graph(r'List1/data.csv')

    start = "KRZYKI"
    end = "Chłodna"
    optimization = "t"
    start_time = "8:00:00 PM"
    
    # Convert start time string to datetime objects
    start_time_dijkstra = datetime.strptime(start_time, TIME_FORMAT)
    start_time_astar = datetime.strptime(start_time, TIME_FORMAT)
    
    # Optimization based on travel time
    if optimization == 't':
        shortest_path_dijkstra = dijkstra(G, start, end)
        shortest_path_astar = astar(G, start, end)
        if shortest_path_dijkstra:
            total_travel_time = shortest_path_dijkstra[0]
            path = shortest_path_dijkstra[1]
            print('\n####################################################################\n')
            print('Shortest path with Dijkstra Algorithm based on travel time:')
            for i in range(len(path) - 1):
                start_stop = path[i]
                end_stop = path[i + 1]
                travel_time = G[start_stop][end_stop]['travel_time']
                line = G[start_stop][end_stop]['line']
                company = G[start_stop][end_stop]['company']
                earliest_time_str = start_time_dijkstra.strftime(TIME_FORMAT)
                end_time_str = (start_time_dijkstra + timedelta(seconds=travel_time)).strftime(TIME_FORMAT)
                print(f'{start_stop} to {end_stop} on line {line} ({company}) from {earliest_time_str} to {end_time_str}')
                start_time_dijkstra += timedelta(seconds=travel_time)
            print(f'Total travel time: {total_travel_time}')
        if shortest_path_astar:
            total_travel_time = 0
            print('\n####################################################################\n')
            print('Shortest path with A* Algorithm based on travel time:')
            for i in range(len(shortest_path_astar) - 1):
                start_stop = shortest_path_astar[i]
                end_stop = shortest_path_astar[i + 1]
                travel_time = G[start_stop][end_stop]['travel_time']
                distance_between_stops = G[start_stop][end_stop]['distance_between_stops']
                line = G[start_stop][end_stop]['line']
                company = G[start_stop][end_stop]['company']
                earliest_time_str = start_time_astar.strftime(TIME_FORMAT)
                end_time_str = (start_time_astar + timedelta(seconds=travel_time)).strftime(TIME_FORMAT)
                print(f'{start_stop} to {end_stop} on line {line} ({company}) from {earliest_time_str} to {end_time_str}')
                start_time_astar += timedelta(seconds=travel_time)
                total_travel_time += travel_time
            print(f'Total travel time: {total_travel_time}')

    # Optimization based on number of transfers
    elif optimization == 'p':
        shortest_path = astar_transfer(G, start, end)
        if shortest_path:
            num_transfers = 0
            total_time = 0
            start_time = datetime.strptime(start_time, TIME_FORMAT)
            print('\n####################################################################\n')
            print('Shortest path with A* Algorithm based on number of transfers:')
            for i in range(len(shortest_path[1]) - 1):
                start_stop = shortest_path[1][i]
                end_stop = shortest_path[1][i + 1]
                travel_time = G[start_stop][end_stop]['travel_time']
                distance_between_stops = G[start_stop][end_stop]['distance_between_stops']
                line = G[start_stop][end_stop]['line']
                company = G[start_stop][end_stop]['company']
                earliest_time_str = start_time.strftime(TIME_FORMAT)
                end_time_str = (start_time + timedelta(seconds=travel_time)).strftime(TIME_FORMAT)
                print(f'{start_stop} to {end_stop} on line {line} ({company}) from {earliest_time_str} to {end_time_str}')
                start_time += timedelta(seconds=travel_time)
                num_transfers += 1
                total_time += travel_time
            print(f'Total travel time: {total_time:.1f} seconds')
            print(f'{num_transfers} total transfers')

if __name__ == "__main__":
    main()
